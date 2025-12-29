import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.act(x)
        return x


class ProjectionToStage4(nn.Module):
    """
    先 1x1 Conv 调整通道，再 resize 到与 R4 相同的空间尺寸。
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        x = self.proj(x)
        x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
        return x


class GuidanceHead(nn.Module):
    """
    预测双向的offset与可选的scale：
    输入: concat([R4, F_mtfe_proj])
    输出通道:
      - 若 use_scale: 2*(2*K + K) = 6K
      - 否则:          2*(2*K)     = 4K
    按顺序切分为 (r->m) 与 (m->r) 两套字段。
    """
    def __init__(self, channels_in: int, k_points: int = 4, use_scale: bool = True):
        super().__init__()
        self.k_points = k_points
        self.use_scale = use_scale
        out_per_dir = 3 * k_points if use_scale else 2 * k_points
        out_channels = out_per_dir * 2
        self.conv1 = DepthwiseSeparableConv(channels_in, channels_in)
        self.conv2 = DepthwiseSeparableConv(channels_in, channels_in)
        self.head = nn.Conv2d(channels_in, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        x = self.conv1(x)
        x = self.conv2(x)
        pred = self.head(x)
        B, C, H, W = pred.shape
        k = self.k_points
        if self.use_scale:
            per_dir = 3 * k
            r2m, m2r = pred[:, :per_dir], pred[:, per_dir:]
            off_r2m = r2m[:, : 2 * k]  # [B, 2K, H, W]
            scl_r2m = r2m[:, 2 * k : 3 * k]  # [B, K, H, W]
            off_m2r = m2r[:, : 2 * k]
            scl_m2r = m2r[:, 2 * k : 3 * k]
        else:
            per_dir = 2 * k
            r2m, m2r = pred[:, :per_dir], pred[:, per_dir:]
            off_r2m = r2m  # [B, 2K, H, W]
            scl_r2m = None
            off_m2r = m2r  # [B, 2K, H, W]
            scl_m2r = None
        return (off_r2m, scl_r2m), (off_m2r, scl_m2r)


class DeformableSampler(nn.Module):
    """
    基于 grid_sample 的通用可变形采样。支持 K 个采样点的平均聚合；可选每点缩放。
    假设 offsets 提供 [B, 2K, H, W]，scales 为 [B, K, H, W] 或 None。
    """
    def __init__(self, k_points: int = 4):
        super().__init__()
        self.k_points = k_points
    
    @staticmethod
    def _make_base_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype),
            indexing='ij'
        )
        base = torch.stack([xx, yy], dim=-1)  # [H, W, 2], x then y
        return base
    
    def forward(self, x: torch.Tensor, offsets: torch.Tensor, scales: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, H, W = x.shape
        K = self.k_points
        assert offsets.shape[1] == 2 * K and offsets.shape[2] == H and offsets.shape[3] == W, "offset shape mismatch"
        if scales is not None:
            assert scales.shape[1] == K and scales.shape[2] == H and scales.shape[3] == W, "scale shape mismatch"
        base = self._make_base_grid(H, W, x.device, x.dtype).unsqueeze(0).expand(B, H, W, 2)  # [B,H,W,2]
        outputs = []
        for i in range(K):
            dx = offsets[:, 2 * i : 2 * i + 1, :, :]
            dy = offsets[:, 2 * i + 1 : 2 * i + 2, :, :]
            # 归一化到 [-1,1] 坐标系的偏移
            off = torch.stack([dx.squeeze(1), dy.squeeze(1)], dim=-1)  # [B, H, W, 2]
            grid = base + off
            y = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
            if scales is not None:
                y = y * torch.sigmoid(scales[:, i : i + 1, :, :])
            outputs.append(y)
        out = torch.stack(outputs, dim=0).mean(dim=0)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = self.act(out + identity)
        return out


class CompressBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.res = ResidualBlock(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1x1(x)
        x = self.res(x)
        return x


class BoundaryHead(nn.Module):
    """
    轻量边界头：3x3 convs + sigmoid
    可选地通过高通近似（梯度）增强。
    """
    def __init__(self, in_channels: int, mid_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1)
        self.sig = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.act(y)
        y = self.conv2(y)
        y = self.sig(y)
        return y


class DecodeFuseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, edge_mid: int = 64):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.edge = BoundaryHead(out_channels, edge_mid)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, up: torch.Tensor, skip: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([up, skip], dim=1)
        x = self.fuse(x)
        bmap = self.edge(x)
        x = x * (1.0 + self.alpha * bmap)
        x = self.act(x)
        return x, bmap


def tv_l1_smoothness(offsets: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(offsets[:, :, :, 1:] - offsets[:, :, :, :-1]).mean()
    dy = torch.abs(offsets[:, :, 1:, :] - offsets[:, :, :-1, :]).mean()
    return dx + dy


class BAEF(nn.Module):
    """
    Bi-directional Alignment-and-Fusion with Edge-Enhanced Decoding (BAEF)

    输入：
      - R4: RGB 第4阶段特征 [B, C4, H4, W4]
      - F_mtfe: MTFE 输出（阶段无关）[B, Ct, Ht, Wt]
      - F_hvdw: HVDW 动态窗口上下文 [B, Ch, H4, W4]
      - skips: 编码器跳连特征列表 [S0, S1, S2, S3]，其中 S3 尺寸与 R4 相同

    输出：
      - dict 包含：
        - P: 最终显著性图（与 S0 同分辨率）
        - F_fuse: 融合后的顶层特征
        - losses: 可选训练损失（cycle 与 smooth）
        - aux: 中间产物（对齐后的 R 与 T，边界图列表）
    """
    def __init__(
        self,
        c4: int,
        ch: int,
        cf: int = 256,
        ct: Optional[int] = None,
        k_points: int = 4,
        use_scale: bool = True,
        lambda_cycle: float = 0.0,
        lambda_smooth: float = 0.0,
        decoder_channels: List[int] = [256, 128, 64],
        edge_mid: int = 64,
    ):
        super().__init__()
        self.c4 = c4
        self.ch = ch
        self.cf = cf
        self.k_points = k_points
        self.use_scale = use_scale
        self.lambda_cycle = lambda_cycle
        self.lambda_smooth = lambda_smooth
        self.ct = ct
        
        # 投影到 stage-4
        if ct is None:
            # 延迟构造（首次前向根据输入通道建立）
            self.proj_mtfe = None
        else:
            self.proj_mtfe = ProjectionToStage4(ct, c4)
        
        # 引导头 Γ：从 [R4||F_mtfe_proj] 预测双向偏移与缩放
        self.gamma = GuidanceHead(channels_in=c4 * 2, k_points=k_points, use_scale=use_scale)
        self.sampler = DeformableSampler(k_points=k_points)
        
        # 融合压缩 φ：concat(HVDW, aligned R, aligned T) → Cf
        in_cat = ch + 2 * c4
        self.compress = CompressBottleneck(in_cat, cf)
        
        # 解码器（自顶向下 3 次，使用 S2、S1、S0）
        assert len(decoder_channels) == 3, "decoder_channels 应为3个阶段"
        self.up1 = nn.ConvTranspose2d(cf, decoder_channels[0], kernel_size=2, stride=2)
        self.dec1 = DecodeFuseBlock(decoder_channels[0] + (self.c4 // 2), decoder_channels[0], edge_mid=edge_mid)
        
        self.up2 = nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1], kernel_size=2, stride=2)
        self.dec2 = DecodeFuseBlock(decoder_channels[1] + (self.c4 // 4), decoder_channels[1], edge_mid=edge_mid)
        
        self.up3 = nn.ConvTranspose2d(decoder_channels[1], decoder_channels[2], kernel_size=2, stride=2)
        # S0 通道未知，假定为 c4//8（与常见金字塔一致）；若不同，可在前向中自适应 1x1 调整
        self.dec3 = DecodeFuseBlock(decoder_channels[2] + max(8, self.c4 // 8), decoder_channels[2], edge_mid=edge_mid)
        
        self.out_head = nn.Conv2d(decoder_channels[2], 1, kernel_size=1)
        self.out_sig = nn.Sigmoid()
    
    def _ensure_proj(self, f_mtfe: torch.Tensor, target_hw: Tuple[int, int]) -> nn.Module:
        if self.proj_mtfe is None:
            self.ct = f_mtfe.shape[1]
            self.proj_mtfe = ProjectionToStage4(self.ct, self.c4).to(f_mtfe.device)
        return self.proj_mtfe
    
    def _maybe_adapt_skip(self, skip: torch.Tensor, expect_channels: int) -> torch.Tensor:
        if skip.shape[1] == expect_channels:
            return skip
        # 动态添加通道适配器，避免在 __init__ 中依赖外部编码器实现细节
        key = f"skip_adapter_{skip.shape[1]}_to_{expect_channels}"
        if not hasattr(self, key):
            adapter = nn.Conv2d(skip.shape[1], expect_channels, kernel_size=1).to(skip.device)
            setattr(self, key, adapter)
        adapter = getattr(self, key)
        return adapter(skip)
    
    def _predict_reverse_fields(self, x_pair: torch.Tensor) -> Tuple[Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        # 共享 Γ 的权重（复用同一模块）
        return self.gamma(x_pair)
    
    def forward(
        self,
        R4: torch.Tensor,
        F_mtfe: torch.Tensor,
        F_hvdw: torch.Tensor,
        skips: List[torch.Tensor],
        compute_loss: bool = False
    ) -> Dict[str, torch.Tensor]:
        B, C4, H4, W4 = R4.shape
        assert C4 == self.c4, "R4 通道数与 c4 不一致"
        assert F_hvdw.shape[2:] == (H4, W4), "F_hvdw 空间尺寸应与 R4 相同"
        
        # 1) 将 MTFE 投影到 stage-4
        self._ensure_proj(F_mtfe, (H4, W4))
        F_mtfe_s4 = self.proj_mtfe(F_mtfe, (H4, W4))  # [B, C4, H4, W4]
        
        # 2) 预测双向偏移/缩放，并执行双向可变形采样
        joint = torch.cat([R4, F_mtfe_s4], dim=1)
        (off_r2m, scl_r2m), (off_m2r, scl_m2r) = self.gamma(joint)
        R_aligned_to_m = self.sampler(R4, off_r2m, scl_r2m)               # \hat{R}_{r->m}^4
        M_aligned_to_r = self.sampler(F_mtfe_s4, off_m2r, scl_m2r)        # \hat{F}_{m->r}^4
        
        # 3) 拼接 HVDW 上下文并压缩
        F_cat = torch.cat([F_hvdw, R_aligned_to_m, M_aligned_to_r], dim=1)
        F_fuse = self.compress(F_cat)
        
        # 4) 级联解码（使用 S2、S1、S0 跳连）
        # 期望 skip 通道近似为 [c4//8, c4//4, c4//2, c4]
        assert len(skips) >= 4, "需要提供 4 个阶段跳连 [S0,S1,S2,S3]"
        S0, S1, S2, S3 = skips[0], skips[1], skips[2], skips[3]
        
        # up to stage-3 -> fuse with S2
        U3u = self.up1(F_fuse)
        S2a = self._maybe_adapt_skip(S2, self.c4 // 2)
        U2, B2 = self.dec1(U3u, S2a)
        
        # up to stage-2 -> fuse with S1
        U2u = self.up2(U2)
        S1a = self._maybe_adapt_skip(S1, self.c4 // 4)
        U1, B1 = self.dec2(U2u, S1a)
        
        # up to stage-1 -> fuse with S0
        U1u = self.up3(U1)
        # 自适应 S0 通道（未知），假设目标通道为 c4//8 或最小为8
        S0a = self._maybe_adapt_skip(S0, max(8, self.c4 // 8))
        U0, B0 = self.dec3(U1u, S0a)
        
        P = self.out_sig(self.out_head(U0))
        
        out: Dict[str, torch.Tensor] = {"P": P, "F_fuse": F_fuse}
        aux = {
            "R_aligned_to_m": R_aligned_to_m,
            "M_aligned_to_r": M_aligned_to_r,
            "B_maps": [B2, B1, B0]
        }
        out["aux"] = aux
        
        # 5) 可选：损失（cycle 与 smooth）
        if compute_loss and (self.lambda_cycle > 0 or self.lambda_smooth > 0):
            losses = {}
            if self.lambda_cycle > 0:
                # 使用对齐后的对，预测反向场
                rev_inp1 = torch.cat([R_aligned_to_m, F_mtfe_s4], dim=1)
                (off_m2r_rev, scl_m2r_rev), (off_r2m_rev, scl_r2m_rev) = self._predict_reverse_fields(rev_inp1)
                R_rec = self.sampler(R_aligned_to_m, off_m2r_rev, scl_m2r_rev)
                M_rec = self.sampler(M_aligned_to_r, off_r2m_rev, scl_r2m_rev)
                L_cycle = F.l1_loss(R_rec, R4) + F.l1_loss(M_rec, F_mtfe_s4)
                losses["cycle"] = L_cycle * self.lambda_cycle
            if self.lambda_smooth > 0:
                L_smooth = tv_l1_smoothness(off_r2m) + tv_l1_smoothness(off_m2r)
                losses["smooth"] = L_smooth * self.lambda_smooth
            out["losses"] = losses
        
        return out

