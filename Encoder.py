import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    将特征图分割为窗口
    
    Args:
        x: [B, H, W, C]
        window_size: 窗口大小
        
    Returns:
        windows: [B*num_windows, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    将窗口重组为特征图
    
    Args:
        windows: [B*num_windows, window_size, window_size, C]
        window_size: 窗口大小
        H, W: 特征图高度和宽度
        
    Returns:
        x: [B, H, W, C]
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    基于窗口的多头自注意力
    """
    
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # 相对位置偏置
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        # 生成相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B*num_windows, window_size*window_size, C]
            mask: 注意力掩码 [num_windows, window_size*window_size, window_size*window_size]
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # 应用掩码（如果有）
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)
        
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # 归一化层
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        # 窗口注意力
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, H, W, C]
            attn_mask: 注意力掩码
        """
        B, H, W, C = x.shape
        
        shortcut = x
        x = x.view(B, H * W, C)
        x = self.norm1(x)
        
        # 转换回空间格式用于窗口分割，保持原始通道数
        x = x.view(B, H, W, C)
        
        # 填充到窗口大小的倍数
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        
        _, Hp, Wp, _ = x.shape
        
        # 循环移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None
        
        # 窗口分割
        x_windows = window_partition(shifted_x, self.window_size)  # [B*num_windows, window_size, window_size, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [B*num_windows, window_size^2, C]
        
        # 窗口注意力
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [B*num_windows, window_size^2, C]
        
        # 窗口重组
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, Hp, Wp, C]
        
        # 逆循环移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        # 移除填充，确保形状正确
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        
        # 转换回序列格式用于残差连接和 MLP
        x = x.view(B, H * W, C)
        shortcut = shortcut.view(B, H * W, C)
        
        # 残差连接
        x = shortcut + self.drop_path(x)
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        # 转换回 [B, H, W, C] 格式
        x = x.view(B, H, W, C)
        
        return x


class PatchMerging(nn.Module):
    """
    Patch Merging 层（下采样）
    """
    
    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W, C]
        Returns:
            x: [B, H/2, W/2, 2*C]
        """
        B, H, W, C = x.shape
        
        # 分割为2x2的patch
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        
        # 拼接
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2, W/2, 2*C]
        
        return x


class Mlp(nn.Module):
    """MLP 模块"""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """随机深度（Stochastic Depth）"""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # 根据输入维度创建正确形状的 random_tensor
        if x.dim() == 4:  # [B, H, W, C]
            random_tensor = keep_prob + torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype)
        elif x.dim() == 3:  # [B, N, C]
            random_tensor = keep_prob + torch.rand(x.shape[0], 1, 1, device=x.device, dtype=x.dtype)
        elif x.dim() == 2:  # [B, C]
            random_tensor = keep_prob + torch.rand(x.shape[0], 1, device=x.device, dtype=x.dtype)
        else:
            random_tensor = keep_prob + torch.rand(x.shape[0], device=x.device, dtype=x.dtype)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class PatchEmbed(nn.Module):
    """
    图像到 Patch Embedding
    """
    
    def __init__(
        self,
        img_size: int = 384,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            x: [B, H/patch_size, W/patch_size, embed_dim]
        """
        B, C, H, W = x.shape
        # 填充到 patch_size 的倍数
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        _, _, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, Hp*Wp, embed_dim]
        x = self.norm(x)
        x = x.view(B, Hp, Wp, -1)  # [B, Hp, Wp, embed_dim]
        
        return x


class SwinEncoder(nn.Module):
    """
    四层 Swin Transformer 编码器
    
    返回四个阶段的特征，用于 HVDW 模块
    """
    
    def __init__(
        self,
        img_size: int = 384,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        patch_norm: bool = True
    ):
        super().__init__()
        
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        
        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        num_patches = self.patch_embed.num_patches
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # 构建每个阶段
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # 随机深度衰减率
        
        self.stages = nn.ModuleList()
        self.patch_merges = nn.ModuleList()
        
        for i_stage in range(self.num_stages):
            stage = nn.ModuleList()
            dim = embed_dim * (2 ** i_stage)
            
            # 添加 Patch Merging（除了第一个阶段）
            # 在阶段 i_stage，输入维度是 embed_dim * (2 ** (i_stage - 1))
            if i_stage > 0:
                input_dim = embed_dim * (2 ** (i_stage - 1))
                patch_merge = PatchMerging(input_dim, norm_layer=norm_layer)
                self.patch_merges.append(patch_merge)
            else:
                self.patch_merges.append(nn.Identity())
            
            # 添加 Swin Transformer Blocks
            for i_layer in range(depths[i_stage]):
                # 交替使用 shift 和 non-shift
                shift_size = 0 if (i_layer % 2 == 0) else window_size // 2
                
                layer = SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads[i_stage],
                    window_size=window_size,
                    shift_size=shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])][i_layer],
                    norm_layer=norm_layer
                )
                stage.append(layer)
            
            self.stages.append(stage)
        
        # 输出投影层（将特征转换回 [B, C, H, W] 格式）
        self.output_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * (2 ** i), embed_dim * (2 ** i)),
                nn.LayerNorm(embed_dim * (2 ** i))
            ) for i in range(self.num_stages)
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            features: 四个阶段的特征列表，每个为 [B, C_s, H_s, W_s]
        """
        # Patch Embedding
        x = self.patch_embed(x)  # [B, H/patch_size, W/patch_size, embed_dim]
        x = self.pos_drop(x)
        
        features = []
        B, H, W, C = x.shape
        
        for i_stage in range(self.num_stages):
            # Patch Merging（除了第一个阶段）
            if i_stage > 0:
                x = self.patch_merges[i_stage](x)
                H, W = H // 2, W // 2
                C = C * 2
            
            # Swin Transformer Blocks
            for layer in self.stages[i_stage]:
                x = layer(x)
            
            # 转换回 [B, C, H, W] 格式
            B, H_s, W_s, C_s = x.shape
            x_stage = x.permute(0, 3, 1, 2).contiguous()  # [B, C_s, H_s, W_s]
            
            # 可选：应用输出投影
            # x_stage = x_stage.view(B, C_s, -1).permute(0, 2, 1)  # [B, H_s*W_s, C_s]
            # x_stage = self.output_projs[i_stage](x_stage)  # [B, H_s*W_s, C_s]
            # x_stage = x_stage.permute(0, 2, 1).view(B, C_s, H_s, W_s)  # [B, C_s, H_s, W_s]
            
            features.append(x_stage)
        
        return features
    
    def get_stage_features(self, x: torch.Tensor, stage: int) -> torch.Tensor:
        """
        获取指定阶段的特征
        
        Args:
            x: 输入图像 [B, C, H, W]
            stage: 阶段索引 (0-3)
            
        Returns:
            feature: 阶段特征 [B, C_s, H_s, W_s]
        """
        features = self.forward(x)
        return features[stage]


def create_swin_encoder(
    in_chans: int = 3,
    img_size: int = 384,
    embed_dim: int = 96,
    depths: List[int] = [2, 2, 6, 2],
    num_heads: List[int] = [3, 6, 12, 24],
    window_size: int = 7,
    **kwargs
) -> SwinEncoder:
    """
    创建 Swin Transformer 编码器的工厂函数
    
    Args:
        in_chans: 输入通道数（1 for thermal, 3 for RGB）
        img_size: 输入图像大小
        embed_dim: 嵌入维度
        depths: 每个阶段的 block 数量
        num_heads: 每个阶段的注意力头数
        window_size: 窗口大小
        
    Returns:
        encoder: Swin Transformer 编码器
    """
    encoder = SwinEncoder(
        img_size=img_size,
        patch_size=4,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        **kwargs
    )
    return encoder
