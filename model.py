import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Sequence
from contextlib import nullcontext

try:
    import torch.amp as torch_amp  # type: ignore[attr-defined]
except (ImportError, AttributeError):  # pragma: no cover - 兼容旧版 torch
    torch_amp = None  # type: ignore[assignment]

try:
    from torch.cuda.amp import autocast as cuda_autocast  # type: ignore
except (ImportError, AttributeError):  # pragma: no cover - CPU 或旧版无 AMP
    cuda_autocast = None  # type: ignore[assignment]

from Encoder import create_swin_encoder
from MTFE import MTFE
from HVDW import HVDW
from BAEF import BAEF


class SmallObjectEnhancement(nn.Module):
    """
    小目标增强模块：专门处理小目标检测的特征增强
    """
    def __init__(self, in_channels=96, out_channels=64):
        super().__init__()

        # 小目标特征增强器
        self.small_obj_detector = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # 超分辨率分支：从低分辨率特征恢复小目标细节
        self.super_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels*4, 1),
            nn.PixelShuffle(2),  # 2x上采样
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # 小目标预测头
        self.small_obj_head = nn.Sequential(
            nn.Conv2d(out_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W] - 通常是stage-1或stage-2特征

        # 小目标特征增强
        feat = self.small_obj_detector(x)

        # 超分辨率上采样
        sr_feat = self.super_res(feat)

        # 小目标预测
        small_pred = self.small_obj_head(sr_feat)

        return {
            'enhanced_feat': feat,
            'sr_feat': sr_feat,
            'small_pred': small_pred
        }


class AdaptiveFeatureFusion(nn.Module):
    """
    自适应特征融合：根据小目标存在概率动态融合多模态特征
    """
    def __init__(self, channels=256):
        super().__init__()
        self.channels = channels
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, rgb_feat, thermal_feat):
        # 特征拼接
        fused = torch.cat([rgb_feat, thermal_feat], dim=1)

        # 生成注意力权重 [B, 2, 1, 1]
        weights = self.attention(fused)

        # 加权融合
        w_rgb = weights[:, 0:1]  # [B, 1, 1, 1]
        w_thermal = weights[:, 1:2]  # [B, 1, 1, 1]

        output = rgb_feat * w_rgb + thermal_feat * w_thermal
        return output


class ThermalRGBSaliencyModel(nn.Module):
    """将编码器、MTFE、HVDW 与 BAEF 串联的整体模型。"""

    def __init__(
        self,
        img_size: int = 224,
        embed_dim: int = 96,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        mtfe_kwargs: Optional[Dict[str, Any]] = None,
        hvdw_kwargs: Optional[Dict[str, Any]] = None,
        baef_kwargs: Optional[Dict[str, Any]] = None,
        *,
        enable_amp_inference: bool = True,
        amp_dtype: torch.dtype = torch.float16,
        simplify_for_small_objects: bool = True,  # 为小目标检测简化模型
    ) -> None:
        super().__init__()

        assert len(depths) == 4, "当前实现假设 4 个编码阶段"
        assert len(num_heads) == 4, "num_heads 长度应与 depths 匹配"

        self.simplify_for_small_objects = simplify_for_small_objects

        mtfe_cfg = dict(mtfe_kwargs or {})
        hvdw_cfg = dict(hvdw_kwargs or {})
        baef_cfg = dict(baef_kwargs or {})

        # 编码器：RGB 与热成像共用同样的结构，输入通道不同。
        self.encoder_rgb = create_swin_encoder(
            in_chans=3,
            img_size=img_size,
            embed_dim=embed_dim,
            depths=list(depths),
            num_heads=list(num_heads),
        )
        self.encoder_thermal = create_swin_encoder(
            in_chans=1,
            img_size=img_size,
            embed_dim=embed_dim,
            depths=list(depths),
            num_heads=list(num_heads),
        )

        # Stage-4 通道数（最低分辨率阶段）。
        c4 = embed_dim * (2 ** (len(depths) - 1))

        # MTFE：保持默认超参，允许外部覆盖。
        self.mtfe = MTFE(**mtfe_cfg)

        # HVDW：默认输出与 RGB 第四阶段通道一致，方便与 BAEF 对接。
        if not self.simplify_for_small_objects:
            hvdw_defaults = {
                "K": 5,
                "num_stages": len(depths),
                "dim": c4,
                "dim_stats": 64,
                "window_size_small": 7,
                "window_size_large": 14,
                "tau": 0.5,
                "alpha": 0.5,
                "num_heads": 8,
                "lambda_fusion": 0.5,
            }
            hvdw_defaults.update(hvdw_cfg)
            self.hvdw = HVDW(**hvdw_defaults)
        else:
            # 简化模式：使用简单的特征聚合替代复杂的HVDW
            self.hvdw = None
            self.simple_fusion = nn.Sequential(
                nn.Conv2d(c4 * 2, c4, 1),  # 简单的通道融合
                nn.BatchNorm2d(c4),
                nn.ReLU(inplace=True),
            )

        # BAEF：默认压缩通道 256，可通过参数修改。
        baef_defaults = {
            "c4": c4,
            "ch": c4,  # 在简化模式下使用c4作为通道数
            "cf": 256,
            "k_points": 4,
            "use_scale": True,
            "lambda_cycle": 0.0,
            "lambda_smooth": 0.0,
            "decoder_channels": [256, 128, 64],
            "edge_mid": 64,
        }
        baef_defaults.update(baef_cfg)
        self.baef = BAEF(**baef_defaults)

        # 小目标增强模块 - 使用stage-1特征（中等分辨率，避免stage-2通道太多）
        stage_1_channels = embed_dim * (2 ** 1)  # stage-1: embed_dim * 2
        self.small_obj_enhance = SmallObjectEnhancement(
            in_channels=stage_1_channels,
            out_channels=64
        )

        # 自适应特征融合
        self.adaptive_fusion = AdaptiveFeatureFusion(c4)  # 输入是单模态特征的通道数

        self.enable_amp_inference = enable_amp_inference
        self.amp_dtype = amp_dtype

    def forward(
        self,
        image_rgb: torch.Tensor,
        image_thermal: torch.Tensor,
        compute_loss: bool = False,
        use_amp: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            image_rgb: RGB 输入 [B, 3, H, W]
            image_thermal: 热输入 [B, 1, H, W]
            compute_loss: 是否在 BAEF 中计算额外损失

        Returns:
            dict: 包含显著性预测、融合特征及中间输出
        """
        amp_enabled = use_amp
        if amp_enabled is None:
            amp_enabled = self.enable_amp_inference and not compute_loss

        amp_context = nullcontext()
        if amp_enabled:
            device_type = image_rgb.device.type
            if torch_amp is not None and hasattr(torch_amp, "autocast"):
                amp_context = torch_amp.autocast(device_type, dtype=self.amp_dtype)
            elif device_type == "cuda" and cuda_autocast is not None:
                amp_context = cuda_autocast(dtype=self.amp_dtype)

        with amp_context:
            # 1) 编码阶段特征
            rgb_stages = self.encoder_rgb(image_rgb)
            thermal_stages = self.encoder_thermal(image_thermal)

            # 2) MTFE：使用第四阶段热特征进行门控
            thermal_prior, thermal_mtfe = self.mtfe(image_thermal, thermal_stages[-1])

            # 3) 特征融合：根据模式选择简单或复杂融合
            if not self.simplify_for_small_objects:
                # 复杂模式：使用HVDW进行动态窗口融合
                hvdw_feat, R4_hvdw = self.hvdw(
                    I_T=image_thermal,
                    thermal_stages=thermal_stages,
                    rgb_stages=rgb_stages,
                    encoder_thermal=self.encoder_thermal,
                )
            else:
                # 简化模式：直接拼接RGB和Thermal的stage-4特征并融合
                rgb_s4 = rgb_stages[-1]  # [B, C4, H/16, W/16]
                thermal_s4 = thermal_stages[-1]  # [B, C4, H/16, W/16]
                combined = torch.cat([rgb_s4, thermal_s4], dim=1)  # [B, C4*2, H/16, W/16]
                R4_hvdw = self.simple_fusion(combined)  # [B, C4, H/16, W/16]
                hvdw_feat = R4_hvdw  # 简化模式下hvdw_feat就是R4_hvdw

            # 4) 小目标增强：使用RGB stage-1特征
            small_obj_out = self.small_obj_enhance(rgb_stages[1])  # stage-1: [B, C*2, H/2, W/2]

            # 5) 自适应融合：结合HVDW和MTFE特征
            fused_feat = self.adaptive_fusion(R4_hvdw, thermal_mtfe)

            # 6) BAEF 融合与解码 - 使用增强的融合特征
            baef_out = self.baef(
                R4=fused_feat,  # 使用自适应融合后的特征替代原始R4_hvdw
                F_mtfe=thermal_mtfe,
                F_hvdw=hvdw_feat,
                skips=rgb_stages,
                compute_loss=compute_loss,
            )

        # 附加中间结果，便于分析与损失设计
        baef_out["P_thermal"] = thermal_prior
        baef_out["F_mtfe"] = thermal_mtfe
        baef_out["F_hvdw"] = hvdw_feat
        baef_out["R4"] = R4_hvdw
        baef_out["rgb_stages"] = rgb_stages
        baef_out["thermal_stages"] = thermal_stages

        # 小目标增强结果
        baef_out["small_obj_pred"] = small_obj_out["small_pred"]
        baef_out["small_obj_feat"] = small_obj_out["enhanced_feat"]

        return baef_out


def compute_saliency_loss(
    model_out: Dict[str, Any],
    target_saliency: torch.Tensor,
    *,
    bce_weight: float = 1.0,
    iou_weight: float = 1.0,
    dice_weight: float = 1.0,
    baef_weight: float = 1.0,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    计算显著性预测损失，并叠加 BAEF 在前向过程中产生的额外损失。

    Args:
        model_out: ThermalRGBSaliencyModel.forward 的输出字典。
        target_saliency: 显著性 Ground Truth，[B,1,H,W]。
        bce_weight: BCE 主损失的系数。
        iou_weight: IoU 损失 (1 - IoU) 的系数。
        dice_weight: Dice 损失 (1 - Dice) 的系数。
        baef_weight: 叠加来自 BAEF 的附加损失的权重。
        eps: 数值稳定项。

    Returns:
        dict: 包含 total、bce、iou（若启用）以及 BAEF 附加损失。
    """
    assert "P" in model_out, "model_out 中缺少显著性预测 P"
    pred = model_out["P"]

    # 确保标签尺寸与预测一致
    if target_saliency.shape[-2:] != pred.shape[-2:]:
        target_resized = F.interpolate(
            target_saliency,
            size=pred.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    else:
        target_resized = target_saliency

    if target_resized.shape[1] != pred.shape[1]:
        raise ValueError(f"标签通道数 {target_resized.shape[1]} 与预测 {pred.shape[1]} 不一致")

    # 统一将预测与标签限制在 [0, 1]，避免非归一化导致损失与指标异常
    pred_prob = torch.clamp(pred.float(), min=eps, max=1.0 - eps)
    target_prob = target_resized.float()
    if target_prob.max() > 1.0:
        # 假设标签可能是 0~255，做一次简单归一化
        target_prob = target_prob / 255.0
    target_prob = torch.clamp(target_prob, 0.0, 1.0)

    # 1) BCE 损失：像素级一致性
    loss_bce = F.binary_cross_entropy(pred_prob, target_prob)
    total = bce_weight * loss_bce

    losses: Dict[str, torch.Tensor] = {"bce": loss_bce}

    # 2) IoU 损失：区域重叠
    if iou_weight > 0.0:
        inter = (pred_prob * target_prob).sum(dim=[1, 2, 3])
        union = pred_prob.sum(dim=[1, 2, 3]) + target_prob.sum(dim=[1, 2, 3]) - inter
        iou = ((inter + eps) / (union + eps)).mean()
        loss_iou = 1.0 - iou
        total = total + iou_weight * loss_iou
        losses["iou"] = loss_iou

    # 3) Dice 损失：更关注前景区域与边界
    if dice_weight > 0.0:
        inter = (pred_prob * target_prob).sum(dim=[1, 2, 3])
        pred_sum = pred_prob.sum(dim=[1, 2, 3])
        target_sum = target_prob.sum(dim=[1, 2, 3])
        dice = ((2.0 * inter + eps) / (pred_sum + target_sum + eps)).mean()
        loss_dice = 1.0 - dice
        total = total + dice_weight * loss_dice
        losses["dice"] = loss_dice

    # 4) 叠加来自 BAEF 的内部损失（如 cycle / smooth 等）
    extra_losses = model_out.get("losses")
    if extra_losses:
        baef_total = torch.zeros_like(total)
        for name, val in extra_losses.items():
            losses[f"baef_{name}"] = val
            baef_total = baef_total + val
        total = total + baef_weight * baef_total
        losses["baef_total"] = baef_total

    losses["total"] = total
    return losses


def compute_saliency_loss_simple(
    model_out: Dict[str, Any],
    target_saliency: torch.Tensor,
    *,
    bce_weight: float = 1.0,
    iou_weight: float = 1.0,
    dice_weight: float = 0.0,  # 默认不启用 Dice，只保留 BCE+IoU 结构损失（参考 Samba 的 structure_loss）
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    一个参考 Samba `structure_loss` 设计的「单一」显著性损失：

        Loss = bce_weight * BCE(P, GT) + iou_weight * IoU_Loss(P, GT) (+ 可选 Dice)

    - 只使用最终显著图 P 与 GT，不叠加 BAEF 等内部损失；
    - 结构上对应 Samba 里的 `BCEWithLogitsLoss + IOU`（我们在概率空间上做 BCE + IoU）；
    - 默认只启用 BCE + IoU，两项都是 1.0，Dice 项默认关闭。
    """
    assert "P" in model_out, "model_out 中缺少显著性预测 P"
    pred = model_out["P"]

    # 1) 尺度对齐
    if target_saliency.shape[-2:] != pred.shape[-2:]:
        target_resized = F.interpolate(
            target_saliency,
            size=pred.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    else:
        target_resized = target_saliency

    if target_resized.shape[1] != pred.shape[1]:
        raise ValueError(f"标签通道数 {target_resized.shape[1]} 与预测 {pred.shape[1]} 不一致")

    # 2) 统一归一化到 [0,1]
    pred_prob = torch.clamp(pred.float(), min=eps, max=1.0 - eps)
    target_prob = target_resized.float()
    if target_prob.max() > 1.0:
        # 兼容 0~255 掩码
        target_prob = target_prob / 255.0
    target_prob = torch.clamp(target_prob, 0.0, 1.0)

    losses: Dict[str, torch.Tensor] = {}

    # 3) BCE 损失（像素级整体拟合），对前景做类别不平衡加权
    # 计算前景/背景像素占比，用于自适应平衡权重
    with torch.no_grad():
        fg = target_prob
        bg = 1.0 - target_prob
        fg_sum = fg.sum(dim=[1, 2, 3])
        bg_sum = bg.sum(dim=[1, 2, 3])
        # 防止出现全 0 前景或全 0 背景
        fg_sum = fg_sum + (fg_sum == 0).float()
        bg_sum = bg_sum + (bg_sum == 0).float()
        # 前景权重约为 背景/前景 的比例，限制在 [1, 20] 之间避免过大
        pos_weight = torch.clamp(bg_sum / fg_sum, min=1.0, max=20.0)

    # 按像素实现加权 BCE：- w_pos * y log p - (1-y) log (1-p)
    bce_pos = -pos_weight.view(-1, 1, 1, 1) * target_prob * torch.log(pred_prob + eps)
    bce_neg = -(1.0 - target_prob) * torch.log(1.0 - pred_prob + eps)
    loss_bce = (bce_pos + bce_neg).mean()
    losses["bce"] = loss_bce

    # 4) IoU 损失（和 Samba 的 IOU 模块类似，鼓励前景区域重叠）
    inter = (pred_prob * target_prob).sum(dim=[1, 2, 3])
    union = pred_prob.sum(dim=[1, 2, 3]) + target_prob.sum(dim=[1, 2, 3]) - inter
    iou = ((inter + eps) / (union + eps)).mean()
    loss_iou = 1.0 - iou  # 作为损失项：IoU 越大，loss 越小
    losses["iou"] = loss_iou

    total = bce_weight * loss_bce + iou_weight * loss_iou

    # 5) 可选 Dice（如果你后面想加一点边界约束，可以把 dice_weight > 0）
    if dice_weight > 0.0:
        inter_d = (pred_prob * target_prob).sum(dim=[1, 2, 3])
        pred_sum_d = pred_prob.sum(dim=[1, 2, 3])
        target_sum_d = target_prob.sum(dim=[1, 2, 3])
        dice = ((2.0 * inter_d + eps) / (pred_sum_d + target_sum_d + eps)).mean()
        loss_dice = 1.0 - dice
        losses["dice"] = loss_dice
        total = total + dice_weight * loss_dice

    # 统一用 "loss" 作为总损失键，避免日志中出现 "total" 命名歧义
    losses["loss"] = total
    return losses


def _compute_soft_edge_map(mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    根据显著性掩码近似生成一个"软边界图"，用于监督 BAEF 解码器中的边界分支。

    Args:
        mask: [B,1,H,W]，数值在 [0,1] 或 [0,255]
    Returns:
        edge: [B,1,H,W]，归一化到 [0,1] 的边界强度图
    """
    m = mask.float()
    if m.max() > 1.0:
        m = m / 255.0
    m = torch.clamp(m, 0.0, 1.0)

    # 简单的一阶梯度近似边界：|dx|+|dy|
    dx = m[:, :, :, 1:] - m[:, :, :, :-1]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = m[:, :, 1:, :] - m[:, :, :-1, :]
    dy = F.pad(dy, (0, 0, 0, 1))
    edge = dx.abs() + dy.abs()

    # 归一化到 [0,1]，避免全 0 时除 0
    max_val = edge.max()
    if max_val > 0:
        edge = edge / (max_val + eps)
    return edge


def _compute_small_object_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    计算小目标专门损失：针对小目标区域的加权BCE损失

    Args:
        pred: 预测显著图 [B,1,H,W]
        target: GT显著图 [B,1,H,W]
    Returns:
        小目标损失标量
    """
    # 计算每个样本的小目标占比
    target_binary = (target >= 0.5).float()
    fg_pixels = target_binary.sum(dim=[1,2,3])  # [B]
    total_pixels = target.numel() / target.shape[0]  # 标量
    small_obj_ratio = fg_pixels / total_pixels  # [B]

    # 小目标阈值：前景占比小于5%认为是小目标场景
    is_small_obj = (small_obj_ratio < 0.05).float()  # [B]

    # 对小目标样本加权损失
    bce_loss = F.binary_cross_entropy(pred, target, reduction='none')  # [B,1,H,W]
    bce_loss = bce_loss.mean(dim=[1,2,3])  # [B]

    # 小目标样本权重更高
    weights = 1.0 + is_small_obj * 2.0  # 小目标权重为3.0，正常为1.0
    weighted_loss = (bce_loss * weights).mean()

    return weighted_loss


def _compute_boundary_precision_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    计算边界精确性损失：使用Sobel算子提取边界并计算MSE

    Args:
        pred: 预测显著图 [B,1,H,W]
        target: GT显著图 [B,1,H,W]
    Returns:
        边界损失标量
    """
    # Sobel算子
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred.device).view(1,1,3,3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=pred.device).view(1,1,3,3)

    # 计算预测边界
    pred_x = F.conv2d(pred, sobel_x, padding=1)
    pred_y = F.conv2d(pred, sobel_y, padding=1)
    pred_edge = torch.sqrt(pred_x**2 + pred_y**2 + eps)

    # 计算GT边界
    target_x = F.conv2d(target, sobel_x, padding=1)
    target_y = F.conv2d(target, sobel_y, padding=1)
    target_edge = torch.sqrt(target_x**2 + target_y**2 + eps)

    # 边界MSE损失
    boundary_loss = F.mse_loss(pred_edge, target_edge)
    return boundary_loss


def compute_saliency_loss_small_objects(
    model_out: Dict[str, Any],
    target_saliency: torch.Tensor,
    *,
    bce_weight: float = 1.0,
    iou_weight: float = 1.0,
    dice_weight: float = 0.5,
    edge_weight: float = 1.0,
    small_obj_weight: float = 2.0,  # 小目标损失权重
    boundary_weight: float = 0.8,   # 边界精确性损失权重
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    面向“小目标/点目标”的增强版显著性损失：

        1) 继承 compute_saliency_loss_simple 的结构（BCE + IoU + 可选 Dice）；
        2) 额外利用 BAEF 解码器输出的多尺度边界图 B2/B1/B0，
           用 GT 掩码生成的软边界图进行监督；
        3) 新增小目标专门损失：对小目标样本加权监督；
        4) 新增边界精确性损失：使用Sobel算子监督边界质量；
        5) 在小目标场景下，边界往往更关键，通过边界分支约束可以提升定位与轮廓质量。
    """
    # 先计算基础的结构损失（BCE + IoU + 可选 Dice）
    base_losses = compute_saliency_loss_simple(
        model_out,
        target_saliency,
        bce_weight=bce_weight,
        iou_weight=iou_weight,
        dice_weight=dice_weight,
        eps=eps,
    )
    total = base_losses["loss"]

    # 1) 小目标专门损失
    if "small_obj_pred" in model_out:
        small_pred = model_out["small_obj_pred"]
        if small_pred.shape[-2:] != target_saliency.shape[-2:]:
            small_pred = F.interpolate(
                small_pred,
                size=target_saliency.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        loss_small_obj = _compute_small_object_loss(small_pred, target_saliency, eps=eps)
        total = total + small_obj_weight * loss_small_obj
        base_losses["small_obj"] = loss_small_obj

    # 2) 边界精确性损失
    pred_main = model_out["P"]
    # 确保预测和目标尺寸一致
    if target_saliency.shape[-2:] != pred_main.shape[-2:]:
        target_resized = F.interpolate(
            target_saliency,
            size=pred_main.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    else:
        target_resized = target_saliency

    loss_boundary = _compute_boundary_precision_loss(pred_main, target_resized, eps=eps)
    total = total + boundary_weight * loss_boundary
    base_losses["boundary"] = loss_boundary

    # 3) 若存在 BAEF 的边界图，则叠加边界监督损失
    aux = model_out.get("aux")
    edge_losses: Dict[str, torch.Tensor] = {}

    if aux is not None:
        bmaps = aux.get("B_maps")
        if isinstance(bmaps, (list, tuple)) and len(bmaps) > 0:
            with torch.no_grad():
                gt_edge = _compute_soft_edge_map(target_saliency, eps=eps)

            per_scale_losses = []
            for idx, bmap in enumerate(bmaps):
                if not isinstance(bmap, torch.Tensor):
                    continue
                pred_edge = torch.clamp(bmap.float(), min=eps, max=1.0 - eps)
                if gt_edge.shape[-2:] != pred_edge.shape[-2:]:
                    gt_edge_resized = F.interpolate(
                        gt_edge,
                        size=pred_edge.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    gt_edge_resized = gt_edge

                loss_edge_i = F.binary_cross_entropy(pred_edge, torch.clamp(gt_edge_resized, 0.0, 1.0))
                name_i = f"edge_b{len(bmaps) - idx}"  # 例如 B2/B1/B0 -> edge_b3/edge_b2/edge_b1
                edge_losses[name_i] = loss_edge_i
                per_scale_losses.append(loss_edge_i)

            if per_scale_losses:
                loss_edge = sum(per_scale_losses) / len(per_scale_losses)
                total = total + edge_weight * loss_edge
                edge_losses["edge"] = loss_edge

    # 合并所有损失项
    all_losses: Dict[str, torch.Tensor] = {}
    all_losses.update(base_losses)
    all_losses.pop("loss", None)
    all_losses.update(edge_losses)
    all_losses["loss"] = total
    return all_losses
