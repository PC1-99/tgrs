#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练脚本 - 用于训练 ThermalRGBSaliencyModel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import time
from tqdm import tqdm
from typing import Optional
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from model import ThermalRGBSaliencyModel, compute_saliency_loss_simple, compute_saliency_loss_small_objects


class SaliencyDataset(Dataset):
    """
    简单的显著性检测数据集类
    用户需要根据自己的数据集结构调整这个类
    """
    def __init__(self, rgb_dir, thermal_dir, mask_dir, img_size=224, transform=None):
        """
        Args:
            rgb_dir: RGB图像目录路径
            thermal_dir: 热成像图像目录路径
            mask_dir: 显著性掩码目录路径
            img_size: 图像大小
            transform: 数据增强变换
        """
        self.img_size = img_size
        self.transform = transform
        
        # 这里假设图像文件名相同，用户可以根据实际情况修改
        # 例如: rgb_dir/001.jpg, thermal_dir/001.jpg, mask_dir/001.png
        self.rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.rgb_dir = rgb_dir
        self.thermal_dir = thermal_dir
        self.mask_dir = mask_dir
        
        # 基础变换
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((img_size, img_size))
        self.to_pil = transforms.ToPILImage()
    
    def __len__(self):
        return len(self.rgb_files)
    
    def __getitem__(self, idx):
        # 读取RGB图像
        rgb_path = os.path.join(self.rgb_dir, self.rgb_files[idx])
        rgb_img = Image.open(rgb_path).convert('RGB')
        rgb_img = self.resize(rgb_img)
        rgb_tensor = self.to_tensor(rgb_img)  # [3, H, W]
        
        # 读取热成像图像 (假设是单通道灰度图)
        thermal_path = os.path.join(self.thermal_dir, self.rgb_files[idx])
        thermal_img = Image.open(thermal_path).convert('L')
        thermal_img = self.resize(thermal_img)
        thermal_tensor = self.to_tensor(thermal_img)  # [1, H, W]
        
        # 读取显著性掩码（自动匹配扩展名）
        rgb_base_name = self.rgb_files[idx].rsplit('.', 1)[0]
        # 尝试常见的扩展名
        mask_name = None
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            candidate = rgb_base_name + ext
            candidate_path = os.path.join(self.mask_dir, candidate)
            if os.path.exists(candidate_path):
                mask_name = candidate
                break
        if mask_name is None:
            # 如果找不到，使用RGB文件相同的扩展名
            rgb_ext = os.path.splitext(self.rgb_files[idx])[1]
            mask_name = rgb_base_name + rgb_ext
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask_img = Image.open(mask_path).convert('L')
        mask_img = self.resize(mask_img)
        mask_tensor = self.to_tensor(mask_img)  # [1, H, W]
        # 二值化掩码到[0, 1]
        mask_tensor = (mask_tensor > 0.5).float()
        
        # 数据增强 (如果有)
        if self.transform:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            rgb_tensor = self.transform(rgb_tensor)
            torch.manual_seed(seed)
            thermal_tensor = self.transform(thermal_tensor)
            torch.manual_seed(seed)
            mask_tensor = self.transform(mask_tensor)
        
        return rgb_tensor, thermal_tensor, mask_tensor


def create_dataloaders(
    train_rgb_dir,
    train_thermal_dir,
    train_mask_dir,
    test_rgb_dir=None,
    test_thermal_dir=None,
    test_mask_dir=None,
    img_size=224,
    batch_size=8,
    num_workers=4,
    use_augmentation=True
):
    """创建训练和测试数据加载器"""
    
    train_transform = None
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            # 可以添加更多数据增强
        ])
    
    train_dataset = SaliencyDataset(
        train_rgb_dir, train_thermal_dir, train_mask_dir,
        img_size=img_size, transform=train_transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = None
    if test_rgb_dir and test_thermal_dir and test_mask_dir:
        test_dataset = SaliencyDataset(
            test_rgb_dir, test_thermal_dir, test_mask_dir,
            img_size=img_size, transform=None
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
    
    return train_loader, test_loader


def compute_metrics(pred, target, eps=1e-6):
    """
    计算评估指标: IoU, F-measure, MAE
    """
    # 确保预测和目标尺寸一致
    if target.shape[-2:] != pred.shape[-2:]:
        target = F.interpolate(
            target,
            size=pred.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0.5).float()
    
    # IoU
    inter = (pred_binary * target_binary).sum(dim=[1, 2, 3])
    union = pred_binary.sum(dim=[1, 2, 3]) + target_binary.sum(dim=[1, 2, 3]) - inter
    iou = (inter + eps) / (union + eps)
    iou_mean = iou.mean().item()
    
    # F-measure (F1-score)
    precision = (inter + eps) / (pred_binary.sum(dim=[1, 2, 3]) + eps)
    recall = (inter + eps) / (target_binary.sum(dim=[1, 2, 3]) + eps)
    f_measure = (2 * precision * recall) / (precision + recall + eps)
    f_measure_mean = f_measure.mean().item()
    
    # MAE (Mean Absolute Error)
    mae = (pred - target).abs().mean().item()
    
    return {
        'iou': iou_mean,
        'f_measure': f_measure_mean,
        'mae': mae
    }


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, writer=None, scaler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (rgb_imgs, thermal_imgs, masks) in enumerate(progress_bar):
        rgb_imgs = rgb_imgs.to(device)
        thermal_imgs = thermal_imgs.to(device)
        masks = masks.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        
        # 混合精度训练
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(rgb_imgs, thermal_imgs, compute_loss=True)
                losses = criterion(outputs, masks)
                loss = losses['loss']
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(rgb_imgs, thermal_imgs, compute_loss=True)
            losses = criterion(outputs, masks)
            loss = losses['loss']
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 更新进度条
        progress_bar.set_postfix({'loss': loss.item()})
        
        # 记录到tensorboard (每N个batch)
        if writer and batch_idx % 50 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            for loss_name, loss_value in losses.items():
                if isinstance(loss_value, torch.Tensor):
                    writer.add_scalar(f'Train/{loss_name}', loss_value.item(), global_step)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def test(model, test_loader, criterion, device, epoch, writer=None):
    """测试"""
    model.eval()
    total_loss = 0.0
    all_metrics = {'iou': [], 'f_measure': [], 'mae': []}
    num_batches = 0
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc=f'Epoch {epoch} [Test]')
        
        for batch_idx, (rgb_imgs, thermal_imgs, masks) in enumerate(progress_bar):
            rgb_imgs = rgb_imgs.to(device)
            thermal_imgs = thermal_imgs.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs = model(rgb_imgs, thermal_imgs, compute_loss=True)
            pred = outputs['P']
            
            # 计算损失
            losses = criterion(outputs, masks)
            loss = losses['loss']
            
            total_loss += loss.item()
            num_batches += 1
            
            # 计算指标
            metrics = compute_metrics(pred, masks)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item(),
                'iou': metrics['iou'],
                'f1': metrics['f_measure']
            })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    
    # 记录到tensorboard
    if writer:
        writer.add_scalar('Test/Loss', avg_loss, epoch)
        for metric_name, metric_value in avg_metrics.items():
            writer.add_scalar(f'Test/{metric_name}', metric_value, epoch)
    
    return avg_loss, avg_metrics


def save_checkpoint(model, optimizer, epoch, loss, metrics, save_path):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)
    print(f"检查点已保存: {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"检查点已加载: {checkpoint_path}, 从epoch {epoch}开始")
    return epoch


def main():
    parser = argparse.ArgumentParser(description='训练 ThermalRGBSaliencyModel')
    
    # 数据相关参数
    parser.add_argument('--train-rgb-dir', type=str, required=True, help='训练集RGB图像目录')
    parser.add_argument('--train-thermal-dir', type=str, required=True, help='训练集热成像图像目录')
    parser.add_argument('--train-mask-dir', type=str, required=True, help='训练集掩码目录')
    parser.add_argument('--test-rgb-dir', type=str, default=None, help='测试集RGB图像目录')
    parser.add_argument('--test-thermal-dir', type=str, default=None, help='测试集热成像图像目录')
    parser.add_argument('--test-mask-dir', type=str, default=None, help='测试集掩码目录')
    
    # 模型参数
    parser.add_argument('--img-size', type=int, default=224, help='输入图像大小')
    parser.add_argument('--embed-dim', type=int, default=96, help='编码器嵌入维度')
    parser.add_argument('--simplify-for-small-objects', action='store_true', help='使用小目标简化模式')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=8, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载器工作进程数')
    
    # 损失函数参数
    parser.add_argument('--use-small-object-loss', action='store_true', help='使用小目标增强损失')
    parser.add_argument('--bce-weight', type=float, default=1.0, help='BCE损失权重')
    parser.add_argument('--iou-weight', type=float, default=1.0, help='IoU损失权重')
    parser.add_argument('--dice-weight', type=float, default=0.5, help='Dice损失权重')
    
    # 其他参数
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='检查点保存目录')
    parser.add_argument('--log-dir', type=str, default='./logs', help='TensorBoard日志目录')
    parser.add_argument('--save-interval', type=int, default=10, help='保存检查点的间隔(epoch)')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--use-amp', action='store_true', help='使用混合精度训练')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, test_loader = create_dataloaders(
        args.train_rgb_dir,
        args.train_thermal_dir,
        args.train_mask_dir,
        args.test_rgb_dir,
        args.test_thermal_dir,
        args.test_mask_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"训练集大小: {len(train_loader.dataset)}")
    if test_loader:
        print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 创建模型
    print("创建模型...")
    model = ThermalRGBSaliencyModel(
        img_size=args.img_size,
        embed_dim=args.embed_dim,
        simplify_for_small_objects=args.simplify_for_small_objects
    ).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # 损失函数
    if args.use_small_object_loss:
        criterion = lambda outputs, targets: compute_saliency_loss_small_objects(
            outputs, targets,
            bce_weight=args.bce_weight,
            iou_weight=args.iou_weight,
            dice_weight=args.dice_weight
        )
    else:
        criterion = lambda outputs, targets: compute_saliency_loss_simple(
            outputs, targets,
            bce_weight=args.bce_weight,
            iou_weight=args.iou_weight,
            dice_weight=args.dice_weight
        )
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    
    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume, device)
    
    # 训练循环
    print("开始训练...")
    best_test_iou = 0.0
    
    for epoch in range(start_epoch, args.epochs):
        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, writer, scaler
        )
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Train/LearningRate', current_lr, epoch)
        
        # 测试
        test_loss = None
        test_metrics = None
        if test_loader:
            test_loss, test_metrics = test(
                model, test_loader, criterion, device, epoch, writer
            )
            
            # 保存最佳模型
            if test_metrics['iou'] > best_test_iou:
                best_test_iou = test_metrics['iou']
                best_model_path = os.path.join(args.save_dir, 'best_model.pth')
                save_checkpoint(
                    model, optimizer, epoch, test_loss, test_metrics, best_model_path
                )
                print(f"新的最佳模型已保存! IoU: {best_test_iou:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(
                model, optimizer, epoch, train_loss, test_metrics, checkpoint_path
            )
        
        # 打印epoch总结
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  训练损失: {train_loss:.4f}")
        if test_loss is not None:
            print(f"  测试损失: {test_loss:.4f}")
            if test_metrics:
                print(f"  测试指标 - IoU: {test_metrics['iou']:.4f}, "
                      f"F-measure: {test_metrics['f_measure']:.4f}, "
                      f"MAE: {test_metrics['mae']:.4f}")
        print()
    
    writer.close()
    print("训练完成!")


if __name__ == '__main__':
    main()

