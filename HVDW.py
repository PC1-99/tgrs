import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class PseudoTimeGenerator(nn.Module):
    """
    伪时间热图像生成器 Ψ_pt
    
    从单个热图像合成 K 个伪时间变体，模拟实时变化和传感器噪声
    
    对应公式18: {I_T^(k)}_k=1^K = Ψ_pt(I_T)
    其中 K 是预设的超参数，表示要生成的伪时间变体数量，不是由模型生成的。
    K 的值通常在模型初始化时指定（默认值为5），用于控制从单个热图像 I_T 
    中合成多少个不同的伪时间变体 I_T^(k)。
    """
    
    def __init__(self, K: int = 5):
        """
        初始化伪时间生成器
        
        Args:
            K: 伪时间变体数量（超参数），对应公式18中的K
               该参数不是由模型生成的，而是在模型初始化时预设的。
               默认值为5，可根据实验需求调整。
        """
        super(PseudoTimeGenerator, self).__init__()
        self.K = K
        self._gaussian_cache: dict[Tuple[int, float], torch.Tensor] = {}
        
    def forward(self, I_T: torch.Tensor) -> List[torch.Tensor]:
        """
        生成 K 个伪时间变体
        
        对应公式18: {I_T^(k)}_k=1^K = Ψ_pt(I_T)
        其中 K 是预设的超参数（在__init__中指定），表示要生成的变体数量。
        函数通过以下方式生成K个变体：
        1. 第一个变体 (k=0): 保持原始图像不变
        2. 后续变体 (k>0): 通过添加渐进的高斯模糊和传感器噪声来模拟时间变化
        
        Args:
            I_T: 输入热图像 [B, 1, H, W]，值域 [0, 1]
            
        Returns:
            variants: K 个伪时间变体的列表，每个为 [B, 1, H, W]
                     对应 {I_T^(k)}_k=1^K
        """
        variants = []
        
        for k in range(self.K):
            # 添加缓慢漂移：轻微的高斯模糊
            if k > 0:
                sigma = 0.5 * (k / self.K)
                kernel_size = int(6 * sigma + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                gaussian_kernel = self._get_gaussian_kernel(kernel_size, sigma, I_T.device, I_T.dtype)
                gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
                I_T_k = F.conv2d(I_T, gaussian_kernel, padding=kernel_size//2)
            else:
                I_T_k = I_T.clone()
            
            # 添加传感器噪声：轻微的高斯噪声
            noise_scale = 0.02 * (k + 1) / self.K
            noise = torch.randn_like(I_T_k) * noise_scale
            I_T_k = I_T_k + noise
            I_T_k = torch.clamp(I_T_k, 0.0, 1.0)
            
            variants.append(I_T_k)
        
        return variants
    
    def _get_gaussian_kernel(
        self,
        kernel_size: int,
        sigma: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """创建 2D 高斯核"""
        key = (kernel_size, float(sigma))
        kernel = self._gaussian_cache.get(key)
        if kernel is None:
            x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
            y = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            kernel = kernel / kernel.sum()
            self._gaussian_cache[key] = kernel
        return kernel.to(device=device, dtype=dtype)


class TimeVariationAdaptiveRecursiveFusion(nn.Module):
    """
    时间变化自适应递归融合
    
    对每个阶段使用卡尔曼滤波类型的递归融合来生成稳定的热模板和变化图
    
    核心公式说明：
    ============
    
    公式20（先验预测 - Prior Prediction）:
    x̂_s,k|k-1 = A_s x̂_s,k-1|k-1
    
    其中：
    - x̂_s,k|k-1: 在时间步k-1的观测基础上，对时间步k的状态x_s的**先验估计**（预测值）
                 "|k-1"表示基于k-1时刻的信息进行预测
    - x̂_s,k-1|k-1: 在时间步k-1的观测基础上，对时间步k-1的状态x_s的**后验估计**（更新后的值）
    - A_s: 状态转移矩阵（State Transition Matrix），表示状态如何从k-1演变到k
           **Random-walk-type transition**: A_s是单位矩阵I，表示状态保持前一个值，没有明确的趋势或驱动力
           这符合"缓慢变化的先验"（slowly varying prior）的假设
    
    公式21（误差协方差预测 - Error Covariance Prediction）:
    P⁻_s,k = A_s P_s,k-1 A_s^T + Q_s
    
    其中：
    - P⁻_s,k: 时间步k的**预测误差协方差矩阵**（上标"-"表示先验/预测）
              量化了预测值x̂_s,k|k-1的不确定性
    - P_s,k-1: 时间步k-1的**估计误差协方差矩阵**（无上标表示后验/更新后）
               量化了估计值x̂_s,k-1|k-1的不确定性
    - A_s^T: 状态转移矩阵A_s的转置
    - Q_s: **过程不确定性矩阵**（Process Uncertainty/Process Noise Covariance）
           表示状态转移过程中引入的随机扰动和未建模的动态
           Q_s越大，表示状态演变的随机性越大
    
    Random-walk-type Transition 详解：
    ================================
    当A_s = I（单位矩阵）时，公式20简化为：
        x̂_s,k|k-1 = I · x̂_s,k-1|k-1 = x̂_s,k-1|k-1
    
    这意味着：
    1. 状态预测就是前一个状态（没有明确的演变趋势）
    2. 状态的变化主要来自过程噪声Q_s（随机扰动）
    3. 这类似于随机游走：x_k = x_{k-1} + w_k，其中w_k是随机噪声
    
    这种设计适合"缓慢变化的先验"假设，因为：
    - 热图像的统计状态（均值/方差/梯度强度）在时间上应该缓慢变化
    - 没有强烈的趋势或周期性，主要受随机噪声影响
    - 通过Q_s控制变化的幅度，实现"缓慢"变化
    
    P和Q的获取方式：
    ==============
    - P_s,k-1: 在递归过程中通过卡尔曼滤波的更新步骤计算得到
               初始值P_0通常设为小的单位矩阵（如0.1*I）
    - Q_s: 是可学习的模型参数（nn.Parameter），在训练过程中通过反向传播优化
           初始值通常设为小的单位矩阵（如0.1*I）
    - P⁻_s,k: 通过公式21从P_s,k-1和Q_s计算得到
    """
    
    def __init__(
        self,
        dim: int = 64,  # 统计状态维度
        num_stages: int = 4,
    ):
        super(TimeVariationAdaptiveRecursiveFusion, self).__init__()
        self.dim = dim
        self.num_stages = num_stages
        
        # 为每个阶段定义转换矩阵 A_s（随机游走类型，使用单位矩阵）
        # A_s = I（单位矩阵），表示random-walk-type transition
        # 对应公式20中的A_s，用于状态预测：x̂_s,k|k-1 = A_s x̂_s,k-1|k-1
        # 由于A_s=I，预测就是保持前一个状态：x̂_s,k|k-1 = x̂_s,k-1|k-1
        self.register_buffer('A_s', torch.eye(dim).unsqueeze(0).repeat(num_stages, 1, 1))
        
        # 过程不确定性 Q_s（可学习）
        # 对应公式21中的Q_s，表示状态转移过程中的随机扰动
        # Q_s是可学习的参数，在训练过程中通过反向传播优化
        # 初始值设为0.1*I，表示较小的过程噪声
        # Q_s越大，状态演变的随机性越大，变化越快
        self.Q_s = nn.Parameter(torch.eye(dim).unsqueeze(0).repeat(num_stages, 1, 1) * 0.1)
        
        # 观测矩阵 H_s（可学习）
        self.H_s = nn.Parameter(torch.eye(dim).unsqueeze(0).repeat(num_stages, 1, 1))

        # 缓存单位矩阵，避免重复创建
        self.register_buffer('identity_stats', torch.eye(dim))
        
        # 观测噪声基础值 r
        self.r_base = nn.Parameter(torch.ones(dim) * 0.1)
        
        # 统计提取函数 h (mean/variance/gradient-strength)
        # 使用卷积层从特征中提取统计信息
        self.stat_extractors = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d((4, 4)),  # 降采样到 4x4
                nn.Conv2d(3, dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(dim, dim, kernel_size=1),
            ) for _ in range(num_stages)
        ])
        
        # 从统计状态重建模板的函数 Φ_s
        self.template_reconstructors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(dim, dim, kernel_size=1),
            ) for _ in range(num_stages)
        ])
    
    def extract_statistics(self, T_s: torch.Tensor, stage: int) -> torch.Tensor:
        """
        从特征中提取统计状态
        
        Args:
            T_s: 阶段特征 [B, C, H, W]
            stage: 阶段索引
            
        Returns:
            stats: 统计状态 [B, dim, H_small, W_small]
        """
        # 计算梯度强度
        grad_x = T_s[:, :, :, 1:] - T_s[:, :, :, :-1]
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
        grad_y = T_s[:, :, 1:, :] - T_s[:, :, :-1, :]
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        grad_strength = torch.mean(grad_mag, dim=1, keepdim=True)
        
        # 计算均值和方差
        mean_T = torch.mean(T_s, dim=1, keepdim=True)
        var_T = torch.var(T_s, dim=1, keepdim=True)
        
        # 拼接统计信息
        stats = torch.cat([mean_T, var_T, grad_strength], dim=1)  # [B, 3, H, W]
        assert stats.shape[1] == 3, "统计通道数应为 mean/var/grad 共 3 个"
        
        # 使用统计提取器
        stats = self.stat_extractors[stage](stats)
        
        return stats
    
    def compute_gradient_confidence(self, T_s: torch.Tensor) -> torch.Tensor:
        """
        计算归一化的梯度置信度 α_{s,k}
        
        Args:
            T_s: 特征 [B, C, H, W]
            
        Returns:
            alpha: 梯度置信度 [B, 1, H, W]，值域 (0, 1]
        """
        # 计算梯度
        grad_x = T_s[:, :, :, 1:] - T_s[:, :, :, :-1]
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
        grad_y = T_s[:, :, 1:, :] - T_s[:, :, :-1, :]
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # 归一化到 (0, 1]
        grad_mag_mean = torch.mean(grad_mag, dim=1, keepdim=True)
        grad_mag_max = torch.max(grad_mag_mean)
        alpha = grad_mag_mean / (grad_mag_max + 1e-8)
        alpha = alpha * 0.9 + 0.1  # 确保在 (0, 1] 范围内
        
        return alpha
    
    def forward(
        self,
        T_stages: List[List[torch.Tensor]],  # 每个阶段 K 个伪时间特征
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        执行递归融合（基于卡尔曼滤波框架）
        
        符号说明：
        ==========
        - x̂_s,k|k-1: 先验状态估计（基于k-1时刻信息预测k时刻状态）
        - x̂_s,k|k: 后验状态估计（基于k时刻观测更新后的状态）
        - P⁻_s,k: 先验误差协方差（预测的不确定性，上标"-"表示先验）
        - P_s,k: 后验误差协方差（更新的不确定性，无上标表示后验）
        - z_s,k: 观测值（从T_s^(k)提取的统计量）
        - A_s: 状态转移矩阵（单位矩阵，random-walk-type）
        - Q_s: 过程不确定性矩阵（可学习参数）
        - R_k: 观测噪声协方差（自适应，根据梯度置信度α_s,k调整）
        - H_s: 观测矩阵（可学习参数）
        - K_k: 卡尔曼增益（平衡预测和观测的权重）
        
        算法流程（对每个阶段s，处理K个伪时间特征）：
        ============================================
        1. 初始化：从第一个特征T_s^(0)提取初始状态x̂_s,0|0和协方差P_s,0
        
        2. 对k=1到K-1，执行递归融合：
           a) 预测步骤（公式20和21）：
              - x̂_s,k|k-1 = A_s x̂_s,k-1|k-1  （状态预测）
              - P⁻_s,k = A_s P_s,k-1 A_s^T + Q_s  （协方差预测）
           
           b) 更新步骤（卡尔曼滤波标准公式）：
              - 计算卡尔曼增益：K_k = P⁻_s,k H_s^T (H_s P⁻_s,k H_s^T + R_k)^(-1)
              - 状态更新：x̂_s,k|k = x̂_s,k|k-1 + K_k (z_s,k - H_s x̂_s,k|k-1)
              - 协方差更新：P_s,k = (I - K_k H_s) P⁻_s,k
        
        3. 从最终状态x̂_s,K-1|K-1重建稳定的热模板Z_pt_s
        
        4. 计算变化图V_s（伪时间特征的方差）
        
        Args:
            T_stages: 每个阶段的伪时间特征列表
                T_stages[s] = [T_s^{(1)}, ..., T_s^{(K)}]，每个为 [B, C, H, W]
            
        Returns:
            Z_pt_stages: 每个阶段的稳定热模板列表
            V_stages: 每个阶段的变化图列表
        """
        Z_pt_stages = []
        V_stages = []
        
        for s in range(self.num_stages):
            T_s_list = T_stages[s]
            K = len(T_s_list)
            B, C, H, W = T_s_list[0].shape
            
            # ========== 初始化状态和协方差 ==========
            # 从第一个观测提取初始统计状态
            # z_0 = h(T_s^(0))，对应观测值z_s,0
            z_0 = self.extract_statistics(T_s_list[0], s)  # [B, dim, H_s, W_s]
            H_small, W_small = z_0.shape[2], z_0.shape[3]
            
            # 重塑为空间位置（每个空间位置有独立的统计状态）
            num_positions = H_small * W_small
            # x_hat: 状态估计，初始化为第一个观测值
            # 对应 x̂_s,0|0（在时间步0的观测基础上，对时间步0的状态估计）
            x_hat = z_0.flatten(2).transpose(1, 2)  # [B, num_positions, dim]
            
            identity = self.identity_stats
            # P: 误差协方差矩阵，初始化为小的单位矩阵
            # 对应 P_s,0（时间步0的估计误差协方差）
            # 初始不确定性较小（0.1*I），表示对初始估计比较有信心
            P = (
                identity.unsqueeze(0).unsqueeze(0)
                .to(device=z_0.device, dtype=z_0.dtype)
                .expand(B, num_positions, -1, -1)
                .clone()
                * 0.1
            )
            
            A_s = self.A_s[s]  # [dim, dim] - 状态转移矩阵（单位矩阵）
            H_s = self.H_s[s]  # [dim, dim] - 观测矩阵
            Q_s = self.Q_s[s]  # [dim, dim] - 过程不确定性矩阵（可学习）
            
            # ========== 递归融合（对应公式20和21） ==========
            for k in range(1, K):
                # ========== 公式20：先验预测（Prior Prediction） ==========
                # x̂_s,k|k-1 = A_s x̂_s,k-1|k-1
                # 由于A_s是单位矩阵，这简化为：x̂_s,k|k-1 = x̂_s,k-1|k-1
                # 即：预测值就是前一个时刻的估计值（random-walk特性）
                x_hat_prior = torch.matmul(x_hat, A_s.t())  # 对应 x̂_s,k|k-1
                
                # ========== 公式21：误差协方差预测（Error Covariance Prediction） ==========
                # P⁻_s,k = A_s P_s,k-1 A_s^T + Q_s
                # 计算预测误差协方差，量化预测值的不确定性
                # 第一项：A_s P_s,k-1 A_s^T（状态转移带来的不确定性传播）
                P_prior = torch.einsum('de,bnef->bndf', A_s, P)  # A_s · P_s,k-1
                P_prior = torch.einsum('bndf,fe->bnde', P_prior, A_s)  # (A_s · P_s,k-1) · A_s^T
                # 第二项：加上过程噪声Q_s（状态转移过程中的随机扰动）
                P_prior = P_prior + Q_s.unsqueeze(0).unsqueeze(0)  # 对应 P⁻_s,k
                
                # ========== 观测步骤 ==========
                # z_s,k = h(T_s^(k))：从第k个伪时间特征提取统计量作为观测值
                z_k = self.extract_statistics(T_s_list[k], s)
                z_k = z_k.flatten(2).transpose(1, 2)  # [B, num_positions, dim]
                
                # ========== 计算梯度置信度 α_s,k ==========
                # α_s,k ∈ (0, 1]：归一化的梯度置信度
                # 用于自适应调整观测噪声R_k：梯度越强（α越大），观测越可靠（R越小）
                alpha_k = self.compute_gradient_confidence(T_s_list[k])
                alpha_k = F.adaptive_avg_pool2d(alpha_k, (H_small, W_small))
                alpha_k = alpha_k.flatten(2).transpose(1, 2)  # [B, num_positions, 1]
                
                # ========== 自适应观测噪声 R_k ==========
                # 观测噪声R_k根据梯度置信度α_s,k自适应调整
                # R_k = R_base / α_s,k
                # α越大（梯度越强）→ R越小（观测越可靠）→ 更信任观测值
                R_k = torch.diag(self.r_base).unsqueeze(0).unsqueeze(0).repeat(B, num_positions, 1, 1)
                R_k = R_k / (alpha_k.unsqueeze(-1) + 1e-8)  # [B, num_positions, dim, dim]
                
                # 卡尔曼增益
                HP = torch.einsum('de,bnef->bndf', H_s, P_prior)
                HPH = torch.einsum('bndf,fe->bnde', HP, H_s)
                S = HPH + R_k
                
                # 计算卡尔曼增益（简化版本，避免矩阵求逆的数值问题）
                K_k = torch.einsum('bndf,fe->bnde', P_prior, H_s)
                identity_eps = (
                    identity.unsqueeze(0).unsqueeze(0)
                    .to(device=S.device, dtype=S.dtype)
                    .expand_as(S)
                    * 1e-6
                )
                S_inv = torch.inverse(S + identity_eps)
                K_k = torch.einsum('bnde,bnef->bndf', K_k, S_inv)
                
                # ========== 状态更新（后验估计） ==========
                # 使用卡尔曼增益K_k融合预测值和观测值
                # x̂_s,k|k = x̂_s,k|k-1 + K_k (z_s,k - H_s x̂_s,k|k-1)
                # 其中 z_s,k - H_s x̂_s,k|k-1 是"创新"（innovation），表示观测与预测的差异
                z_pred = torch.matmul(x_hat_prior, H_s.t())  # H_s x̂_s,k|k-1（预测的观测值）
                innovation = z_k - z_pred  # 创新：实际观测 - 预测观测
                x_hat = x_hat_prior + torch.einsum('bnde,bne->bnd', K_k, innovation)  # 对应 x̂_s,k|k
                
                # ========== 协方差更新（后验协方差） ==========
                # P_s,k = (I - K_k H_s) P⁻_s,k
                # 更新后的误差协方差，量化更新后估计值的不确定性
                identity_expand = (
                    identity.unsqueeze(0).unsqueeze(0)
                    .to(device=P_prior.device, dtype=P_prior.dtype)
                    .expand_as(P_prior)
                )
                I_KH = identity_expand - torch.matmul(K_k, H_s)  # I - K_k H_s
                P = torch.matmul(I_KH, P_prior)  # 对应 P_s,k（用于下一次迭代的P_s,k-1）
            
            # 重建模板
            x_final = x_hat.transpose(1, 2).view(B, self.dim, H_small, W_small)
            Z_pt_s = self.template_reconstructors[s](x_final)
            Z_pt_s = F.interpolate(Z_pt_s, size=(H, W), mode='bilinear', align_corners=False)
            
            # 计算变化图（伪时间特征的方差）
            T_s_stack = torch.stack(T_s_list, dim=0)  # [K, B, C, H, W]
            V_s = torch.var(T_s_stack, dim=0)  # [B, C, H, W]
            V_s = torch.mean(V_s, dim=1, keepdim=True)  # [B, 1, H, W]
            
            Z_pt_stages.append(Z_pt_s)
            V_stages.append(V_s)
        
        return Z_pt_stages, V_stages


class GMSA(nn.Module):
    """
    Global Multi-Scale Attention (GMSA) 融合模块
    
    融合 RGB 和热特征
    """
    
    def __init__(self, dim: int, num_heads: int = 8):
        super(GMSA, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # 输出投影
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        GMSA 融合
        
        Args:
            R: RGB 特征 [B, C, H, W]
            T: 热特征 [B, C, H, W]
            
        Returns:
            G: 融合特征 [B, C, H, W]
        """
        B, C, H, W = R.shape
        
        # 重塑为序列格式
        R_seq = R.flatten(2).transpose(1, 2)  # [B, H*W, C]
        T_seq = T.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # 使用 RGB 作为 Query，热作为 Key 和 Value
        Q = self.q_proj(R_seq)
        K = self.k_proj(T_seq)
        V = self.v_proj(T_seq)
        
        # 多头注意力
        Q = Q.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        scale = self.head_dim ** -0.5
        attn = (Q @ K.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = (attn @ V).transpose(1, 2).contiguous()
        out = out.view(B, H*W, C)
        
        # 输出投影
        out = self.out_proj(out)
        
        # 重塑回空间格式
        out = out.transpose(1, 2).view(B, C, H, W)
        
        # 残差连接
        out = out + R
        
        return out


class VariationAwareDynamicWindow(nn.Module):
    """
    变化感知动态窗口
    
    根据语义置信度和热稳定性分配不同大小的窗口，并在每个窗口内进行模板门控注意力处理
    """
    
    def __init__(
        self,
        dim: int,
        window_size_small: int = 7,
        window_size_large: int = 14,
        tau: float = 0.5,
        alpha: float = 0.5,
        num_heads: int = 8,
        region_size: int = 8,
        lambda_fusion: float = 0.5,
    ):
        super(VariationAwareDynamicWindow, self).__init__()
        self.dim = dim
        self.window_size_small = window_size_small
        self.window_size_large = window_size_large
        self.tau = tau
        self.alpha = alpha
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.region_size = region_size
        self.lambda_fusion = lambda_fusion
        
        # Q, K, V 投影（用于窗口注意力）
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        
        # 模板和变化图的调制网络
        self.G_k = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
        self.G_v = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
        
        # 变化抑制函数 ψ
        self.psi = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )
        
    def window_partition(self, x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """将特征图分割为窗口"""
        B, C, H, W = x.shape
        H_pad = (H + window_size - 1) // window_size * window_size
        W_pad = (W + window_size - 1) // window_size * window_size
        
        if H_pad != H or W_pad != W:
            x = F.pad(x, (0, W_pad - W, 0, H_pad - H))
        
        num_windows_h = H_pad // window_size
        num_windows_w = W_pad // window_size
        
        x = x.view(B, C, num_windows_h, window_size, num_windows_w, window_size)
        windows = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        windows = windows.view(-1, C, window_size, window_size)
        
        return windows, (H_pad, W_pad)
    
    def window_reverse(self, windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
        """将窗口重组为特征图"""
        B = windows.shape[0] // (
            ((H + window_size - 1) // window_size)
            * ((W + window_size - 1) // window_size)
        )
        H_pad = (H + window_size - 1) // window_size * window_size
        W_pad = (W + window_size - 1) // window_size * window_size
        num_windows_h = H_pad // window_size
        num_windows_w = W_pad // window_size
        
        x = windows.view(B, num_windows_h, num_windows_w, -1, window_size, window_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, -1, H_pad, W_pad)
        
        if H_pad != H or W_pad != W:
            x = x[:, :, :H, :W]
        
        return x
    
    def _self_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor
    ) -> torch.Tensor:
        """
        对应图中的 Self Attention：在窗口内执行多头注意力。
        """
        scale = self.head_dim ** -0.5
        attn = (Q @ K.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        return attn @ V

    def _local_global_fusion(
        self,
        f_prime: torch.Tensor,
        f_global: torch.Tensor
    ) -> torch.Tensor:
        """
        对应图中的 local-global Function：融合局部窗口特征与全局上下文。
        """
        return self.lambda_fusion * f_prime + (1 - self.lambda_fusion) * f_global

    def _tgvs_kv(
        self,
        Z_pt_window: torch.Tensor,
        V_window: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对应图中的 TGVS-KV：使用模板与变化图调制窗口内的 K/V。

        Args:
            Z_pt_window: 窗口模板 [B, C, window_size, window_size]
            V_window: 窗口变化图 [B, 1, window_size, window_size]

        Returns:
            mod_k: 调制因子 [B, num_heads, window_size^2, head_dim]
            mod_v: 调制因子 [B, num_heads, window_size^2, head_dim]
        """
        B = Z_pt_window.shape[0]

        mod_k = torch.sigmoid(self.G_k(Z_pt_window))
        mod_k = mod_k * (1 - self.psi(V_window))
        mod_k = mod_k.flatten(2).transpose(1, 2).view(B, -1, self.num_heads, self.head_dim)
        mod_k = mod_k.permute(0, 2, 1, 3)

        mod_v = torch.sigmoid(self.G_v(Z_pt_window))
        mod_v = mod_v * (1 - self.psi(V_window))
        mod_v = mod_v.flatten(2).transpose(1, 2).view(B, -1, self.num_heads, self.head_dim)
        mod_v = mod_v.permute(0, 2, 1, 3)

        return mod_k, mod_v

    def _get_window_attention_output(
        self,
        G_window: torch.Tensor,
        Z_pt_window: torch.Tensor,
        V_window: torch.Tensor,
        window_size: int
    ) -> torch.Tensor:
        """
        获取窗口自注意力输出 f'_W（不进行局部-全局融合）
        
        Args:
            G_window: 窗口特征 [B, C, window_size, window_size]
            Z_pt_window: 窗口模板 [B, C, window_size, window_size]
            V_window: 窗口变化图 [B, 1, window_size, window_size]
            window_size: 窗口大小
            
        Returns:
            f_prime: 窗口自注意力输出 f'_W [B, C, window_size, window_size]
        """
        B, C, H_win, W_win = G_window.shape
        
        # 重塑为序列格式
        G_seq = G_window.flatten(2).transpose(1, 2)  # [B, window_size^2, C]
        
        # Q, K, V
        qkv = self.qkv_proj(G_seq)
        qkv = qkv.reshape(B, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, window_size^2, head_dim]
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # 模板和变化图调制
        mod_k, mod_v = self._tgvs_kv(Z_pt_window, V_window)
        K = K * mod_k
        V = V * mod_v
        
        # 窗口自注意力
        f_prime = self._self_attention(Q, K, V)  # [B, num_heads, window_size^2, head_dim]
        
        # 重塑回空间格式
        f_prime = f_prime.permute(0, 2, 1, 3).contiguous()
        f_prime = f_prime.view(B, -1, C)
        f_prime = f_prime.transpose(1, 2)
        f_prime = f_prime.view(B, C, H_win, W_win)
        
        return f_prime
    
    def compute_window_scores(
        self,
        p_sal: torch.Tensor,
        V: torch.Tensor,
        region_size: int = 8
    ) -> torch.Tensor:
        """
        计算窗口得分
        
        按照公式：p̄_i = (1/|Q_i|) * Σ_{u∈Q_i} p_sal^(4)(u)
                  V̂_i = Norm((1/|Q_i|) * Σ_{u∈Q_i} V_4(u))
        
        Args:
            p_sal: 语义置信度 [B, 1, H, W]
            V: 变化图 [B, 1, H, W]
            region_size: 区域大小（非重叠区域 Q_i 的大小）
            
        Returns:
            scores: 窗口得分 [B, num_regions_h, num_regions_w]
            region_size: 实际使用的区域边长
            H_crop: 裁剪后的高度
            W_crop: 裁剪后的宽度
        """
        B, _, H, W = p_sal.shape
        
        if H == 0 or W == 0:
            raise ValueError("输入特征图尺寸无效，H 或 W 不能为 0。")
        
        # 根据当前特征图尺寸自适应区域大小
        region_size = max(1, min(region_size, H, W))
        
        # 计算非重叠区域的数量，至少为 1
        num_regions_h = max(1, H // region_size)
        num_regions_w = max(1, W // region_size)
        
        # 裁剪到可整除的大小
        H_crop = num_regions_h * region_size
        W_crop = num_regions_w * region_size
        p_sal = p_sal[:, :, :H_crop, :W_crop]
        V = V[:, :, :H_crop, :W_crop]
        
        # 将特征图分割成固定大小的非重叠区域 Q_i
        # 使用 unfold 来提取非重叠块
        # p_sal: [B, 1, H, W] -> [B, 1, num_regions_h, region_size, num_regions_w, region_size]
        p_sal_reshaped = p_sal.view(B, 1, num_regions_h, region_size, num_regions_w, region_size)
        V_reshaped = V.view(B, 1, num_regions_h, region_size, num_regions_w, region_size)
        
        # 对每个区域 Q_i 计算平均值：p̄_i = (1/|Q_i|) * Σ_{u∈Q_i} p_sal^(4)(u)
        # |Q_i| = region_size * region_size
        p_sal_avg = p_sal_reshaped.mean(dim=(3, 5))  # [B, 1, num_regions_h, num_regions_w]
        
        # 对每个区域 Q_i 计算平均值：V̂_i = (1/|Q_i|) * Σ_{u∈Q_i} V_4(u)
        V_avg = V_reshaped.mean(dim=(3, 5))  # [B, 1, num_regions_h, num_regions_w]
        
        # 归一化 V：V̂_i = Norm(V̂_i)
        V_norm = (V_avg - V_avg.min()) / (V_avg.max() - V_avg.min() + 1e-8)
        
        # 计算联合得分：S_i = α * p̄_i + (1-α) * (1 - V̂_i)
        scores = self.alpha * p_sal_avg + (1 - self.alpha) * (1 - V_norm)
        
        return scores.squeeze(1), region_size, H_crop, W_crop  # [B, num_regions_h, num_regions_w]
    
    def forward(
        self,
        G: torch.Tensor,
        p_sal: torch.Tensor,
        V: torch.Tensor,
        Z_pt: torch.Tensor,
    ) -> torch.Tensor:
        """
        变化感知动态窗口处理
        
        根据窗口得分S_i分配窗口大小，并在每个窗口内进行模板门控窗口注意力处理
        
        Args:
            G: 融合特征 [B, C, H, W]
            p_sal: 语义置信度 [B, 1, H, W]
            V: 变化图 [B, 1, H, W]
            Z_pt: 稳定热模板 [B, C, H, W]
            
        Returns:
            output: 处理后的特征 [B, C, H, W]
        """
        B, C, H, W = G.shape
        
        # 根据当前分辨率自适应区域大小，并计算窗口得分
        region_size_init = max(1, min(self.region_size, H, W))
        scores, region_size_eff, H_crop, W_crop = self.compute_window_scores(p_sal, V, region_size_init)
        num_regions_h, num_regions_w = scores.shape[1], scores.shape[2]
        
        # 裁剪输入特征到可整除 region_size 的大小
        G_crop = G[:, :, :H_crop, :W_crop]
        Z_pt_crop = Z_pt[:, :, :H_crop, :W_crop]
        V_crop = V[:, :, :H_crop, :W_crop]
        
        # 根据得分分配窗口大小：S_i >= τ → 小窗口，S_i < τ → 大窗口
        window_assignments = (scores >= self.tau).long()  # [B, num_regions_h, num_regions_w], 1 for small, 0 for large
        
        # 第一步：收集所有窗口的 f'（窗口自注意力输出），用于计算全局 GAP
        all_f_prime_windows = []  # 存储所有窗口的 f'
        all_window_info = []  # 存储窗口信息，用于后续处理
        
        for b in range(B):
            for i in range(num_regions_h):
                for j in range(num_regions_w):
                    # 确定该区域使用的窗口大小（基于当前batch的得分）
                    win_size = self.window_size_small if window_assignments[b, i, j] > 0 else self.window_size_large
                    
                    # 提取对应区域（使用固定的region_size，与compute_window_scores保持一致）
                    h_start = i * region_size_eff
                    h_end = (i + 1) * region_size_eff
                    w_start = j * region_size_eff
                    w_end = (j + 1) * region_size_eff
                    
                    G_region = G_crop[b:b+1, :, h_start:h_end, w_start:w_end]
                    Z_pt_region = Z_pt_crop[b:b+1, :, h_start:h_end, w_start:w_end]
                    V_region = V_crop[b:b+1, :, h_start:h_end, w_start:w_end]
                    
                    # 保存原始区域大小
                    region_h, region_w = G_region.shape[2], G_region.shape[3]
                    
                    # 如果区域大小小于窗口大小，需要填充到窗口大小
                    if region_h < win_size or region_w < win_size:
                        # 填充到窗口大小
                        pad_h = max(0, win_size - region_h)
                        pad_w = max(0, win_size - region_w)
                        G_region = F.pad(G_region, (0, pad_w, 0, pad_h))
                        Z_pt_region = F.pad(Z_pt_region, (0, pad_w, 0, pad_h))
                        V_region = F.pad(V_region, (0, pad_w, 0, pad_h))
                    
                    # 将区域分割为窗口（使用动态分配的窗口大小）
                    G_windows, (H_pad, W_pad) = self.window_partition(G_region, win_size)
                    Z_pt_windows, _ = self.window_partition(Z_pt_region, win_size)
                    V_windows, _ = self.window_partition(V_region, win_size)
                    
                    # 处理每个窗口（使用模板门控窗口注意力）
                    # 得到所有窗口的 f'_W（窗口自注意力结果）
                    num_windows = G_windows.shape[0]
                    for w_idx in range(num_windows):
                        G_win = G_windows[w_idx:w_idx+1]
                        Z_pt_win = Z_pt_windows[w_idx:w_idx+1]
                        V_win = V_windows[w_idx:w_idx+1]
                        f_prime_win = self._get_window_attention_output(
                            G_win, Z_pt_win, V_win, win_size
                        )
                        all_f_prime_windows.append(f_prime_win)
                        # 保存窗口信息：batch索引、区域索引、窗口索引、窗口大小、区域大小、填充后大小
                        all_window_info.append({
                            'b': b, 'i': i, 'j': j, 'w_idx': w_idx,
                            'win_size': win_size, 'region_h': region_h, 'region_w': region_w,
                            'H_pad': H_pad, 'W_pad': W_pad,
                            'h_start': h_start, 'h_end': h_end, 'w_start': w_start, 'w_end': w_end
                        })
        
        # 第二步：对所有窗口的 f' 进行全局平均池化：GAP(f')
        # 按照公式：f''_W = λ * f'_W + (1-λ) * GAP(f')
        # 其中 GAP(f') 是对所有窗口的 f' 进行全局平均池化
        if len(all_f_prime_windows) > 0:
            # 由于不同窗口可能有不同大小，先对每个窗口进行全局平均池化
            # 然后对所有窗口的全局特征求平均
            f_global_per_window_list = []
            for f_prime_win in all_f_prime_windows:
                # 对每个窗口进行全局平均池化
                f_global_win = F.adaptive_avg_pool2d(f_prime_win, (1, 1))  # [1, C, 1, 1]
                f_global_per_window_list.append(f_global_win)
            
            # 拼接所有窗口的全局特征
            f_global_all = torch.cat(f_global_per_window_list, dim=0)  # [total_windows, C, 1, 1]
            _, C, _, _ = f_global_all.shape
            
            # 对所有窗口的全局特征求平均
            f_global_value = f_global_all.mean(dim=0, keepdim=True)  # [1, C, 1, 1]
        else:
            # 如果没有窗口，使用零值
            C = G_crop.shape[1]
            f_global_value = torch.zeros(1, C, 1, 1, device=G_crop.device, dtype=G_crop.dtype)
        
        # 第三步：使用全局 GAP 值对所有窗口进行局部-全局融合
        # 初始化输出特征图
        output_crop = torch.zeros_like(G_crop)
        
        # 按区域组织窗口，进行融合和重组
        window_idx = 0
        for b in range(B):
            for i in range(num_regions_h):
                for j in range(num_regions_w):
                    # 获取该区域的窗口信息
                    region_windows = []
                    region_info = None
                    win_size = None
                    H_pad = None
                    W_pad = None
                    h_start = None
                    h_end = None
                    w_start = None
                    w_end = None
                    region_h = None
                    region_w = None
                    
                    # 收集该区域的所有窗口
                    while window_idx < len(all_window_info):
                        info = all_window_info[window_idx]
                        if info['b'] == b and info['i'] == i and info['j'] == j:
                            region_windows.append(all_f_prime_windows[window_idx])
                            if region_info is None:
                                region_info = info
                                win_size = info['win_size']
                                H_pad = info['H_pad']
                                W_pad = info['W_pad']
                                h_start = info['h_start']
                                h_end = info['h_end']
                                w_start = info['w_start']
                                w_end = info['w_end']
                                region_h = info['region_h']
                                region_w = info['region_w']
                            window_idx += 1
                        else:
                            break
                    
                    if len(region_windows) == 0:
                        continue
                    
                    # 对每个窗口进行局部-全局融合：f''_W = λ * f'_W + (1-λ) * GAP(f')
                    processed_windows = []
                    for f_prime_win in region_windows:
                        # 将全局 GAP 值扩展到当前窗口大小
                        f_global_expanded = f_global_value.expand(1, C, win_size, win_size)
                        f_double_prime = self._local_global_fusion(f_prime_win, f_global_expanded)
                        
                        # 输出投影
                        f_double_prime_seq = f_double_prime.flatten(2).transpose(1, 2)
                        output_seq = self.out_proj(f_double_prime_seq)
                        output = output_seq.transpose(1, 2).view(1, C, win_size, win_size)
                        processed_windows.append(output)
                    
                    # 重组窗口
                    processed_windows_tensor = torch.cat(processed_windows, dim=0)
                    processed_region = self.window_reverse(processed_windows_tensor, win_size, H_pad, W_pad)
                    
                    # 裁剪回原始区域大小，并将处理后的区域放回输出
                    output_crop[b:b+1, :, h_start:h_end, w_start:w_end] = processed_region[:, :, :region_h, :region_w]
        
        # 如果输入被裁剪了，需要将输出填充回原始大小
        if H_crop < H or W_crop < W:
            output = G.clone()
            output[:, :, :H_crop, :W_crop] = output_crop
        else:
            output = output_crop
        
        return output


class HVDW(nn.Module):
    """
    Heatmap Variation-Aware Dynamic Window (HVDW)
    
    对伪时间热输入执行时间变化自适应递归融合，解决遥感场景中的实时变化和噪声问题
    """
    
    def __init__(
        self,
        K: int = 5,  # 伪时间变体数量（超参数，对应公式18中的K）
        num_stages: int = 4,  # 编码器阶段数
        dim: int = 256,  # 特征维度
        dim_stats: int = 64,  # 统计状态维度
        window_size_small: int = 7,
        window_size_large: int = 14,
        tau: float = 0.5,  # 窗口分配阈值
        alpha: float = 0.5,  # 语义-稳定性权重
        num_heads: int = 8,
        lambda_fusion: float = 0.5,  # 局部-全局融合权重
    ):
        super(HVDW, self).__init__()
        
        self.K = K
        self.num_stages = num_stages
        self.dim = dim
        
        # 伪时间生成器
        self.pseudo_time_generator = PseudoTimeGenerator(K=K)
        
        # 时间变化自适应递归融合
        self.recursive_fusion = TimeVariationAdaptiveRecursiveFusion(
            dim=dim_stats,
            num_stages=num_stages
        )
        
        # GMSA 融合（第4阶段）
        self.gmsa = GMSA(dim=dim, num_heads=num_heads)
        
        # 语义置信度预测头
        self.semantic_head = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 变化感知动态窗口（已集成模板门控窗口注意力）
        self.variation_aware_window = VariationAwareDynamicWindow(
            dim=dim,
            window_size_small=window_size_small,
            window_size_large=window_size_large,
            tau=tau,
            alpha=alpha,
            num_heads=num_heads,
            lambda_fusion=lambda_fusion
        )
    
    def forward(
        self,
        I_T: torch.Tensor,
        thermal_stages: List[torch.Tensor],
        rgb_stages: List[torch.Tensor],
        encoder_thermal: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            I_T: 输入热图像 [B, 1, H, W]，值域 [0, 1]
            thermal_stages: 预先计算的热编码器阶段特征列表
            rgb_stages: 预先计算的 RGB 编码器阶段特征列表
            encoder_thermal: 用于处理伪时间变体的热编码器（返回阶段特征列表）
            
        Returns:
            F_HVDW: HVDW 输出 [B, C, H, W]
            R_4: RGB 第四阶段特征 [B, C, H/16, W/16]（未经过 GMSA 处理）
        """
        if encoder_thermal is None:
            raise ValueError("encoder_thermal 不能为空，用于编码伪时间变体。")
        if not isinstance(thermal_stages, (list, tuple)):
            raise TypeError("thermal_stages 应为 list 或 tuple。")
        if not isinstance(rgb_stages, (list, tuple)):
            raise TypeError("rgb_stages 应为 list 或 tuple。")
        if len(thermal_stages) != self.num_stages:
            raise ValueError(f"thermal_stages 长度应为 {self.num_stages}，实际为 {len(thermal_stages)}。")
        if len(rgb_stages) != self.num_stages:
            raise ValueError(f"rgb_stages 长度应为 {self.num_stages}，实际为 {len(rgb_stages)}。")

        # 1. 生成伪时间变体
        # 对应公式18: {I_T^(k)}_k=1^K = Ψ_pt(I_T)
        # 其中 K 是预设的超参数（在模型初始化时指定），不是由模型生成的
        # 函数 Ψ_pt 从单个热图像 I_T 合成 K 个伪时间变体，模拟实时变化和传感器噪声
        I_T_variants = self.pseudo_time_generator(I_T)  # K 个 [B, 1, H, W]

        # 2. 构建每个阶段的伪时间特征列表，复用原始帧特征
        T_stages = [
            [thermal_stages[s]] for s in range(self.num_stages)
        ]
        for idx, I_T_k in enumerate(I_T_variants):
            if idx == 0:
                continue  # 第一个变体与原始帧一致，已复用
            variant_feats = encoder_thermal(I_T_k)
            if not isinstance(variant_feats, (list, tuple)):
                raise TypeError("encoder_thermal 需返回包含阶段特征的 list/tuple。")
            if len(variant_feats) != self.num_stages:
                raise ValueError(
                    f"encoder_thermal 返回阶段数应为 {self.num_stages}，实际为 {len(variant_feats)}。"
                )
            for s in range(self.num_stages):
                T_stages[s].append(variant_feats[s])
        
        # 3. 时间变化自适应递归融合
        Z_pt_stages, V_stages = self.recursive_fusion(T_stages)
        
        # 4. 复用最后阶段的 RGB 特征
        last_stage_idx = self.num_stages - 1
        R_last = rgb_stages[last_stage_idx]
        
        # 5. 深度 GMSA 融合（最后阶段）
        T_last = Z_pt_stages[last_stage_idx]  # 使用稳定的热模板
        # 调整尺寸和通道
        if T_last.shape[2:] != R_last.shape[2:]:
            T_last = F.interpolate(T_last, size=R_last.shape[2:], mode='bilinear', align_corners=False)
        if T_last.shape[1] != R_last.shape[1]:
            adapter_name_T = 'last_stage_channel_adapter_T'
            if not hasattr(self, adapter_name_T):
                adapter = nn.Conv2d(T_last.shape[1], R_last.shape[1], kernel_size=1).to(T_last.device)
                self.add_module(adapter_name_T, adapter)
            else:
                adapter = getattr(self, adapter_name_T)
            T_last = adapter(T_last)
        
        G_last = self.gmsa(R_last, T_last)
        
        # 6. 预测深度语义置信度
        p_sal_last = self.semantic_head(G_last)
        
        # 7. 变化感知窗口评分和分配
        V_last = V_stages[last_stage_idx]
        if V_last.shape[2:] != G_last.shape[2:]:
            V_last = F.interpolate(V_last, size=G_last.shape[2:], mode='bilinear', align_corners=False)
        
        if Z_pt_stages[last_stage_idx].shape[2:] != G_last.shape[2:]:
            Z_pt_last = F.interpolate(Z_pt_stages[last_stage_idx], size=G_last.shape[2:], mode='bilinear', align_corners=False)
        else:
            Z_pt_last = Z_pt_stages[last_stage_idx]
        
        if Z_pt_last.shape[1] != G_last.shape[1]:
            adapter_name_Z = 'last_stage_channel_adapter_Z'
            if not hasattr(self, adapter_name_Z):
                adapter = nn.Conv2d(Z_pt_last.shape[1], G_last.shape[1], kernel_size=1).to(Z_pt_last.device)
                self.add_module(adapter_name_Z, adapter)
            else:
                adapter = getattr(self, adapter_name_Z)
            Z_pt_last = adapter(Z_pt_last)
        
        # 8. 变化感知动态窗口处理
        # 根据公式：p̄_i = (1/|Q_i|) * Σ_{u∈Q_i} p_sal^(4)(u), V̂_i = Norm((1/|Q_i|) * Σ_{u∈Q_i} V_4(u))
        # 然后计算 S_i = α * p̄_i + (1-α) * (1 - V̂_i)，根据阈值τ分配窗口大小
        # S_i >= τ → 小窗口，S_i < τ → 大窗口
        # variation_aware_window会：
        # 1. 计算窗口得分S_i
        # 2. 根据得分分配窗口大小（小窗口或大窗口）
        # 3. 在每个窗口内进行模板门控窗口注意力处理
        # 4. 返回处理后的特征
        F_HVDW = self.variation_aware_window(G_last, p_sal_last, V_last, Z_pt_last)
        
        return F_HVDW, R_last

