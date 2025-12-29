import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MTFE(nn.Module):
    """
    Multi-dimensional Thermal Feature Enhancement (MTFE)
    
    将单通道热图像转换为三个互补的热线索（纹理/边缘、扩散一致性、伪波段发射率），
    通过注意力机制聚合为统一的先验，并用该先验门控注意力以增强边界保真度。
    """
    
    def __init__(
        self,
        C_p: int = 64,  # 共同通道数
        B: int = 4,  # 伪波段分支的滤波器数量
        window_size: int = 7,  # 窗口注意力窗口大小
        num_heads: int = 4,  # 注意力头数
        diffusion_iterations: int = 10,  # 扩散迭代次数
        diffusion_alpha: float = 0.1,  # 扩散步长
        diffusion_kappa: float = 1.0,  # 扩散参数 kappa
        sigma1: float = 1.0,  # DoG 第一个高斯标准差
        sigma2: float = 2.0,  # DoG 第二个高斯标准差
    ):
        super(MTFE, self).__init__()
        
        self.C_p = C_p
        self.B = B
        self.window_size = window_size
        self.num_heads = num_heads
        self.diffusion_iterations = diffusion_iterations
        self.diffusion_alpha = diffusion_alpha
        self.diffusion_kappa = diffusion_kappa
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        
        # 注册 Sobel 和 Laplacian 核（不可训练）
        self.register_buffer('sobel_x', self._get_sobel_x())
        self.register_buffer('sobel_y', self._get_sobel_y())
        self.register_buffer('laplacian', self._get_laplacian())
        
        # (C) 伪波段分支：创建多尺度 DoG 滤波器组
        self.band_filters = self._create_band_filters()
        
        # 注意力聚合模块
        # 投影各分支到共同通道 C_p
        # F_tex: 2 通道 -> C_p
        self.phi_tex = nn.Conv2d(2, C_p, kernel_size=1)
        # F_diff: 2 通道 -> C_p (D + DoG)
        self.phi_diff = nn.Conv2d(2, C_p, kernel_size=1)
        # F_band: B 通道 -> C_p
        self.phi_band = nn.Conv2d(B, C_p, kernel_size=1)
        
        # 分支注意力：输入是 3*C_p 通道，输出是 3 个分支的注意力权重
        self.branch_attention_conv = nn.Conv2d(3 * C_p, 3, kernel_size=1)
        
        # 先验门控窗口注意力模块
        # 用于映射先验到注意力令牌空间
        self.prior_projection = nn.Conv2d(C_p, C_p, kernel_size=1)
        
        # K 和 V 的调制网络（小型 MLP/卷积）
        # 这些网络将在运行时根据输入特征通道数动态创建
    
    def _get_sobel_x(self):
        """Sobel X 核"""
        return torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    def _get_sobel_y(self):
        """Sobel Y 核"""
        return torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    def _get_laplacian(self):
        """Laplacian 核"""
        return torch.tensor([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
    
    def _create_band_filters(self):
        """创建多尺度带通滤波器组（使用 DoG 作为带通滤波器）"""
        filters = []
        scales = [0.5, 1.0, 1.5, 2.0]  # 多尺度
        for scale in scales[:self.B]:
            # 创建高斯核对
            kernel_size = int(6 * scale + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            sigma1 = scale
            sigma2 = scale * 1.6
            kernel1 = self._gaussian_kernel_2d(kernel_size, sigma1)
            kernel2 = self._gaussian_kernel_2d(kernel_size, sigma2)
            dog_kernel = kernel2 - kernel1
            filters.append(dog_kernel.unsqueeze(0).unsqueeze(0))
        return nn.ParameterList([nn.Parameter(f, requires_grad=False) for f in filters])
    
    def _gaussian_kernel_2d(self, kernel_size: int, sigma: float):
        """创建 2D 高斯核"""
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        y = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel
    
    def _gaussian_kernel_1d(self, kernel_size: int, sigma: float):
        """创建 1D 高斯核（用于 DoG）"""
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel = torch.exp(-(x**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel
    
    def _get_gaussian_kernel_2d(self, sigma: float, kernel_size: Optional[int] = None):
        """获取 2D 高斯核用于 DoG"""
        if kernel_size is None:
            kernel_size = int(6 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
        return self._gaussian_kernel_2d(kernel_size, sigma)
    
    def texture_edge_branch(self, I_T: torch.Tensor) -> torch.Tensor:
        """
        (A) 纹理/边缘分支
        
        Args:
            I_T: 输入热图像 [B, 1, H, W]
            
        Returns:
            F_tex: 纹理特征 [B, 2, H, W]
        """
        # Sobel 卷积
        G_x = F.conv2d(I_T, self.sobel_x, padding=1)
        G_y = F.conv2d(I_T, self.sobel_y, padding=1)
        G = torch.sqrt(G_x**2 + G_y**2 + 1e-8)
        
        # Laplacian 卷积
        L = torch.abs(F.conv2d(I_T, self.laplacian, padding=1))
        
        # 拼接
        F_tex = torch.cat([G, L], dim=1)  # [B, 2, H, W]
        
        return F_tex
    
    def diffusion_consistency_branch(self, I_T: torch.Tensor) -> torch.Tensor:
        """
        (B) 扩散一致性分支
        
        Args:
            I_T: 输入热图像 [B, 1, H, W]
            
        Returns:
            F_diff: 扩散特征 [B, 2, H, W]
        """
        # 各向异性扩散
        D = self._anisotropic_diffusion(I_T)
        
        # Difference of Gaussians
        kernel1 = self._get_gaussian_kernel_2d(self.sigma1).to(I_T.device)
        kernel2 = self._get_gaussian_kernel_2d(self.sigma2).to(I_T.device)
        kernel1 = kernel1.view(1, 1, *kernel1.shape).expand(1, 1, -1, -1)
        kernel2 = kernel2.view(1, 1, *kernel2.shape).expand(1, 1, -1, -1)
        
        G_sigma1 = F.conv2d(I_T, kernel1, padding=kernel1.shape[-1]//2)
        G_sigma2 = F.conv2d(I_T, kernel2, padding=kernel2.shape[-1]//2)
        DoG = torch.abs(G_sigma2 - G_sigma1)
        
        # 拼接
        F_diff = torch.cat([D, DoG], dim=1)  # [B, 2, H, W]
        
        return F_diff
    
    def _anisotropic_diffusion(self, I_T: torch.Tensor) -> torch.Tensor:
        """
        各向异性扩散迭代
        
        I_T^{(t+1)} = I_T^{(t)} + alpha * div(c(||grad I_T^{(t)}||) * grad I_T^{(t)})
        c(s) = exp(-(s/kappa)^2)
        """
        I_t = I_T.clone()
        
        for _ in range(self.diffusion_iterations):
            # 计算梯度
            # 使用中心差分
            pad_I = F.pad(I_t, (1, 1, 1, 1), mode='replicate')
            grad_x = pad_I[:, :, 1:-1, 2:] - pad_I[:, :, 1:-1, :-2]
            grad_y = pad_I[:, :, 2:, 1:-1] - pad_I[:, :, :-2, 1:-1]
            grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
            
            # 扩散系数
            c = torch.exp(-(grad_mag / self.diffusion_kappa)**2)
            
            # 计算散度 div(c * grad)
            # c * grad_x 和 c * grad_y
            c_grad_x = c * grad_x
            c_grad_y = c * grad_y
            
            # 散度计算
            pad_cx = F.pad(c_grad_x, (1, 1, 0, 0), mode='constant', value=0)
            pad_cy = F.pad(c_grad_y, (0, 0, 1, 1), mode='constant', value=0)
            div_x = pad_cx[:, :, :, 2:] - pad_cx[:, :, :, :-2]
            div_y = pad_cy[:, :, 2:, :] - pad_cy[:, :, :-2, :]
            div = div_x + div_y
            
            # 更新
            I_t = I_t + self.diffusion_alpha * div
        
        return I_t
    
    def pseudo_band_emissivity_branch(self, I_T: torch.Tensor) -> torch.Tensor:
        """
        (C) 伪波段发射率分支
        
        Args:
            I_T: 输入热图像 [B, 1, H, W]
            
        Returns:
            F_band: 波段特征 [B, B, H, W]
        """
        B_b_list = []
        for b in range(self.B):
            H_b = self.band_filters[b]
            # 确保 H_b 在正确的设备上
            H_b = H_b.to(I_T.device)
            B_b = torch.abs(F.conv2d(I_T, H_b, padding=H_b.shape[-1]//2))
            B_b_list.append(B_b)
        
        F_band = torch.cat(B_b_list, dim=1)  # [B, B, H, W]
        
        return F_band
    
    def attention_based_prior_aggregation(
        self,
        F_tex: torch.Tensor,
        F_diff: torch.Tensor,
        F_band: torch.Tensor
    ) -> torch.Tensor:
        """
        基于注意力的先验聚合
        
        Args:
            F_tex: 纹理特征 [B, 2, H, W]
            F_diff: 扩散特征 [B, 2, H, W]
            F_band: 波段特征 [B, B, H, W]
            
        Returns:
            P_thermal: 热先验 [B, C_p, H, W]
        """
        # 投影到共同通道
        F_tex_tilde = self.phi_tex(F_tex)  # [B, C_p, H, W]
        F_diff_tilde = self.phi_diff(F_diff)  # [B, C_p, H, W]
        F_band_tilde = self.phi_band(F_band)  # [B, C_p, H, W]
        
        # 拼接并计算分支注意力
        F_concat = torch.cat([F_tex_tilde, F_diff_tilde, F_band_tilde], dim=1)  # [B, 3*C_p, H, W]
        A = self.branch_attention_conv(F_concat)  # [B, 3, H, W]
        
        # Softmax 在分支维度上
        w = F.softmax(A, dim=1)  # [B, 3, H, W]
        w_tex, w_diff, w_band = w[:, 0:1], w[:, 1:2], w[:, 2:3]  # 每个 [B, 1, H, W]
        
        # 加权融合
        P_thermal = w_tex * F_tex_tilde + w_diff * F_diff_tilde + w_band * F_band_tilde
        
        return P_thermal
    
    def window_partition(self, x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        将特征图分割为窗口
        
        Args:
            x: [B, C, H, W]
            window_size: 窗口大小
            
        Returns:
            windows: [B*num_windows, C, window_size, window_size]
            (H_pad, W_pad): 填充后的尺寸
        """
        B, C, H, W = x.shape
        
        # 填充到窗口大小的倍数
        H_pad = (H + window_size - 1) // window_size * window_size
        W_pad = (W + window_size - 1) // window_size * window_size
        
        if H_pad != H or W_pad != W:
            x = F.pad(x, (0, W_pad - W, 0, H_pad - H))
        
        # 重塑为窗口
        num_windows_h = H_pad // window_size
        num_windows_w = W_pad // window_size
        
        x = x.view(B, C, num_windows_h, window_size, num_windows_w, window_size)
        windows = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        windows = windows.view(-1, C, window_size, window_size)
        
        return windows, (H_pad, W_pad)
    
    def window_reverse(self, windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
        """
        将窗口重组为特征图
        
        Args:
            windows: [B*num_windows, C, window_size, window_size]
            window_size: 窗口大小
            H, W: 原始尺寸
            
        Returns:
            x: [B, C, H, W]
        """
        num_windows_h = (H + window_size - 1) // window_size
        num_windows_w = (W + window_size - 1) // window_size
        B = windows.shape[0] // (num_windows_h * num_windows_w)
        
        H_pad = num_windows_h * window_size
        W_pad = num_windows_w * window_size
        
        x = windows.view(B, num_windows_h, num_windows_w, -1, window_size, window_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, -1, H_pad, W_pad)
        
        # 裁剪到原始尺寸
        if H_pad != H or W_pad != W:
            x = x[:, :, :H, :W]
        
        return x
    
    def prior_gated_window_attention(
        self,
        T: torch.Tensor,
        P_thermal: torch.Tensor,
        C_in: Optional[int] = None
    ) -> torch.Tensor:
        """
        先验门控窗口内注意力
        
        Args:
            T: 热特征 [B, C_in, H, W]
            P_thermal: 热先验 [B, C_p, H, W]
            C_in: 输入特征通道数（如果为 None，从 T 推断）
            
        Returns:
            F_thermal_MTFE: MTFE 输出 [B, C_in, H, W]
        """
        if C_in is None:
            C_in = T.shape[1]
        
        B, C, H, W = T.shape
        
        # 将先验投影到注意力令牌空间
        # 这里需要将 P_thermal 广播/映射到与 T 相同的空间分辨率
        # 使用简单的投影和插值
        P_th_proj = self.prior_projection(P_thermal)
        if P_th_proj.shape[-2:] != (H, W):
            P_th_hat = F.interpolate(
                P_th_proj,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
        else:
            P_th_hat = P_th_proj
        
        # 将 P_th_hat 扩展到与输入特征相同的通道数（如果需要）
        if P_th_hat.shape[1] != C:
            # 使用 1x1 卷积调整通道
            adapter_name = f'prior_channel_adapter_{C}'
            if not hasattr(self, adapter_name):
                adapter = nn.Conv2d(self.C_p, C, kernel_size=1).to(T.device)
                self.add_module(adapter_name, adapter)
            else:
                adapter = getattr(self, adapter_name)
            P_th_hat = adapter(P_th_hat)
        
        # 分割为窗口
        T_windows, (H_pad, W_pad) = self.window_partition(T, self.window_size)
        P_windows, _ = self.window_partition(P_th_hat, self.window_size)
        
        num_windows = T_windows.shape[0]
        
        # 重塑为序列格式用于注意力计算
        T_windows_seq = T_windows.flatten(2).transpose(1, 2)  # [B*num_windows, window_size^2, C]
        P_windows_seq = P_windows.flatten(2).transpose(1, 2)  # [B*num_windows, window_size^2, C]
        
        # 创建 Q, K, V（这里简化处理，实际中可能需要传入预计算的 Q, K, V）
        # 为了完整性，我们创建简单的线性投影
        qkv_name = f'qkv_proj_{C}'
        if not hasattr(self, qkv_name):
            qkv_proj = nn.Linear(C, C * 3).to(T.device)
            self.add_module(qkv_name, qkv_proj)
        else:
            qkv_proj = getattr(self, qkv_name)
        
        qkv = qkv_proj(T_windows_seq)  # [B*num_windows, window_size^2, 3*C]
        qkv = qkv.reshape(num_windows, -1, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*num_windows, num_heads, window_size^2, C//num_heads]
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # 使用先验调制 K 和 V
        # 将 P_windows_seq 重塑为与 K, V 相同的形状
        P_seq = P_windows_seq.view(num_windows, -1, self.num_heads, C // self.num_heads)
        P_seq = P_seq.permute(0, 2, 1, 3)  # [num_windows, num_heads, window_size^2, C//num_heads]
        
        # 通过 G_k 和 G_v 处理先验（需要重塑回空间格式）
        P_space = P_seq.permute(0, 2, 1, 3).contiguous()
        P_space = P_space.view(num_windows, C, self.window_size, self.window_size)
        
        # 应用调制网络（动态创建如果不存在）
        G_k_name = f'G_k_{C}'
        G_v_name = f'G_v_{C}'
        
        if not hasattr(self, G_k_name):
            G_k = nn.Sequential(
                nn.Conv2d(C, C, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(C, C, kernel_size=1)
            ).to(T.device)
            self.add_module(G_k_name, G_k)
        else:
            G_k = getattr(self, G_k_name)
        
        if not hasattr(self, G_v_name):
            G_v = nn.Sequential(
                nn.Conv2d(C, C, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(C, C, kernel_size=1)
            ).to(T.device)
            self.add_module(G_v_name, G_v)
        else:
            G_v = getattr(self, G_v_name)
        
        mod_k = torch.sigmoid(G_k(P_space))
        mod_v = torch.sigmoid(G_v(P_space))
        
        # 重塑回注意力格式
        mod_k = mod_k.flatten(2).transpose(1, 2).view(num_windows, -1, self.num_heads, C // self.num_heads)
        mod_k = mod_k.permute(0, 2, 1, 3)
        mod_v = mod_v.flatten(2).transpose(1, 2).view(num_windows, -1, self.num_heads, C // self.num_heads)
        mod_v = mod_v.permute(0, 2, 1, 3)
        
        # 调制 K 和 V
        K = K * mod_k
        V = V * mod_v
        
        # 计算注意力
        scale = (C // self.num_heads) ** -0.5
        attn = (Q @ K.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力到 V
        out = (attn @ V)  # [num_windows, num_heads, window_size^2, C//num_heads]
        
        # 重塑回空间格式
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(num_windows, -1, C)
        out = out.transpose(1, 2)
        out = out.view(num_windows, C, self.window_size, self.window_size)
        
        # 重组窗口
        F_thermal_MTFE = self.window_reverse(out, self.window_size, H, W)
        
        return F_thermal_MTFE
    
    def forward(
        self,
        I_T: torch.Tensor,
        T: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            I_T: 输入热图像 [B, 1, H, W]，值域 [0, 1]
            T: 可选的热特征 [B, C_in, H, W]，如果不提供则返回先验
            
        Returns:
            P_thermal: 热先验 [B, C_p, H, W]
            F_thermal_MTFE: MTFE 输出（如果提供了 T），否则为 None
        """
        # 三个分支特征提取
        F_tex = self.texture_edge_branch(I_T)
        F_diff = self.diffusion_consistency_branch(I_T)
        F_band = self.pseudo_band_emissivity_branch(I_T)
        
        # 基于注意力的先验聚合
        P_thermal = self.attention_based_prior_aggregation(F_tex, F_diff, F_band)
        
        # 如果提供了 T，执行先验门控窗口注意力
        if T is not None:
            F_thermal_MTFE = self.prior_gated_window_attention(T, P_thermal)
            return P_thermal, F_thermal_MTFE
        else:
            return P_thermal, None