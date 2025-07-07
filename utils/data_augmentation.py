"""
光谱数据增强模块
Author: Ziyu Fang
Date: 2024-12-25
Email: fangziyushiwo@126.com
"""

import torch
import numpy as np
import random
from typing import Dict, List, Optional, Union


class SpectralAugmentation:
    """光谱数据增强类"""
    
    def __init__(self, augmentation_config: Optional[Dict] = None):
        """
        初始化数据增强器
        
        Args:
            augmentation_config: 数据增强配置字典，如果为None则不进行任何增强
        """
        self.config = augmentation_config or {}
        self.enabled = bool(self.config)
        
        # 解析配置
        if self.enabled:
            self.probability = self.config.get('probability', 0.5)  # 整体增强概率
            self.methods = self.config.get('methods', {})
            
    def __call__(self, spectrum: torch.Tensor) -> torch.Tensor:
        """
        对光谱数据进行增强
        
        Args:
            spectrum: 输入光谱张量 [batch_size, spectrum_length] 或 [spectrum_length]
            
        Returns:
            增强后的光谱张量
        """
        if not self.enabled or random.random() > self.probability:
            return spectrum
            
        # 确保是浮点型
        spectrum = spectrum.float()
        original_shape = spectrum.shape
        
        # 如果是单条光谱，增加batch维度
        if spectrum.dim() == 1:
            spectrum = spectrum.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        # 应用各种增强方法
        for method_name, method_config in self.methods.items():
            if method_config.get('enabled', False):
                method_prob = method_config.get('probability', 0.5)
                if random.random() <= method_prob:
                    spectrum = self._apply_method(spectrum, method_name, method_config)
                    
        # 恢复原始形状
        if squeeze_output:
            spectrum = spectrum.squeeze(0)
            
        return spectrum
    
    def _apply_method(self, spectrum: torch.Tensor, method_name: str, config: Dict) -> torch.Tensor:
        """应用特定的增强方法"""
        
        if method_name == 'gaussian_noise':
            return self._add_gaussian_noise(spectrum, config)
        elif method_name == 'spectral_shift':
            return self._spectral_shift(spectrum, config)
        elif method_name == 'spectral_scaling':
            return self._spectral_scaling(spectrum, config)
        elif method_name == 'baseline_drift':
            return self._baseline_drift(spectrum, config)
        elif method_name == 'spectral_flip':
            return self._spectral_flip(spectrum, config)
        elif method_name == 'spectral_smooth':
            return self._spectral_smooth(spectrum, config)
        elif method_name == 'spectral_dropout':
            return self._spectral_dropout(spectrum, config)
        elif method_name == 'wavelength_shift':
            return self._wavelength_shift(spectrum, config)
        else:
            print(f"警告: 未知的数据增强方法 '{method_name}'")
            return spectrum
    
    def _add_gaussian_noise(self, spectrum: torch.Tensor, config: Dict) -> torch.Tensor:
        """添加高斯噪声"""
        noise_std = config.get('std', 0.01)
        noise = torch.randn_like(spectrum) * noise_std
        return spectrum + noise
    
    def _spectral_shift(self, spectrum: torch.Tensor, config: Dict) -> torch.Tensor:
        """光谱值整体偏移"""
        shift_range = config.get('range', 0.05)
        shift = (random.random() - 0.5) * 2 * shift_range
        return spectrum + shift
    
    def _spectral_scaling(self, spectrum: torch.Tensor, config: Dict) -> torch.Tensor:
        """光谱值缩放"""
        scale_range = config.get('range', 0.1)
        scale = 1.0 + (random.random() - 0.5) * 2 * scale_range
        return spectrum * scale
    
    def _baseline_drift(self, spectrum: torch.Tensor, config: Dict) -> torch.Tensor:
        """基线漂移"""
        drift_strength = config.get('strength', 0.02)
        length = spectrum.shape[-1]
        
        # 生成平滑的基线漂移
        x = torch.linspace(0, 1, length, device=spectrum.device)
        drift = torch.sin(2 * np.pi * x * random.uniform(0.5, 3.0)) * drift_strength
        
        return spectrum + drift.unsqueeze(0).expand_as(spectrum)
    
    def _spectral_flip(self, spectrum: torch.Tensor, config: Dict) -> torch.Tensor:
        """光谱翻转（镜像）"""
        return torch.flip(spectrum, dims=[-1])
    
    def _spectral_smooth(self, spectrum: torch.Tensor, config: Dict) -> torch.Tensor:
        """光谱平滑"""
        kernel_size = config.get('kernel_size', 3)
        
        # 简单的移动平均平滑
        if kernel_size > 1:
            # 使用一维卷积进行平滑
            kernel = torch.ones(1, 1, kernel_size, device=spectrum.device) / kernel_size
            padded_spectrum = torch.nn.functional.pad(
                spectrum.unsqueeze(1), 
                (kernel_size//2, kernel_size//2), 
                mode='reflect'
            )
            smoothed = torch.nn.functional.conv1d(padded_spectrum, kernel)
            return smoothed.squeeze(1)
        
        return spectrum
    
    def _spectral_dropout(self, spectrum: torch.Tensor, config: Dict) -> torch.Tensor:
        """光谱dropout - 随机将部分光谱点设为0"""
        dropout_prob = config.get('probability', 0.1)
        mask = torch.rand_like(spectrum) > dropout_prob
        return spectrum * mask
    
    def _wavelength_shift(self, spectrum: torch.Tensor, config: Dict) -> torch.Tensor:
        """波长轴偏移（光谱整体左右移动）"""
        max_shift = config.get('max_shift', 5)
        length = spectrum.shape[-1]
        
        shift = random.randint(-max_shift, max_shift)
        if shift == 0:
            return spectrum
            
        if shift > 0:
            # 右移
            shifted = torch.cat([
                spectrum[..., shift:],
                spectrum[..., :shift]
            ], dim=-1)
        else:
            # 左移
            shifted = torch.cat([
                spectrum[..., -shift:],
                spectrum[..., :-shift]
            ], dim=-1)
            
        return shifted


def create_augmentation(config: Optional[Dict] = None) -> SpectralAugmentation:
    """
    创建数据增强器的工厂函数
    
    Args:
        config: 增强配置，如果为None则创建不增强的实例
        
    Returns:
        SpectralAugmentation实例
    """
    return SpectralAugmentation(config)


# 示例配置
EXAMPLE_CONFIG = {
    "probability": 0.7,  # 整体增强概率
    "methods": {
        "gaussian_noise": {
            "enabled": True,
            "probability": 0.5,
            "std": 0.01
        },
        "spectral_scaling": {
            "enabled": True,
            "probability": 0.3,
            "range": 0.1
        },
        "baseline_drift": {
            "enabled": True,
            "probability": 0.4,
            "strength": 0.02
        },
        "spectral_smooth": {
            "enabled": False,
            "probability": 0.2,
            "kernel_size": 3
        },
        "wavelength_shift": {
            "enabled": True,
            "probability": 0.3,
            "max_shift": 5
        }
    }
} 