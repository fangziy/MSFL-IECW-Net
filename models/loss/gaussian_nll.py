import torch
import torch.nn as nn
from typing import Optional

class GaussianNLLLoss(nn.Module):
    def __init__(self, reduction: str = 'mean', constant: float = 0.0):
        """
        高斯负对数似然损失函数
        
        参数:
            reduction: 损失聚合方式，可选 'mean', 'sum', 'none'
            constant: 可选常数偏移项，对应原损失函数中的 +5
        """
        super(GaussianNLLLoss, self).__init__()
        self.reduction = reduction
        self.constant = constant

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, y_sigma: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算损失
        
        参数:
            y_pred: 预测值，形状 [B, ...]
            y_true: 真实值，形状 [B, ...]
            y_sigma: 预测的标准差，形状 [B, ...] 或 [B, 1, ...]
        """
        return gaussian_nll_loss(y_pred, y_true, y_sigma, 
                                reduction=self.reduction, 
                                constant=self.constant)

def gaussian_nll_loss(y_pred: torch.Tensor, 
                     y_true: torch.Tensor, 
                     y_sigma: torch.Tensor,
                     reduction: str = 'mean',
                     constant: float = 0.0) -> torch.Tensor:
    """
    计算高斯负对数似然损失
    
    损失公式:
        loss = log(y_sigma)/2 + (y_true - y_pred)^2/(2*y_sigma) + constant
        
    参数:
        y_pred: 预测值，形状 [B, ...]
        y_true: 真实值，形状 [B, ...]
        y_sigma: 预测的标准差，形状 [B, ...] 或 [B, 1, ...]
        reduction: 损失聚合方式，可选 'mean', 'sum', 'none'
        constant: 可选常数偏移项
        
    返回:
        标量损失值(如果reduction为'mean'或'sum')或逐样本损失(如果reduction为'none')
    """
    # 确保输入形状匹配
    assert y_pred.shape == y_true.shape, "预测值和真实值形状不匹配"
    assert y_sigma.shape == y_pred.shape or (
        y_sigma.shape[0] == y_pred.shape[0] and 
        y_sigma.dim() == y_pred.dim() and
        all(s == 1 for s in y_sigma.shape[1:])
    ), "标准差形状不符合要求"
    
    # 计算对数似然损失
    squared_error = (y_true - y_pred) ** 2
    loss = 0.5 * torch.log(y_sigma) + squared_error / (2 * y_sigma) + constant
    
    # 应用reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"不支持的reduction方法: {reduction}")

# 单元测试
if __name__ == "__main__":
    # 测试1: 基本功能
    B, C = 32, 5
    y_pred = torch.randn(B, C, requires_grad=True)
    y_true = torch.randn(B, C)
    y_sigma = torch.ones(B, C) * 0.5  # 固定标准差
    
    # 手动计算损失
    manual_loss = (torch.log(y_sigma)/2 + (y_true - y_pred)**2/(2*y_sigma)).mean() + 5
    print(f"手动计算损失: {manual_loss.item()}")
    
    # 使用封装函数计算
    loss_fn = GaussianNLLLoss(reduction='mean', constant=5.0)
    loss = loss_fn(y_pred, y_true, y_sigma)
    print(f"封装函数计算损失: {loss.item()}")
    
    # 验证梯度
    loss.backward()
    print(f"梯度检查: {y_pred.grad[0, 0].item()}")
    
    # 测试2: 不同reduction方式
    loss_sum = GaussianNLLLoss(reduction='sum', constant=0.0)(y_pred, y_true, y_sigma)
    loss_none = GaussianNLLLoss(reduction='none', constant=0.0)(y_pred, y_true, y_sigma)
    print(f"sum损失: {loss_sum.item()}")
    print(f"none损失形状: {loss_none.shape}")
    
    # 测试3: 广播机制
    y_sigma_broadcast = torch.ones(B, 1) * 0.5  # 广播标准差
    loss_broadcast = GaussianNLLLoss(reduction='mean')(y_pred, y_true, y_sigma_broadcast)
    print(f"广播标准差损失: {loss_broadcast.item()}")