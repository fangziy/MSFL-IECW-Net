import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import os

# 添加学习率调度器
def create_scheduler(optimizer, args):
    """根据配置创建学习率调度器"""
    # 检查是否有scheduler配置
    if not hasattr(args, 'scheduler') or args.scheduler is None:
        return None
    
    # 处理新的嵌套配置格式
    if isinstance(args.scheduler, dict):
        scheduler_name = args.scheduler.get('name', '')
        step_size = args.scheduler.get('step_size', 30)
        gamma = args.scheduler.get('gamma', 0.1)
        milestones = args.scheduler.get('milestones', [])
        patience = args.scheduler.get('patience', 10)
        min_lr = args.scheduler.get('min_lr', 0)
    else:
        # 处理旧的字符串格式
        scheduler_name = args.scheduler
        step_size = getattr(args, 'step_size', 30)
        gamma = getattr(args, 'gamma', 0.1)
        milestones = getattr(args, 'milestones', [])
        patience = getattr(args, 'patience', 10)
        min_lr = getattr(args, 'min_lr', 0)
    
    if scheduler_name == 'StepLR':
        # 每个step_size个epoch，学习率乘以gamma
        return optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=step_size, 
            gamma=gamma
        )
    elif scheduler_name == 'MultiStepLR':
        # 在指定的milestones处，学习率乘以gamma
        return optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=milestones, 
            gamma=gamma
        )
    elif scheduler_name == 'ReduceLROnPlateau':
        # 当指标停止改善时，学习率降低
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=gamma, 
            patience=patience
        )
    elif scheduler_name == 'CosineAnnealingLR':
        # 余弦退火调度器
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=getattr(args, 'num_epochs', 100), 
            eta_min=min_lr
        )
    else:
        # 默认不使用调度器
        return None