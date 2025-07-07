import torch.nn as nn
from torch.nn import functional as F
from models.loss import *


def create_criterion(object_type, loss_dict):
    # 初始化结果字典
    criterion_dict = {}

    if object_type == 'cls':
        name = loss_dict['cls_loss']['loss_name']
        if name == 'FocalLoss':
            criterion_dict['cls'] = FocalLoss(gamma=2)
        elif name == 'CrossEntropyLoss':
            criterion_dict['cls'] = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported classification loss: {name}")
    
    elif object_type == 'reg':
        name = loss_dict['reg_loss']['loss_name']
        if name == 'MSELoss':
            criterion_dict['reg'] = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported regression loss: {name}")
    
    elif object_type == 'cls_reg' or object_type == 'reg_cls':
        # 获取分类和回归损失名称
        cls_name = loss_dict['cls_loss'].get('loss_name') if 'cls_loss' in loss_dict else None
        reg_name = loss_dict['reg_loss'].get('loss_name') if 'reg_loss' in loss_dict else None
        
        # 验证至少有一个损失存在
        if not cls_name and not reg_name:
            raise ValueError("At least one of cls_loss or reg_loss must be provided for object_type 'cls_reg'")
        
        # 创建分类损失（如果存在）
        if cls_name:
            if cls_name == 'FocalLoss':
                criterion_dict['cls'] = FocalLoss(gamma=2)
            elif cls_name == 'CrossEntropyLoss':
                criterion_dict['cls'] = nn.CrossEntropyLoss()
            else:
                raise ValueError(f"Unsupported classification loss: {cls_name}")
        
        # 创建回归损失（如果存在）
        if reg_name:
            if reg_name == 'MSELoss':
                criterion_dict['reg'] = nn.MSELoss()
            else:
                raise ValueError(f"Unsupported regression loss: {reg_name}")
        
    else:
        raise ValueError(f"Unsupported object_type: {object_type}")
    
    # 确保至少有一个损失函数
    if not criterion_dict:
        raise ValueError(f"No loss function created for object_type: {object_type}")
    
    return criterion_dict