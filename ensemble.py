"""
Author: Ziyu Fang   
Date: 2024-12-25
Email: fangziyushiwo@126.com

Model Ensemble Script - 模型融合脚本
支持多个模型的集成预测，可以进行简单平均或加权平均
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils import load_config_from_yaml, mk_dir
from models import RRNet, MPBDNet, StarNet, BGANet
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from models.create.create_loss import create_criterion

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 创建解析器
parser = argparse.ArgumentParser(description='Model Ensemble')

# 添加参数
parser.add_argument('--config', type=str, required=True, help='ensemble配置文件路径')
parser.add_argument('--save_predictions', action='store_true', help='是否保存预测结果')
parser.add_argument('--output_dir', type=str, default='./ensemble_results', help='结果保存目录')

# 解析命令行参数
args = parser.parse_args()

# 加载配置
config = load_config_from_yaml(args.config)

# 设置随机数种子
seed_value = 2024
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_model(model_config, device):
    """根据配置加载单个模型"""
    model_name = model_config['name']
    model_path = model_config['path']
    
    # 创建模型实例
    if model_name == 'RRNet':
        model = RRNet(
            object_type=config['object_type'],
            num_cls=len(config['cls_dict']),
            num_reg=len(config['reg_columns']),
            list_inplanes=model_config.get('list_inplanes', [20,40,80,160]),
            len_spectrum=model_config.get('len_spectrum', 3834),
            sequence_len=model_config.get('sequence_len', 50),
            mode=model_config.get('mode', 'raw')
        ).to(device)
    elif model_name == 'MPBDNet':
        model = MPBDNet(
            object_type=config['object_type'],
            num_cls=len(config['cls_dict']),
            num_reg=len(config['reg_columns']),
            list_inplanes=model_config.get('list_inplanes', [20,40,80,160]),
            len_spectrum=model_config.get('len_spectrum', 4900)
        ).to(device)
    elif model_name == 'StarNet':
        model = StarNet(
            object_type=config['object_type'],
            num_cls=len(config['cls_dict']),
            num_reg=len(config['reg_columns']),
            len_spectrum=model_config.get('len_spectrum', 3834)
        ).to(device)
    elif model_name == 'BGANet':
        model = BGANet(
            object_type=config['object_type'],
            num_cls=len(config['cls_dict']),
            num_reg=len(config['reg_columns']),
            len_spectrum=model_config.get('len_spectrum', 4900)
        ).to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # 加载预训练权重，忽略不匹配的层
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        print(f"警告：权重加载出现不匹配，尝试部分加载... {e}")
        # 获取预训练权重
        pretrained_dict = torch.load(model_path, map_location=device)
        # 获取当前模型的state_dict
        model_dict = model.state_dict()
        # 过滤掉不匹配的层
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        # 更新模型字典
        model_dict.update(filtered_dict)
        # 加载过滤后的权重
        model.load_state_dict(model_dict)
        print(f"成功加载了 {len(filtered_dict)}/{len(pretrained_dict)} 个匹配的权重层")
    
    model.eval()
    
    return model

def ensemble_predict(models, weights, data_loader, object_type):
    """集成预测"""
    all_predictions_reg = []
    all_predictions_cls = []
    all_labels_reg = []
    all_labels_cls = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            batch_preds_reg = []
            batch_preds_cls = []
            
            # 收集所有模型的预测
            for model, weight in zip(models, weights):
                outputs = model(inputs)
                
                if 'reg' in outputs:
                    batch_preds_reg.append(outputs['reg'].cpu().numpy() * weight)
                if 'cls' in outputs:
                    batch_preds_cls.append(outputs['cls'].cpu().numpy() * weight)
            
            # 加权平均
            if batch_preds_reg:
                ensemble_reg = np.sum(batch_preds_reg, axis=0)
                all_predictions_reg.append(ensemble_reg)
            
            if batch_preds_cls:
                ensemble_cls = np.sum(batch_preds_cls, axis=0)
                all_predictions_cls.append(ensemble_cls)
            
            # 处理标签
            if object_type == 'cls_reg':
                all_labels_reg.append(targets[:, :len(config['reg_columns'])].cpu().numpy())
                all_labels_cls.append(targets[:, len(config['reg_columns']):].cpu().numpy())
            elif object_type == 'reg':
                all_labels_reg.append(targets.cpu().numpy())
            elif object_type == 'cls':
                all_labels_cls.append(targets.cpu().numpy())
    
    # 拼接结果
    predictions = {}
    labels = {}
    
    if all_predictions_reg:
        predictions['reg'] = np.concatenate(all_predictions_reg, axis=0)
        labels['reg'] = np.concatenate(all_labels_reg, axis=0)
    
    if all_predictions_cls:
        predictions['cls'] = np.concatenate(all_predictions_cls, axis=0)
        labels['cls'] = np.concatenate(all_labels_cls, axis=0)
    
    return predictions, labels

def evaluate_ensemble(predictions, labels, object_type, config):
    """评估集成模型性能"""
    print("=== 模型融合结果 ===")
    
    if object_type in ['cls', 'cls_reg', 'reg_cls']:
        y_true = labels['cls'].argmax(1)
        y_pred = predictions['cls'].argmax(1)
        
        # 计算整体准确率
        accuracy = (y_true == y_pred).mean()
        print(f'集成模型准确率: {accuracy:.4f}')
        
        # CEMP星（标签1）作为正样本的指标
        cemp_label = 1
        y_true_binary = (y_true == cemp_label).astype(int)
        y_pred_binary = (y_pred == cemp_label).astype(int)
        precision_cemp = precision_score(y_true_binary, y_pred_binary, zero_division='warn')
        recall_cemp = recall_score(y_true_binary, y_pred_binary, zero_division='warn')
        f1_cemp = f1_score(y_true_binary, y_pred_binary, zero_division='warn')
        
        # 整体加权平均指标
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division='warn')
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division='warn')
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division='warn')
        
        print('=== CEMP星（正样本）指标 ===')
        print(f'CEMP precision: {precision_cemp:.4f}')
        print(f'CEMP recall: {recall_cemp:.4f}')
        print(f'CEMP f1: {f1_cemp:.4f}')
        print('=== 整体加权平均指标 ===')
        print(f'weighted precision: {precision_weighted:.4f}')
        print(f'weighted recall: {recall_weighted:.4f}')
        print(f'weighted f1: {f1_weighted:.4f}')
    
    if object_type in ['reg', 'cls_reg', 'reg_cls']:
        # 反标准化回归结果
        label_reg = labels['reg'] * config['y_std'] + config['y_mean']
        pred_reg = predictions['reg'] * config['y_std'] + config['y_mean']
        
        mae = np.abs(label_reg - pred_reg).mean(axis=0)
        print('=== 回归指标 ===')
        print(f'MAE: {mae}')

def main():
    # 创建输出目录
    mk_dir(args.output_dir)
    
    # 加载数据
    print("正在加载数据...")
    X_val = np.load(os.path.join(config['data_dir'], "val/X_val.npy"))
    
    # 根据任务类型处理标签
    if config['object_type'] == 'reg':
        columns = config['reg_columns']
        y_val = pd.read_csv(os.path.join(config['data_dir'], "val/y_val.csv"), index_col=0)[columns].values
        y_val = (y_val - config['y_mean']) / config['y_std']
    elif config['object_type'] == 'cls':
        columns = config['cls_columns']
        y_val = pd.read_csv(os.path.join(config['data_dir'], "val/y_val.csv"), index_col=0)[columns].values
        y_val = np.eye(len(config['cls_dict']))[y_val].reshape(-1, len(config['cls_dict']))
    elif config['object_type'] in ['cls_reg', 'reg_cls']:
        y_val_reg = pd.read_csv(os.path.join(config['data_dir'], "val/y_val.csv"), index_col=0)[config['reg_columns']].values
        y_val_cls = pd.read_csv(os.path.join(config['data_dir'], "val/y_val.csv"), index_col=0)[config['cls_columns']].values
        y_val_reg = (y_val_reg - config['y_mean']) / config['y_std']
        y_val_cls = np.eye(len(config['cls_dict']))[y_val_cls].reshape(-1, len(config['cls_dict']))
        y_val = np.concatenate([y_val_reg, y_val_cls], axis=1)
    
    # 创建数据加载器
    val_dataset = CustomDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 加载所有模型
    print("正在加载模型...")
    models = []
    weights = []
    
    for model_config in config['models']:
        print(f"加载模型: {model_config['name']} - {model_config['path']}")
        model = load_model(model_config, device)
        models.append(model)
        weights.append(model_config.get('weight', 1.0))
    
    # 归一化权重
    weights = np.array(weights)
    weights = weights / weights.sum()
    print(f"模型权重: {weights}")
    
    # 进行集成预测
    print("正在进行集成预测...")
    predictions, labels = ensemble_predict(models, weights, val_loader, config['object_type'])
    
    # 评估结果
    evaluate_ensemble(predictions, labels, config['object_type'], config)
    
    # 保存预测结果
    if args.save_predictions:
        print("正在保存预测结果...")
        if 'cls' in predictions:
            np.save(os.path.join(args.output_dir, 'ensemble_predictions_cls.npy'), predictions['cls'])
            np.save(os.path.join(args.output_dir, 'ensemble_labels_cls.npy'), labels['cls'])
        if 'reg' in predictions:
            np.save(os.path.join(args.output_dir, 'ensemble_predictions_reg.npy'), predictions['reg'])
            np.save(os.path.join(args.output_dir, 'ensemble_labels_reg.npy'), labels['reg'])
        print(f"预测结果已保存到 {args.output_dir}")

if __name__ == "__main__":
    main() 