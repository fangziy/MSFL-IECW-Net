"""
Author: Ziyu Fang   
Date: 2024-12-25
Email: fangziyushiwo@126.com
"""

import os
import argparse
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import transforms
from tqdm import tqdm
from utils import load_config_from_yaml, mk_dir
from utils.data_augmentation import create_augmentation
from models import RRNet
from models import MPBDNet,StarNet,BGANet
from models.loss import FocalLoss

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 创建解析器
parser = argparse.ArgumentParser(description='train')

# 添加参数
parser.add_argument('--object_name', type=str, default='没有名字', help='名称')
parser.add_argument('--object_type', type=str, default='cls', help='cls,reg,cls_reg,reg_cls')

parser.add_argument('--num_epochs', type=int, default=200, help='训练的轮数')
parser.add_argument('--train_root_directory', type=str, help='训练集根目录路径')
parser.add_argument('--val_root_directory', type=str, help='验证集根目录路径')
parser.add_argument('--test_root_directory', type=str, help='测试集根目录路径')
parser.add_argument('--pretrain', type=bool, default=True, help='是否使用预训练模型')
parser.add_argument('--pretrain_path', type=str, default='./model_save/MPBD_reg/best.pth', help='预训练模型的路径')
parser.add_argument('--resume', type=bool, default=True, help='是否从断点恢复训练')
parser.add_argument('--resume_from', type=str, default='/data/fog/fog_recognition/pretrian/10_29_best_2cls.pth', help='断点文件的路径')
parser.add_argument('--save_dir', type=str, default='./model_save', help='模型保存路径')

parser.add_argument('--config', type=str, help='config 路径')

# 解析命令行参数
args = parser.parse_args()

# 存储与默认值不同的参数键值对
updated_args = {}

# 存储配置文件中的参数键值对
config_args = {}

# 更新参数
for key, value in vars(args).items():
    if value != parser.get_default(key):
        updated_args[key] = value

config_path = args.config
if config_path:
    config_args = load_config_from_yaml(config_path)
else:
    config_args = {}

# 更新args字典，后出现的键值对覆盖前面出现的
args_dict = vars(args)
args_dict.update(config_args)
args_dict.update(updated_args)


if args.object_type == 'reg_cls':
    args.object_type='cls_reg'

# 设置随机数种子
seed_value = 2024
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True

device="cuda:0" if torch.cuda.is_available() else "cpu"

if args.object_type == 'reg':
    columns=args.reg_columns
if args.object_type == 'cls':
    columns=args.cls_columns
if args.object_type == 'cls_reg' or args.object_type == 'reg_cls':
    columns=args.reg_columns+args.cls_columns


class CustomDataset(Dataset):
    def __init__(self, X, y, augmentation_config=None, is_training=True):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.is_training = is_training
        
        # 创建数据增强器（只在训练时使用）
        self.augmentation = None
        if is_training and augmentation_config:
            self.augmentation = create_augmentation(augmentation_config)
            print(f"数据增强已启用，配置: {len(augmentation_config.get('methods', {}))} 种方法")
        elif is_training:
            print("未配置数据增强，将不进行数据增强")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        # 在训练时应用数据增强
        if self.is_training and self.augmentation:
            x = self.augmentation(x)
            
        return x, y

mk_dir(args.save_dir+'/'+args.object_name)

# 加载数据
X_train = np.load(os.path.join(args.data_dir, "train/X_train.npy"))
X_val = np.load(os.path.join(args.data_dir, "val/X_val.npy"))



if args.object_type == 'reg':
    y_train = pd.read_csv(os.path.join(args.data_dir, "train/y_train.csv"), index_col=0)[columns].values
    y_val = pd.read_csv(os.path.join(args.data_dir, "val/y_val.csv"), index_col=0)[columns].values
    y_train = (y_train - args.y_mean) / args.y_std
    y_val = (y_val - args.y_mean) / args.y_std

if args.object_type == 'cls':
    y_train = pd.read_csv(os.path.join(args.data_dir, "train/y_train.csv"), index_col=0)[columns].values
    y_val = pd.read_csv(os.path.join(args.data_dir, "val/y_val.csv"), index_col=0)[columns].values
    #转化为onehot
    y_train = np.eye(len(args.cls_dict))[y_train].reshape(-1,len(args.cls_dict) )
    y_val = np.eye(len(args.cls_dict))[y_val].reshape(-1,len(args.cls_dict) )

if args.object_type == 'cls_reg' or args.object_type == 'reg_cls':
    y_train_reg=pd.read_csv(os.path.join(args.data_dir, "train/y_train.csv"), index_col=0)[args.reg_columns].values
    y_val_reg=pd.read_csv(os.path.join(args.data_dir, "val/y_val.csv"), index_col=0)[args.reg_columns].values
    y_train_cls=pd.read_csv(os.path.join(args.data_dir, "train/y_train.csv"), index_col=0)[args.cls_columns].values
    y_val_cls=pd.read_csv(os.path.join(args.data_dir, "val/y_val.csv"), index_col=0)[args.cls_columns].values
    y_train_reg = (y_train_reg - args.y_mean) / args.y_std
    y_val_reg = (y_val_reg - args.y_mean) / args.y_std
    y_train_cls = np.eye(len(args.cls_dict))[y_train_cls].reshape(-1,len(args.cls_dict) )
    y_val_cls = np.eye(len(args.cls_dict))[y_val_cls].reshape(-1,len(args.cls_dict) )
    y_train=np.concatenate([y_train_reg,y_train_cls],axis=1)
    y_val=np.concatenate([y_val_reg,y_val_cls],axis=1)
    


# 创建训练集和验证集的 Dataset
# 从配置中获取数据增强参数
augmentation_config = args.data_augmentation

train_dataset = CustomDataset(X_train, y_train, augmentation_config=augmentation_config, is_training=True)
val_dataset = CustomDataset(X_val, y_val, is_training=False)  # 验证集不使用数据增强

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

#模型选择
if args.model['name']=='RRNet':
    model = RRNet(object_type=args.object_type,
                  num_cls=len(args.cls_dict),
                  num_reg=len(args.reg_columns),
                  list_inplanes=args.model['list_inplanes'],
                  len_spectrum=args.model['len_spectrum'],
                  sequence_len=args.model['sequence_len'],
                  mode=args.mode).to(device)
if args.model['name']=='MPBDNet':
    model = MPBDNet(object_type=args.object_type,
                    num_cls=len(args.cls_dict),
                    list_inplanes=args.model['list_inplanes'],
                    len_spectrum=args.model['len_spectrum'],
                    num_reg=len(args.reg_columns)).to(device)
if args.model['name']=='StarNet':
    model = StarNet(object_type=args.object_type,
                    num_cls=len(args.cls_dict),
                    num_reg=len(args.reg_columns),
                    len_spectrum=args.model.get('len_spectrum', 3834)).to(device)
if args.model['name']=='BGANet':
    model = BGANet(object_type=args.object_type,
                   num_cls=len(args.cls_dict),
                   num_reg=len(args.reg_columns),
                   len_spectrum=args.model['len_spectrum'],
                   ).to(device)


lr=args.lr


from models.create.create_loss import create_criterion
from models.create.create_scheduler import create_scheduler
criterion=create_criterion(args.object_type, args.loss)

optimizer = optim.AdamW(model.parameters(), lr=lr)

# 训练模型

def calculate_loss(criterion,outputs, targets):  
    if args.object_type == 'reg':
        return criterion['reg'](outputs['reg'], targets)
    if args.object_type == 'cls':

        return criterion['cls'](outputs['cls'], targets)
    if args.object_type == 'cls_reg' or args.object_type == 'reg_cls':
        # targets是一个二维数组，前半部分是回归目标，后半部分是分类目标
        # outputs['reg']是回归输出，outputs['cls']是分类输出
        # targets[:, :len(args.reg_columns)]是回归目标，targets[:, len(args.reg_columns):]是分类目标
        if 'reg' not in outputs or 'cls' not in outputs:
            raise ValueError("Outputs must contain 'reg' and 'cls' keys for cls_reg object_type.")

        reg_loss = criterion['reg'](outputs['reg'], targets[:, :len(args.reg_columns)])

        cls_loss = criterion['cls'](outputs['cls'], targets[:, len(args.reg_columns):].argmax(dim=1))
        return reg_loss + cls_loss

def train(args):
    best_loss = 1000
    scheduler = create_scheduler(optimizer, args)
    for epoch in tqdm(range(args.num_epochs)):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = calculate_loss(criterion, outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        test_label_reg = []
        test_label_cls = []
        test_pre_reg = []
        test_pre_cls = []
        
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                if args.object_type == 'cls_reg':
                    test_label_reg.append(y[:, :len(args.reg_columns)].cpu().numpy())
                    test_label_cls.append(y[:, len(args.reg_columns):].cpu().numpy())
                if args.object_type == 'reg':
                    test_label_reg.append(y.cpu().numpy())
                if args.object_type == 'cls':
                    test_label_cls.append(y.cpu().numpy())
                
                if 'reg' in y_pred:
                    test_pre_reg.append(y_pred['reg'].cpu().numpy())
                if 'cls' in y_pred:
                    test_pre_cls.append(y_pred['cls'].cpu().numpy())

        # 拼接所有批次的结果
        test_label_reg = np.concatenate(test_label_reg, axis=0) if test_label_reg else np.array([])
        test_label_cls = np.concatenate(test_label_cls, axis=0) if test_label_cls else np.array([])
        test_pre_reg = np.concatenate(test_pre_reg, axis=0) if test_pre_reg else np.array([])
        test_pre_cls = np.concatenate(test_pre_cls, axis=0) if test_pre_cls else np.array([])
        val_loss = 0
        # 计算验证损失
        if args.object_type in ['reg', 'reg_cls', 'cls_reg']:
            reg_loss = criterion['reg'](torch.from_numpy(test_pre_reg), torch.from_numpy(test_label_reg)).item()
            val_loss+= reg_loss* args.loss['reg_loss']['rate']
        elif args.object_type in ['cls', 'reg_cls', 'cls_reg']:
            cls_loss = criterion['cls'](torch.from_numpy(test_pre_cls), torch.from_numpy(test_label_cls.argmax(axis=1))).item()
            val_loss+= cls_loss* args.loss['cls_loss']['rate']


        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            if args.object_type == 'cls':
                best_acc = (test_label_cls.argmax(1) == test_pre_cls.argmax(1)).mean()
            if args.object_type == 'reg':
                best_mae = np.abs(test_label_reg - test_pre_reg).mean(axis=0)
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.object_name, 'best.pth'))

        # 反标准化回归结果
        if args.object_type == 'reg':
            test_label_reg = test_label_reg * args.y_std + args.y_mean
            test_pre_reg = test_pre_reg * args.y_std + args.y_mean
        elif args.object_type == 'cls_reg':
            test_label_reg = test_label_reg * args.y_std + args.y_mean
            test_pre_reg = test_pre_reg * args.y_std + args.y_mean

        # 输出评估指标
        print('\nval loss:', val_loss)
        if args.object_type in  ['cls', 'cls_reg','reg_cls']:
            # 获取预测和真实标签
            y_true = test_label_cls.argmax(1)
            y_pred = test_pre_cls.argmax(1)
            
            # 计算各种指标
            from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
            accuracy = (y_true == y_pred).mean()
            
            # CEMP星（标签1）作为正样本的指标
            cemp_label = 1  # CEMP对应标签1
            # 使用pos_label参数直接计算CEMP类的二分类指标
            y_true_binary = (y_true == cemp_label).astype(int)
            y_pred_binary = (y_pred == cemp_label).astype(int)
            precision_cemp = precision_score(y_true_binary, y_pred_binary, zero_division='warn')
            recall_cemp = recall_score(y_true_binary, y_pred_binary, zero_division='warn')
            f1_cemp = f1_score(y_true_binary, y_pred_binary, zero_division='warn')
            
            # 整体加权平均指标
            precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division='warn')
            recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division='warn')
            f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division='warn')
            
            print('val acc:', accuracy)
            print('=== CEMP星（正样本）指标 ===')
            print('CEMP precision:', precision_cemp)
            print('CEMP recall:', recall_cemp)
            print('CEMP f1:', f1_cemp)
            print('=== 整体加权平均指标 ===')
            print('weighted precision:', precision_weighted)
            print('weighted recall:', recall_weighted)
            print('weighted f1:', f1_weighted)
        if args.object_type in ['reg', 'cls_reg','reg_cls']:
            mae = np.abs(test_label_reg - test_pre_reg).mean(axis=0)
            print('val mae:', mae)
        # 更新学习率调度器
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)  # Plateau需要验证损失作为参数
            else:
                scheduler.step()  # 其他调度器直接step
            print(f'学习率: {optimizer.param_groups[0]["lr"]}')

if __name__ == '__main__':
    train(args)