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
from models import RRNet
from models import MPBDNet,StarNet,BGANet,ResNet
from models.loss import FocalLoss

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 创建解析器
parser = argparse.ArgumentParser(description='train')

# 添加参数
parser.add_argument('--object_name', type=str, default='没有名字', help='名称')
parser.add_argument('--object_type', type=str, default='cls', help='cls,reg,cls_reg,reg_cls')
parser.add_argument('--num_classes', type=int, help='类别数量')
parser.add_argument('--num_epochs', type=int, default=100, help='训练的轮数')
parser.add_argument('--train_root_directory', type=str, help='训练集根目录路径')
parser.add_argument('--val_root_directory', type=str, help='验证集根目录路径')
parser.add_argument('--test_root_directory', type=str, help='测试集根目录路径')
parser.add_argument('--pretrain', type=bool, default=True, help='是否使用预训练模型')
parser.add_argument('--pretrain_path', type=str, default='./model_save/MPBD_reg/best.pth', help='预训练模型的路径')
parser.add_argument('--resume', type=bool, default=True, help='是否从断点恢复训练')
parser.add_argument('--num_noise_block', type=int, default=9, help='噪音块数量')
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



# 设置随机数种子
seed_value = 2024
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True

device="cuda:0" if torch.cuda.is_available() else "cpu"


y_mean = args.y_mean
y_std = args.y_mean
lr=args.lr
if args.object_type == 'reg':
    columns=args.reg_columns
    y_size=len(args.reg_columns)
if args.object_type == 'cls':
    columns=args.cls_columns
    y_size=args.num_classes
if args.object_type == 'cls_reg':
    columns=args.reg_columns+args.cls_columns
    y_size=len(args.reg_columns)+args.num_classes



class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

mk_dir(args.save_dir+'/'+args.object_name)

# 加载数据
X_train = np.load(os.path.join(args.data_dir, "train/X_train.npy"))
X_val = np.load(os.path.join(args.data_dir, "val/X_val.npy"))
y_train = pd.read_csv(os.path.join(args.data_dir, "train/y_train.csv"), index_col=0)[columns].values
y_val = pd.read_csv(os.path.join(args.data_dir, "val/y_val.csv"), index_col=0)[columns].values


if args.object_type == 'reg':
    y_train = pd.read_csv(os.path.join(args.data_dir, "train/y_train.csv"), index_col=0)[columns].values
    y_val = pd.read_csv(os.path.join(args.data_dir, "val/y_val.csv"), index_col=0)[columns].values
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std

if args.object_type == 'cls':
    y_train = pd.read_csv(os.path.join(args.data_dir, "train/y_train.csv"), index_col=0)[columns].values
    y_val = pd.read_csv(os.path.join(args.data_dir, "val/y_val.csv"), index_col=0)[columns].values
    #转化为onehot
    y_train = np.eye(args.num_classes)[y_train].reshape(-1,args.num_classes)
    y_val = np.eye(args.num_classes)[y_val].reshape(-1,args.num_classes)

# 创建训练集和验证集的 Dataset
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)



#损失函数选择
if args.object_type == 'cls':
    if args.loss=='FocalLoss':
        criterion = FocalLoss(gamma=2)
    if args.loss=='CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
if args.object_type == 'reg':
    criterion = nn.MSELoss()



# 训练模型
num_epochs = 200

best_loss=1000
val_loss_list=[]
val_acc_list=[]
for i in range(args.num_noise_block):
    #模型选择
    if args.model=='RRNet':
        model = RRNet(num_label=y_size,list_inplanes=args.list_inplanes,mode=args.mode).to(device)
    if args.model=='MPBDNet':
        model = MPBDNet(num_label=y_size,list_inplanes=args.list_inplanes).to(device)
    if args.model=='StarNet':
        model = StarNet(num_label=y_size).to(device)
    if args.model=='BGANet':
        model = BGANet(num_label=y_size,len_spectrum=3834).to(device)
    if args.model=='ResNet':
        model = ResNet(num_label=y_size,list_inplanes=args.list_inplanes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    print(X_train.shape[1])
    print(args.num_noise_block)
    assert X_train.shape[1]%args.num_noise_block==0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for inputs, targets in train_loader:
            nosie_lenth=len(inputs)//args.num_noise_block
            random_noise=torch.randn_like(inputs[:,:nosie_lenth])
            inputs, targets = inputs.to(device), targets.to(device)
            inputs[:,nosie_lenth*i:nosie_lenth*(i+1)]=random_noise
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            optimizer.step()
            if loss.item()<best_loss:
                best_loss=loss.item()
                torch.save(model.state_dict(), os.path.join(args.save_dir,args.object_name,'best.pth'))
        model.eval()
        test_label=[]
        test_pre=[]
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                test_label.append(y.cpu().numpy())
                test_pre.append(y_pred.cpu().numpy())
        test_label = np.concatenate(test_label, axis=0)
        test_pre = np.concatenate(test_pre, axis=0)
        val_loss=criterion(torch.from_numpy(test_label),torch.from_numpy(test_pre)).item()
        if val_loss<best_loss:
            best_loss=val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir,args.object_name,'best.pth'))
        if args.object_type == 'reg':
            test_label = test_label * args.y_std + args.y_mean
            test_pre = test_pre * args.y_std + args.y_mean
        if args.object_type == 'cls_reg':
            test_label = test_label[:,len(args.cls_columns):] * args.y_std + args.y_mean
            test_pre = test_pre[:,len(args.cls_columns):] * args.y_std + args.y_mean

        #输出混淆矩阵
        if args.object_type == 'cls':
            print('val acc:',(test_label.argmax(1)==test_pre.argmax(1)).mean())
            print('val loss:',val_loss)
            val_acc_list.append((test_label.argmax(1)==test_pre.argmax(1)).mean())
            val_loss_list.append(val_loss)

        else:
            print('val loss:',val_loss)
            print('val mae:',np.abs(test_label-test_pre).mean())