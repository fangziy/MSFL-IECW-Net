# 

## 项目简介

## 环境准备

```

conda create -n fog_recognition python=3.12 pytorch==2.2.2 cudatoolkit=11.8 

pip install -r requirements.txt

```

## 项目结构

```

project/

├── data/         # 数据集

     ├── spectra/

     ├── train/

     └── val/

├── utils/        

├── models/         

├── train.py   

├── val.py   

└── README.md     # 项目说明文档

```

## 数据集准备

数据集为一个csv表格数据。

可以通过在配置文件中指定features_name来区分X和，y。其中最后一列为y，其余列为X。

## 配置文件

```
#项目名称
object_name: 'MPBD_reg'
数据路径
data_dir : "./data"

#任务类型
object_type: 'reg'
#回归部分列名
reg_columns : ["FeH", "CH", "Teff", "logg", "CFe"]
#分类部分列名
cls_columns: ['f_CEMP']
cls_dict: {0: 'MP', 1: 'CEMP', 2: 'NMP'}
num_classes: 3

#回归标签归一化
y_mean : [-8.81020067e-01, -9.78005818e-01, 4.91095684e+03, 2.55795385e+00, -9.69857512e-02]
y_std : [7.61077821e-01, 8.42965361e-01, 4.97515564e+02, 1.15882902e+00, 3.06636238e-01]

#模型选择
model: 'MPBDNet'
#模型参数选择
list_inplanes: [20,40,80,160]
lr: 0.0001
#损失函数
loss: 'MSELoss'
#继续训练
resume_from : 'D:/Notebook_workdir/thesis/model_save/MPBD_reg/best.pth'
```

## 训练

```

#config中给出配置文件的路径
python train.py --config=config/fog_prediction.yaml

```

## 验证

```

python val.py --config=config/fog_prediction.yaml

```
