# 恒星参数估计及CEMP星搜寻

## 项目简介

大天区面积多目标光纤光谱望远镜 (Large Sky Area Multi-Object Fiber Spectroscopic Telescope, LAMOST) 是我国自主研发设计的大型光学波段观测设备，为天文学研究提供了关键的数据支持。LAMOST提供了大量的观测光谱数据，从中筛选出稀有且具有重要科学价值的目标具有重要意义。贫金属(Metal-Poor， MP)星是一类稀有而古老的恒星，而碳超丰贫金属(Carbon-Enhanced Metal-Poor， CEMP)星则是其中的一个子类，它们显示出相对于铁元素的碳富集效应。CEMP星被认为是由大爆炸后第一代恒星污染的气体形成的，它们是研究早期宇宙、星系演化和核合成的重要对象。关于CEMP的研究对于研究早期宇宙、星系演化和核合成有着重要意义。因此对于MP星及 CEMP星的寻找及其理化参数的估计具有重要的研究价值。然而，由于MP及CEMP样本的稀缺性，传统方法难以对这两类样本的特征进行充分的学习，因此对于这两类样本的检出率及参数估计的精度均较低。为解决此问题，本文根据MP及CEMP样本在恒星物参数空间中的样本分布特点及类别分布的非平衡性，相应地设计了}深度是学习方法来探讨低分辨率恒星光谱的恒星物理参数估计及恒星的分类问题，并从以下两个方面展开了深入研究。本文贡献具体如下：
(1) 本工作构建了一个用于CEMP星及MP星搜索的参考数据集并提出了相应的数据预处理方法。本工作的数据集使用了LAMOST DR8 中的4723 个非贫金属星样本，以及 5032 个贫金属星样
本构建。在贫金属星样本中，有 167 个 CEMP 星观测样本。其中$T_\texttt{eff}$,$\log~g$, [Fe/H], [C/H]标签来自APOGEE DR17恒星目录, LAMOST-Subaru恒星目录,和SAGA数据库。本文将原始光谱数据插值到相同的波段范围并按照统一的标准采样。采样后的光谱经过连续谱拟合并归一化后将被输入进MSFL-IECW模型中进行训练测试。

 (2) 本工作基于分类方法提出了多尺度特征学习与跨波段信息挖掘网络（MSFL—IECW）。该网络使用多尺度的卷积层来提取不同尺度的特征并使用多个方向的LSTM用以结合跨波段信息。本方法对于MP的搜寻recall达到了0.9592，precision达到了0.9493，F1达到了0.9542，对于CEMP的搜寻recall达到了0.6428，precision达到了0.7826，F1达到了0.7058，总体的accuracy达到了0.949。本方案与ResNet，Inception，RRNet，StarNet，BAGNet等进行了比较，结果显示本方案的精度大幅领先于其他方案。

 (3) 本工作提出了一种基于分类约束的回归估计方法，在精确估计恒星参数的同时避免了参数分布与物理性质上的悖论。基于此，本方法对于$T_\texttt{eff}$, $\log~g$, [Fe/H], [C/H]的估计MAE分别达到了74.9127, 0.1749, 0.0889 ,0.1275, 误差均值分别为-2.1355, -0.0286, -0.0139, -0.0023, 误差方差分别为120.4321, 0.2599, 0.1496, 0.206。同时本工作基于参数估计结果对CEMP星进行了搜寻，实现了 0.9554 的准确率、0.6970 的召回率、0.8519 的精确率和 0.7667 的 F1 分数。

 (4) 在LAMOST获取光谱上的应用及其展望。本工作将研究的算法应用于LAMOST DR8低分辨率观测光谱来估算恒星参数并对MP星及CEMP星进行搜寻。通过这种方法，我们从LAMOST DR8的低分辨率恒星光谱数据库中，发现了 819,671 颗贫金属恒星候选体以及 12,766 颗富碳贫金属恒星候选体。在这些富碳贫金属恒星候选体中，有 9,461 颗极贫金属（VMP）恒星候选体和 164 颗极度贫金属（EMP）恒星候选体。这些技术未来可以用于我国建设的大型空间天文望远镜空间站多功能光学设施（CSST）的巡天项目，对未来的CEMP星搜寻提供一种可行的方案。

## 版权所有

```
Author: Ziyu Fang
Date: 2024-12-25
Email: fangziyushiwo@126.com
```

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
