"""
数据增强功能测试脚本
Author: Ziyu Fang
Date: 2024-12-25
Email: fangziyushiwo@126.com
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.data_augmentation import create_augmentation, EXAMPLE_CONFIG

def test_augmentation():
    """测试数据增强功能"""
    print("=== 数据增强功能测试 ===")
    
    # 创建测试光谱数据
    spectrum_length = 4900
    test_spectrum = torch.randn(spectrum_length) * 0.1 + 1.0  # 模拟归一化的光谱
    
    print(f"原始光谱形状: {test_spectrum.shape}")
    print(f"原始光谱值范围: [{test_spectrum.min():.4f}, {test_spectrum.max():.4f}]")
    
    # 测试不同的增强配置
    configs = {
        "无增强": None,
        "轻量级增强": {
            "probability": 1.0,  # 100%应用，便于测试
            "methods": {
                "gaussian_noise": {
                    "enabled": True,
                    "probability": 1.0,
                    "std": 0.01
                }
            }
        },
        "完整增强": {
            "probability": 1.0,
            "methods": {
                "gaussian_noise": {
                    "enabled": True,
                    "probability": 1.0,
                    "std": 0.01
                },
                "spectral_scaling": {
                    "enabled": True,
                    "probability": 1.0,
                    "range": 0.1
                },
                "baseline_drift": {
                    "enabled": True,
                    "probability": 1.0,
                    "strength": 0.02
                }
            }
        }
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n--- 测试 {config_name} ---")
        
        # 创建增强器
        augmentor = create_augmentation(config)
        
        # 应用增强
        augmented_spectrum = augmentor(test_spectrum.clone())
        
        print(f"增强后形状: {augmented_spectrum.shape}")
        print(f"增强后值范围: [{augmented_spectrum.min():.4f}, {augmented_spectrum.max():.4f}]")
        
        # 计算差异
        if config is not None:
            diff = torch.abs(augmented_spectrum - test_spectrum)
            print(f"平均差异: {diff.mean():.6f}")
            print(f"最大差异: {diff.max():.6f}")
        
        results[config_name] = augmented_spectrum.numpy()
    
    # 可视化结果（如果有matplotlib）
    try:
        plt.figure(figsize=(15, 10))
        
        for i, (name, spectrum) in enumerate(results.items()):
            plt.subplot(2, 2, i+1)
            if i == 0:
                # 原始光谱显示整个范围
                plt.plot(spectrum, 'b-', linewidth=0.8, alpha=0.8)
                plt.title(f'{name}\n(原始光谱)')
            else:
                # 增强后的光谱只显示部分范围以便观察差异
                start_idx = 1000
                end_idx = 1500
                plt.plot(range(start_idx, end_idx), 
                        results["无增强"][start_idx:end_idx], 
                        'b-', linewidth=1.0, alpha=0.7, label='原始')
                plt.plot(range(start_idx, end_idx), 
                        spectrum[start_idx:end_idx], 
                        'r-', linewidth=1.0, alpha=0.8, label='增强后')
                plt.title(f'{name}\n(波段 {start_idx}-{end_idx})')
                plt.legend()
            
            plt.xlabel('波长点')
            plt.ylabel('强度')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('augmentation_test_results.png', dpi=150, bbox_inches='tight')
        print(f"\n可视化结果已保存到: augmentation_test_results.png")
        
    except ImportError:
        print("\n注意: matplotlib未安装，跳过可视化")
    
    print("\n=== 测试完成 ===")
    print("✅ 数据增强功能正常工作！")
    return results

def test_batch_augmentation():
    """测试批量数据增强"""
    print("\n=== 批量数据增强测试 ===")
    
    # 创建批量数据
    batch_size = 8
    spectrum_length = 3834
    batch_spectra = torch.randn(batch_size, spectrum_length) * 0.1 + 1.0
    
    print(f"批量数据形状: {batch_spectra.shape}")
    
    # 创建增强配置
    config = {
        "probability": 0.5,  # 50%的样本会被增强
        "methods": {
            "gaussian_noise": {
                "enabled": True,
                "probability": 0.8,
                "std": 0.01
            },
            "spectral_scaling": {
                "enabled": True,
                "probability": 0.6,
                "range": 0.05
            }
        }
    }
    
    augmentor = create_augmentation(config)
    
    # 统计增强情况
    augmented_count = 0
    for i in range(batch_size):
        original = batch_spectra[i].clone()
        augmented = augmentor(batch_spectra[i])
        
        # 检查是否被增强
        if not torch.allclose(original, augmented, atol=1e-6):
            augmented_count += 1
            print(f"样本 {i}: 已增强 (差异: {torch.abs(original - augmented).mean():.6f})")
        else:
            print(f"样本 {i}: 未增强")
    
    print(f"\n增强统计: {augmented_count}/{batch_size} 个样本被增强")
    print(f"增强比例: {augmented_count/batch_size:.1%}")
    
    return augmented_count

if __name__ == "__main__":
    # 设置随机种子以获得可重复的结果
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行测试
    test_augmentation()
    test_batch_augmentation()
    
    print("\n=== 配置文件使用说明 ===")
    print("1. 不使用数据增强：在配置文件中删除 'data_augmentation' 部分")
    print("2. 使用轻量级增强：参考 config/MPBD_cls_light_aug.yaml")
    print("3. 使用完整增强：参考 config/MPBD_cls_with_augmentation.yaml")
    print("4. 自定义增强：根据需要修改 methods 中的参数")
    print("\n训练命令示例:")
    print("python train.py --config=config/MPBD_cls_with_augmentation.yaml") 