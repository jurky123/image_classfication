#!/usr/bin/env python3
"""
快速开始 - ConvNeXt 迁移学习示例
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.convnext_model import ConvNeXtClassifier, print_model_info
from src.data.preprocessing import get_train_transforms, get_val_transforms


def example_model_creation():
    """示例 1: 创建模型"""
    print("\n" + "="*70)
    print("示例 1: 创建 ConvNeXt-Tiny 模型")
    print("="*70)
    
    # 创建模型
    model = ConvNeXtClassifier(
        num_classes=102,
        variant='tiny',
        pretrained=True,
        freeze_backbone=True
    )
    
    # 显示模型信息
    model.summary()
    
    # 显示当前的梯度更新状态
    print("\n当前梯度更新情况:")
    print(f"  骨干网络可训练: {model.model.features[0][0].weight.requires_grad}")
    print(f"  分类头可训练: {model.model.classifier[-1].weight.requires_grad}")
    
    return model


def example_unfreeze_backbone():
    """示例 2: 解冻骨干网络"""
    print("\n" + "="*70)
    print("示例 2: 解冻骨干网络")
    print("="*70)
    
    model = ConvNeXtClassifier(
        num_classes=102,
        variant='tiny',
        pretrained=True,
        freeze_backbone=True
    )
    
    print("\n初始状态 - 骨干网络冻结:")
    print(f"  总参数: {model.get_total_params():,}")
    print(f"  可训练参数: {model.get_trainable_params():,}")
    
    # 解冻最后 2 个 stage
    model.unfreeze_backbone(num_stages_to_unfreeze=2)
    
    print("\n解冻后期 2 个阶段:")
    print(f"  总参数: {model.get_total_params():,}")
    print(f"  可训练参数: {model.get_trainable_params():,}")
    
    return model


def example_data_loading():
    """示例 3: 加载数据"""
    print("\n" + "="*70)
    print("示例 3: 加载数据集")
    print("="*70)
    
    # 数据转换
    train_transforms = get_train_transforms(image_size=224, augment=True)
    val_transforms = get_val_transforms(image_size=224)
    
    # 创建数据集
    try:
        train_dataset = ImageFolder('data/train', transform=train_transforms)
        val_dataset = ImageFolder('data/val', transform=val_transforms)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        print(f"\n✓ 数据加载成功")
        print(f"  训练集: {len(train_dataset)} 张图片")
        print(f"  验证集: {len(val_dataset)} 张图片")
        print(f"  类别数: {len(train_dataset.classes)}")
        
        # 显示一个批次
        images, labels = next(iter(train_loader))
        print(f"\n批次信息:")
        print(f"  图片形状: {images.shape}")
        print(f"  标签形状: {labels.shape}")
        
        return train_loader, val_loader
    
    except FileNotFoundError:
        print("\n✗ 找不到数据目录，请确保已运行 prepare_data.py")
        return None, None


def example_forward_pass():
    """示例 4: 前向传播"""
    print("\n" + "="*70)
    print("示例 4: 前向传播测试")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    model = ConvNeXtClassifier(
        num_classes=102,
        variant='tiny',
        pretrained=True,
        freeze_backbone=True
    ).to(device)
    
    model.eval()
    
    # 创建随机输入
    dummy_input = torch.randn(8, 3, 224, 224).to(device)
    
    print(f"\n输入形状: {dummy_input.shape}")
    print(f"设备: {device}")
    
    # 前向传播
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"输出形状: {output.shape}")
    print(f"输出示例 (前3个样本的前5个类): \n{output[:3, :5]}")
    
    return model


def example_training_setup():
    """示例 5: 训练设置"""
    print("\n" + "="*70)
    print("示例 5: 训练设置")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    model = ConvNeXtClassifier(
        num_classes=102,
        variant='tiny',
        pretrained=True,
        freeze_backbone=True
    ).to(device)
    
    # 损失函数
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()
    
    # 优化器 - 只优化可训练参数
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=30,
        eta_min=1e-6
    )
    
    print(f"\n✓ 训练设置完成")
    print(f"  模型: ConvNeXt-Tiny")
    print(f"  设备: {device}")
    print(f"  损失函数: CrossEntropyLoss")
    print(f"  优化器: AdamW (lr=1e-4, weight_decay=1e-5)")
    print(f"  学习率调度: CosineAnnealingLR (T_max=30)")
    print(f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model, criterion, optimizer, scheduler


def print_quick_start_guide():
    """打印快速开始指南"""
    print("\n" + "="*70)
    print("🚀 快速开始指南")
    print("="*70)
    
    guide = """
1️⃣  准备数据:
    python scripts/prepare_data.py --clean

2️⃣  查看数据探索:
    jupyter notebook notebooks/data_exploration.ipynb

3️⃣  开始训练 (ConvNeXt-Tiny):
    python scripts/train_convnext.py --variant tiny --epochs 30 --batch-size 32

4️⃣  训练选项:
    --variant {tiny, small, base}    模型大小
    --epochs NUM                      训练轮数 (推荐 30-50)
    --batch-size SIZE                 批大小 (推荐 16-64)
    --lr LEARNING_RATE               学习率 (推荐 1e-4)
    --unfreeze-at EPOCH              何时解冻骨干网络 (推荐 epoch 10)

5️⃣  完整训练命令示例:
    # 使用 ConvNeXt-Tiny，默认设置
    python scripts/train_convnext.py
    
    # 使用 ConvNeXt-Small，50 个 epoch
    python scripts/train_convnext.py --variant small --epochs 50
    
    # 自定义学习率和批大小
    python scripts/train_convnext.py --lr 2e-4 --batch-size 64

6️⃣  模型信息:
    ConvNeXt-Tiny:  28.6M 参数, ImageNet 准确率 82.1%
    ConvNeXt-Small: 50.2M 参数, ImageNet 准确率 83.6%
    ConvNeXt-Base:  88.6M 参数, ImageNet 准确率 84.4%

7️⃣  训练策略:
    第 1-10 个 epoch: 冻结骨干网络，只训练分类头
    第 10+ 个 epoch: 解冻最后 2 个阶段，联合训练
    学习率: 使用 CosineAnnealingLR 自动衰减

8️⃣  输出文件:
    models/saved_models/best_model.pth    最佳模型
    models/checkpoints/                   训练检查点
    logs/training.log                     训练日志
    """
    
    print(guide)


def main():
    """主函数"""
    print("\n" + "="*70)
    print("ConvNeXt 迁移学习 - 快速开始示例")
    print("="*70)
    
    # 检查 CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✓ 设备: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA 版本: {torch.version.cuda}")
    
    # 运行示例
    try:
        # 示例 1: 模型创建
        model1 = example_model_creation()
        
        # 示例 2: 解冻骨干网络
        model2 = example_unfreeze_backbone()
        
        # 示例 3: 数据加载
        train_loader, val_loader = example_data_loading()
        
        # 示例 4: 前向传播
        model4 = example_forward_pass()
        
        # 示例 5: 训练设置
        model5, criterion, optimizer, scheduler = example_training_setup()
        
    except Exception as e:
        print(f"\n✗ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 打印指南
    print_quick_start_guide()
    
    print("\n✨ 现在可以开始训练了！\n")


if __name__ == '__main__':
    main()
