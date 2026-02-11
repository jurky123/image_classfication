# ConvNeXt 迁移学习完整指南

## 📌 概述

你已配置好 **ConvNeXt-Tiny** 迁移学习方案，这是一个现代的 CNN 架构，在图像分类任务上表现优异。

## 🎯 ConvNeXt 模型特点

### ConvNeXt 架构
- 由 Meta AI 提出（2022年）
- 结合了 ResNet 和现代设计原则
- 在 ImageNet 上超越 Vision Transformer
- 优化了速度和准确率的平衡

### 三种规格

| 规格 | 参数量 | 计算量 | ImageNet准确率 | 推荐场景 |
|------|--------|--------|---------------|---------|
| **Tiny** | 28.6M | 4.5G | 82.1% | 轻量部署，快速训练 |
| **Small** | 50.2M | 8.7G | 83.6% | 平衡性能 |
| **Base** | 88.6M | 15.4G | 84.4% | 最高精度 |

## 📦 安装依赖

```bash
# 确保已安装必要的包
pip install torch torchvision timm tqdm

# 验证安装
python -c "import torch; import torchvision; print(f'PyTorch: {torch.__version__}')"
```

## 🚀 快速开始

### 1. 运行示例代码
```bash
# 查看 5 个使用示例
python examples/quick_start.py
```

这会展示：
- ✅ 模型创建
- ✅ 骨干网络解冻
- ✅ 数据加载
- ✅ 前向传播
- ✅ 训练设置

### 2. 开始训练

#### 基础训练（默认参数）
```bash
python scripts/train_convnext.py
```

#### 指定模型大小
```bash
# 使用 ConvNeXt-Tiny（默认）
python scripts/train_convnext.py --variant tiny

# 使用 ConvNeXt-Small（更好的精度）
python scripts/train_convnext.py --variant small

# 使用 ConvNeXt-Base（最高精度）
python scripts/train_convnext.py --variant base
```

#### 自定义超参数
```bash
python scripts/train_convnext.py \
    --variant tiny \
    --epochs 50 \
    --batch-size 64 \
    --lr 2e-4 \
    --unfreeze-at 10
```

## 💻 代码使用示例

### 示例 1: 创建模型
```python
from src.models.convnext_model import ConvNeXtClassifier

# 创建模型
model = ConvNeXtClassifier(
    num_classes=102,           # 类别数
    variant='tiny',            # 模型大小: tiny/small/base
    pretrained=True,           # 使用 ImageNet 预训练权重
    freeze_backbone=True       # 冻结骨干网络
)

# 显示模型信息
model.summary()
```

### 示例 2: 冻结和解冻
```python
# 冻结骨干网络（初始阶段）
model.freeze_backbone(freeze=True)

# 部分解冻最后 2 个阶段（后期训练）
model.unfreeze_backbone(num_stages_to_unfreeze=2)

# 获取参数统计
print(f"总参数: {model.get_total_params():,}")
print(f"可训练参数: {model.get_trainable_params():,}")
```

### 示例 3: 加载数据
```python
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from src.data.preprocessing import get_train_transforms, get_val_transforms

# 数据转换
train_transforms = get_train_transforms(image_size=224, augment=True)
val_transforms = get_val_transforms(image_size=224)

# 创建数据集
train_dataset = ImageFolder('data/train', transform=train_transforms)
val_dataset = ImageFolder('data/val', transform=val_transforms)

# 创建加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

### 示例 4: 完整训练循环
```python
import torch
import torch.nn as nn
import torch.optim as optim

# 创建模型、优化器等
model = ConvNeXtClassifier(num_classes=102, variant='tiny', pretrained=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# 训练循环
for epoch in range(num_epochs):
    # 训练
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 验证
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # 计算指标...
    
    scheduler.step()
```

## 🎓 迁移学习策略

### 推荐的两阶段训练

#### 第一阶段：冻结骨干网络（10 个 epoch）
- 只训练分类头
- 学习率: 1e-4
- 优势: 快速适应新任务，防止过拟合

```python
model.freeze_backbone(freeze=True)
# 训练 10 个 epoch
```

#### 第二阶段：解冻骨干网络（20 个 epoch）
- 解冻最后几个阶段，联合训练
- 学习率: 1e-5（更小）
- 优势: 微调预训练特征，获得更好精度

```python
model.unfreeze_backbone(num_stages_to_unfreeze=2)
# 使用更小的学习率继续训练
```

## 📊 性能指标

### 预期结果（Flowers 102）
使用 ConvNeXt-Tiny，应该能达到：
- Top-1 准确率: **88-92%**
- 训练时间: **1-2 小时**（单 GPU）

### 影响性能的因素
1. **模型大小**: Base > Small > Tiny（精度递减）
2. **训练轮数**: 30-50 个 epoch 通常足够
3. **学习率**: 太大易发散，太小收敛慢
4. **数据增强**: 显著提升泛化能力
5. **批大小**: 32-64 通常最优

## ⚙️ 超参数建议

### 学习率
```python
初始阶段: 1e-4    # 冻结骨干网络时
微调阶段: 1e-5    # 解冻骨干网络时
```

### 批大小
```python
GPU 内存 < 6GB:  batch_size=16
GPU 内存 6-12GB: batch_size=32
GPU 内存 > 12GB: batch_size=64
```

### 优化器
```python
# 推荐 AdamW（效果最好）
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5  # 重要：防止过拟合
)
```

### 学习率调度
```python
# CosineAnnealingLR（自动衰减）
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-6
)
```

## 🔧 常见问题

### Q1: 如何选择模型大小？
**A**: 
- **Tiny**: GPU 内存 < 4GB，需要快速训练
- **Small**: 一般用户，平衡精度和速度
- **Base**: 最求最高精度，有充足计算资源

### Q2: 为什么要冻结骨干网络？
**A**: 
1. 减少可训练参数，加快训练
2. 防止过拟合（数据集较小）
3. 保留预训练的通用特征
4. 后期解冻进一步微调

### Q3: 何时解冻骨干网络？
**A**: 通常在 epoch 10 左右，当验证集性能停止明显提升时

### Q4: 内存不足怎么办？
**A**: 
```python
# 方案 1: 减小批大小
batch_size = 16

# 方案 2: 使用梯度累积
accumulation_steps = 2

# 方案 3: 使用混合精度训练
from torch.cuda.amp import autocast
with autocast():
    output = model(images)
```

### Q5: 如何加快训练速度？
**A**:
1. 增加 `num_workers`（数据加载并行化）
2. 使用 `pin_memory=True`
3. 减小图片分辨率（如 192x192）
4. 混合精度训练（fp16）

## 📈 监控训练进度

### 查看训练日志
```bash
# 实时监控
tail -f logs/training.log

# 查看最近的训练
tail -20 logs/training.log
```

### 查看保存的模型
```bash
# 最佳模型
ls -lh models/saved_models/best_model.pth

# 所有检查点
ls -lh models/checkpoints/
```

## 🎯 完整训练脚本

推荐的完整命令：

```bash
# 用 ConvNeXt-Tiny 训练 30 个 epoch
python scripts/train_convnext.py \
    --variant tiny \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-4 \
    --unfreeze-at 10

# 用 ConvNeXt-Small 训练 50 个 epoch（更好精度）
python scripts/train_convnext.py \
    --variant small \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --unfreeze-at 15
```

## 📚 相关资源

- [ConvNeXt 论文](https://arxiv.org/abs/2201.03545)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [迁移学习指南](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [数据增强技术](https://pytorch.org/vision/stable/transforms.html)

## 🚦 检查清单

- [ ] 安装依赖 (`pip install -r requirements.txt`)
- [ ] 准备数据 (`python scripts/prepare_data.py`)
- [ ] 运行示例 (`python examples/quick_start.py`)
- [ ] 开始训练 (`python scripts/train_convnext.py`)
- [ ] 监控进度 (查看 logs/)
- [ ] 评估模型 (使用 best_model.pth)

---

**现在可以开始训练了！祝你成功！** 🎉
