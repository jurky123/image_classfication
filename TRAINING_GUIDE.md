# ConvNeXt 迁移学习训练指南

## 📋 项目概述

本项目使用 **ConvNeXt-Tiny** 预训练模型对 Oxford Flowers 102 数据集进行图像分类。采用两阶段迁移学习策略：
1. **第一阶段（冻结）**：固定骨干网络，只训练分类头（前10个epoch）
2. **第二阶段（微调）**：解冻最后几层，进行完整微调（后20个epoch）

---

## 🚀 快速开始

### 1. 环境配置

```bash
# 安装所需依赖
pip install torch torchvision timm scikit-learn tqdm pyyaml

# 验证安装
python test_convnext_setup.py
```

### 2. 数据准备

```bash
# 如果还未整理数据，先运行数据准备脚本
python scripts/prepare_data.py --clean

# 检查数据是否准备好
ls data/train/  # 应该包含 102 个文件夹（每个类别一个）
```

### 3. 开始训练

```bash
# 使用默认参数训练（ConvNeXt-Tiny）
python scripts/train_convnext.py

# 使用自定义参数
python scripts/train_convnext.py \
    --variant tiny \
    --batch_size 32 \
    --epochs 30 \
    --lr 1e-4 \
    --device cuda
```

---

## 📊 训练参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--variant` | tiny | 模型大小：tiny/small/base |
| `--batch_size` | 32 | 批次大小 |
| `--epochs` | 30 | 总训练轮数 |
| `--lr` | 1e-4 | 学习率 |
| `--unfreeze_epoch` | 10 | 第几个epoch开始解冻 |
| `--device` | cuda | 计算设备：cuda/cpu |
| `--num_workers` | 4 | 数据加载工作进程数 |

---

## 🎯 模型对比

| 模型 | 参数量 | ImageNet精度 | 推荐场景 |
|------|-------|-----------|---------|
| **Tiny** | 28.6M | 82.1% | 快速训练，资源有限 |
| **Small** | 50.2M | 83.6% | 平衡性能和速度 |
| **Base** | 88.6M | 84.6% | 最高精度 |

---

## 📈 两阶段训练策略详解

### 第一阶段：冻结学习（Epochs 1-10）

```
骨干网络（Backbone）：✓ 固定（不更新权重）
分类头（Head）：✗ 训练（更新权重）
学习率：1e-4
目的：快速调整分类头以适应新的102个类别
```

**优点：**
- 避免过度调整预训练特征
- 快速收敛（预训练特征已很好）
- 减少梯度计算（只更新分类头）

### 第二阶段：微调（Epochs 10-30）

```
骨干网络（Backbone）：✗ 解冻最后2个Stage
分类头（Head）：✗ 继续训练
学习率：1e-4（自动递减）
目的：微调预训练特征以更好适应新任务
```

**优点：**
- 提高特征表示能力
- 更好地捕获数据集特有特征
- CosineAnnealing学习率调度防止发散

---

## 📊 预期性能

### ConvNeXt-Tiny 在 Oxford Flowers 102 上

```
冻结阶段（10个epoch）
├─ 第1-2个epoch：快速上升到 ~70% 准确率
├─ 第3-5个epoch：稳定在 ~80% 准确率
├─ 第10个epoch：达到 ~85% 准确率
└─ 训练时间：约 20-30 分钟

微调阶段（20个epoch）
├─ 第11-15个epoch：继续上升到 ~88%
├─ 第16-20个epoch：平稳到 ~90%
├─ 第21-30个epoch：微调到 ~91-92%
└─ 总训练时间：约 50-70 分钟
```

---

## 💾 输出文件说明

训练完成后，您将获得以下文件：

```
models/
├─ saved_models/
│  └─ best_model.pth          ← 最佳模型（验证集最高准确率）
├─ checkpoints/
│  ├─ epoch_0.pth
│  ├─ epoch_5.pth
│  └─ epoch_29.pth            ← 各个epoch的检查点
└─ logs/
   ├─ training.log            ← 训练日志
   └─ training_history_*.json  ← 详细的训练历史
```

---

## 📊 监控训练进度

### 实时监控

```bash
# 在另一个终端查看日志
tail -f models/logs/training.log

# 监控GPU使用情况（如有GPU）
nvidia-smi -l 1  # 每秒刷新
```

### 分析训练历史

```python
import json
import matplotlib.pyplot as plt

# 加载训练历史
with open('models/logs/training_history_*.json', 'r') as f:
    history = json.load(f)

# 绘制学习曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()
```

---

## 🔧 常见问题

### Q1: 如何在CPU上训练？
```bash
python scripts/train_convnext.py --device cpu
```

### Q2: 训练中断后如何恢复？
```python
from src.models.convnext_model import ConvNeXtClassifier
from torch import load

# 加载检查点
model = ConvNeXtClassifier(num_classes=102, variant='tiny')
checkpoint = load('models/checkpoints/epoch_15.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 从第16个epoch继续训练...
```

### Q3: 如何调整批次大小？
```bash
# 减小批次大小（GPU内存有限）
python scripts/train_convnext.py --batch_size 16

# 增大批次大小（提高训练速度）
python scripts/train_convnext.py --batch_size 64
```

### Q4: 模型在验证集上过度拟合怎么办？
```bash
# 增加数据增强强度（修改 src/data/preprocessing.py）
# 减小学习率
python scripts/train_convnext.py --lr 5e-5

# 使用更大的模型正则化或dropout
```

### Q5: 如何使用训练好的模型进行推理？
```python
import torch
from src.models.convnext_model import ConvNeXtClassifier
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image

# 加载模型
model = ConvNeXtClassifier(num_classes=102, variant='tiny', pretrained=False)
checkpoint = torch.load('models/saved_models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 准备输入
image = Image.open('path/to/image.jpg')
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], 
             [0.229, 0.224, 0.225])
])
x = transform(image).unsqueeze(0)

# 推理
with torch.no_grad():
    output = model(x)
    probabilities = torch.softmax(output, dim=1)
    top_class = probabilities.argmax(dim=1)

print(f"预测类别: {top_class.item()}")
print(f"信心: {probabilities.max().item():.2%}")
```

---

## 🎓 学习资源

- [ConvNeXt 原论文](https://arxiv.org/abs/2201.03545)
- [PyTorch 迁移学习教程](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [timm 库文档](https://github.com/rwightman/pytorch-image-models)
- [Oxford Flowers 102 数据集](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

---

## 📞 支持

如遇问题，请检查：
1. ✓ 依赖已安装：`python test_convnext_setup.py`
2. ✓ 数据已准备：`ls data/train/ | wc -l` （应该是 102）
3. ✓ 模型文件存在：`ls src/models/`
4. ✓ GPU可用（可选）：`python -c "import torch; print(torch.cuda.is_available())"`

---

## 📝 训练检查清单

- [ ] 依赖安装完成
- [ ] 数据已准备（train/val/test 文件夹）
- [ ] 模型代码无语法错误
- [ ] GPU/CPU 资源充足
- [ ] 设置了合适的超参数
- [ ] 创建了 logs 和 checkpoints 目录
- [ ] 运行了第一个训练epoch
- [ ] 监控了训练日志输出

开始训练：
```bash
python scripts/train_convnext.py
```

祝您训练顺利！🎉
