# 图像分类项目完整使用指南
# Complete Image Classification Project Guide

## 目录 / Table of Contents

1. [项目概述 / Project Overview](#项目概述--project-overview)
2. [快速开始 / Quick Start](#快速开始--quick-start)
3. [项目架构详解 / Architecture Details](#项目架构详解--architecture-details)
4. [使用流程 / Usage Workflow](#使用流程--usage-workflow)
5. [配置说明 / Configuration Guide](#配置说明--configuration-guide)
6. [模块详解 / Module Details](#模块详解--module-details)
7. [扩展开发 / Extension Development](#扩展开发--extension-development)
8. [常见问题 / FAQ](#常见问题--faq)

---

## 项目概述 / Project Overview

### 项目目的 / Purpose

这是一个专业的图像分类深度学习框架，旨在提供：
- **模块化设计**：各功能模块独立，易于维护和扩展
- **标准化流程**：从数据准备到模型部署的完整流程
- **灵活配置**：通过配置文件管理所有超参数
- **可复现性**：固定随机种子，确保实验可重复

This is a professional deep learning framework for image classification that provides:
- **Modular Design**: Independent functional modules, easy to maintain and extend
- **Standardized Pipeline**: Complete workflow from data preparation to model deployment
- **Flexible Configuration**: Manage all hyperparameters through config files
- **Reproducibility**: Fixed random seeds to ensure repeatable experiments

### 适用场景 / Use Cases

- ✅ 图像分类任务 (Image classification tasks)
- ✅ 迁移学习应用 (Transfer learning applications)
- ✅ 模型性能对比 (Model performance comparison)
- ✅ 深度学习教学 (Deep learning education)

---

## 快速开始 / Quick Start

### 1. 环境安装 / Environment Setup

```bash
# 克隆项目 / Clone repository
git clone https://github.com/jurky123/image_classfication.git
cd image_classfication

# 创建虚拟环境 / Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 安装依赖 / Install dependencies
pip install -r requirements.txt

# 或者使用开发模式安装 / Or install in development mode
pip install -e .
```

### 2. 准备数据 / Prepare Data

```bash
# 数据目录结构应该如下 / Data directory structure should be:
data/
├── raw/              # 原始数据 / Raw data
├── train/            # 训练集 / Training set
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
│       ├── img3.jpg
│       └── img4.jpg
├── val/              # 验证集 / Validation set
│   ├── class1/
│   └── class2/
└── test/             # 测试集 / Test set
    ├── class1/
    └── class2/
```

### 3. 配置模型 / Configure Model

编辑 `configs/config.yaml` 文件来设置训练参数：

Edit `configs/config.yaml` to set training parameters:

```yaml
model:
  name: 'resnet50'        # 选择模型 / Choose model
  num_classes: 10         # 分类数量 / Number of classes
  pretrained: true        # 是否使用预训练权重 / Use pretrained weights

training:
  batch_size: 32          # 批次大小 / Batch size
  num_epochs: 50          # 训练轮数 / Number of epochs
  learning_rate: 0.001    # 学习率 / Learning rate
```

### 4. 开始训练 / Start Training

```bash
# 使用默认配置训练 / Train with default config
python scripts/train.py

# 使用自定义配置 / Train with custom config
python scripts/train.py --config configs/custom_config.yaml
```

### 5. 评估模型 / Evaluate Model

```bash
# 评估训练好的模型 / Evaluate trained model
python scripts/evaluate.py --model_path models/saved_models/best_model.pth

# 查看详细评估报告 / View detailed evaluation report
python scripts/evaluate.py --model_path models/saved_models/best_model.pth --verbose
```

### 6. 进行预测 / Make Predictions

```bash
# 单张图片预测 / Predict single image
python scripts/predict.py --image_path path/to/image.jpg --model_path models/saved_models/best_model.pth

# 批量预测 / Batch prediction
python scripts/predict.py --image_dir path/to/images/ --model_path models/saved_models/best_model.pth
```

---

## 项目架构详解 / Architecture Details

### 整体架构图 / Overall Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Image Classification Framework            │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Data Module │     │ Model Module │     │ Utils Module │
│              │     │              │     │              │
│ • DataLoader │────▶│ • BaseModel  │────▶│ • Logger     │
│ • Preprocess │     │ • CNN Models │     │ • Checkpoint │
│ • Augment    │     │ • Transfer   │     │ • Common     │
└──────────────┘     └──────────────┘     └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │  Training Loop   │
                    │                  │
                    │ • Train          │
                    │ • Validate       │
                    │ • Evaluate       │
                    │ • Predict        │
                    └──────────────────┘
```

### 目录结构说明 / Directory Structure Explanation

```
image_classfication/
│
├── src/                          # 📦 核心源码包 / Core source package
│   ├── data/                     # 数据处理模块 / Data processing
│   │   ├── dataloader.py         # 数据加载器类 / DataLoader classes
│   │   └── preprocessing.py      # 预处理函数 / Preprocessing functions
│   │
│   ├── models/                   # 模型定义模块 / Model definitions
│   │   ├── base_model.py         # 模型基类 / Base model class
│   │   ├── cnn_models.py         # CNN架构 / CNN architectures
│   │   └── transfer_learning.py  # 迁移学习 / Transfer learning
│   │
│   ├── utils/                    # 工具函数模块 / Utility functions
│   │   ├── common.py             # 通用工具 / Common utilities
│   │   ├── logger.py             # 日志系统 / Logging system
│   │   └── checkpoint.py         # 检查点管理 / Checkpoint management
│   │
│   ├── visualization/            # 可视化模块 / Visualization
│   │   └── plots.py              # 绘图函数 / Plotting functions
│   │
│   ├── train.py                  # 训练逻辑 / Training logic
│   ├── evaluate.py               # 评估逻辑 / Evaluation logic
│   └── predict.py                # 推理逻辑 / Inference logic
│
├── configs/                      # ⚙️ 配置文件 / Configuration files
│   ├── config.yaml               # 主配置 / Main config
│   └── base_config.yaml          # 基础配置 / Base config
│
├── data/                         # 💾 数据存储 / Data storage
│   ├── raw/                      # 原始数据 / Raw data
│   ├── processed/                # 处理后数据 / Processed data
│   ├── train/                    # 训练集 / Training set
│   ├── val/                      # 验证集 / Validation set
│   └── test/                     # 测试集 / Test set
│
├── models/                       # 🤖 模型存储 / Model storage
│   ├── checkpoints/              # 训练检查点 / Training checkpoints
│   └── saved_models/             # 最终模型 / Final models
│
├── scripts/                      # 🚀 可执行脚本 / Executable scripts
│   ├── train.py                  # 训练入口 / Training entry
│   ├── evaluate.py               # 评估入口 / Evaluation entry
│   ├── predict.py                # 预测入口 / Prediction entry
│   └── prepare_data.py           # 数据准备 / Data preparation
│
├── tests/                        # 🧪 单元测试 / Unit tests
│   ├── test_data.py              # 数据测试 / Data tests
│   ├── test_models.py            # 模型测试 / Model tests
│   ├── test_train.py             # 训练测试 / Training tests
│   └── test_utils.py             # 工具测试 / Utility tests
│
├── notebooks/                    # 📓 Jupyter笔记本 / Jupyter notebooks
│   └── data_exploration.md       # 数据探索 / Data exploration
│
└── docs/                         # 📚 文档目录 / Documentation
    └── README.md                 # 文档说明 / Documentation guide
```

---

## 使用流程 / Usage Workflow

### 完整工作流程 / Complete Workflow

```
1. 数据准备阶段 / Data Preparation Phase
   ↓
   准备原始数据 → 运行 prepare_data.py → 生成 train/val/test
   Prepare raw data → Run prepare_data.py → Generate train/val/test
   
2. 配置阶段 / Configuration Phase
   ↓
   编辑 config.yaml → 设置模型和超参数 → 选择数据增强策略
   Edit config.yaml → Set model and hyperparameters → Choose augmentation
   
3. 训练阶段 / Training Phase
   ↓
   运行 train.py → 监控训练日志 → 自动保存检查点
   Run train.py → Monitor training logs → Auto-save checkpoints
   
4. 评估阶段 / Evaluation Phase
   ↓
   运行 evaluate.py → 生成评估指标 → 查看混淆矩阵
   Run evaluate.py → Generate metrics → View confusion matrix
   
5. 推理阶段 / Inference Phase
   ↓
   加载最佳模型 → 运行 predict.py → 获得预测结果
   Load best model → Run predict.py → Get predictions
```

### 详细步骤说明 / Detailed Step Instructions

#### 步骤 1：数据准备 / Step 1: Data Preparation

```python
# 使用 prepare_data.py 脚本
# Use prepare_data.py script

python scripts/prepare_data.py \
    --data_dir /path/to/raw/data \
    --output_dir ./data \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

**功能说明 / Functionality:**
- 自动读取原始数据 / Automatically read raw data
- 按比例划分数据集 / Split dataset by ratio
- 生成标准目录结构 / Generate standard directory structure
- 验证数据完整性 / Validate data integrity

#### 步骤 2：配置训练参数 / Step 2: Configure Training

编辑 `configs/config.yaml`:

```yaml
# 训练配置 / Training Configuration
training:
  batch_size: 32              # 根据GPU内存调整 / Adjust based on GPU memory
  num_epochs: 100             # 训练轮数 / Number of epochs
  learning_rate: 0.001        # 初始学习率 / Initial learning rate
  optimizer: 'adam'           # 优化器选择 / Optimizer choice
  weight_decay: 0.0001        # L2正则化 / L2 regularization
  
  # 学习率调度器 / Learning rate scheduler
  scheduler:
    type: 'step'              # 可选: step, cosine, plateau
    step_size: 30             # 每30个epoch降低学习率
    gamma: 0.1                # 学习率衰减因子

# 模型配置 / Model Configuration
model:
  name: 'resnet50'            # 模型选择 / Model selection
                              # 可选: resnet18, resnet50, vgg16, simple_cnn
  num_classes: 10             # 分类数量（根据数据集调整）
  pretrained: true            # 使用预训练权重
  dropout: 0.5                # Dropout率

# 数据配置 / Data Configuration
data:
  img_size: [224, 224]        # 图像大小 / Image size
  num_workers: 4              # 数据加载线程数
  
  # 数据增强 / Data Augmentation
  augmentation:
    horizontal_flip: true     # 水平翻转
    vertical_flip: false      # 垂直翻转
    rotation: 15              # 旋转角度
    zoom_range: 0.2           # 缩放范围
    brightness_range: [0.8, 1.2]  # 亮度范围
```

#### 步骤 3：开始训练 / Step 3: Start Training

```bash
# 基础训练命令 / Basic training command
python scripts/train.py

# 带参数的训练 / Training with arguments
python scripts/train.py \
    --config configs/config.yaml \
    --gpu 0 \
    --resume models/checkpoints/checkpoint_epoch_10.pth
```

**训练过程会自动：/ Training will automatically:**
- ✅ 加载和预处理数据 / Load and preprocess data
- ✅ 初始化模型和优化器 / Initialize model and optimizer
- ✅ 每个epoch显示进度 / Show progress each epoch
- ✅ 保存最佳模型 / Save best model
- ✅ 记录TensorBoard日志 / Log to TensorBoard
- ✅ 定期保存检查点 / Periodically save checkpoints

**监控训练：/ Monitor Training:**

```bash
# 启动 TensorBoard / Launch TensorBoard
tensorboard --logdir=./logs

# 在浏览器打开 / Open in browser
# http://localhost:6006
```

#### 步骤 4：模型评估 / Step 4: Model Evaluation

```bash
# 评估最佳模型 / Evaluate best model
python scripts/evaluate.py \
    --model_path models/saved_models/best_model.pth \
    --data_dir data/test \
    --output_dir results/

# 生成详细报告 / Generate detailed report
python scripts/evaluate.py \
    --model_path models/saved_models/best_model.pth \
    --data_dir data/test \
    --save_confusion_matrix \
    --save_predictions
```

**评估输出：/ Evaluation Output:**
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1-Score)
- 混淆矩阵 (Confusion Matrix)
- 分类报告 (Classification Report)

#### 步骤 5：模型推理 / Step 5: Model Inference

```bash
# 单张图片预测 / Single image prediction
python scripts/predict.py \
    --image_path test_image.jpg \
    --model_path models/saved_models/best_model.pth

# 批量预测 / Batch prediction
python scripts/predict.py \
    --image_dir test_images/ \
    --model_path models/saved_models/best_model.pth \
    --output_csv predictions.csv
```

---

## 配置说明 / Configuration Guide

### 完整配置参数说明 / Complete Configuration Parameters

#### 训练参数 / Training Parameters

| 参数 Parameter | 说明 Description | 默认值 Default | 建议值 Recommended |
|---------------|-----------------|---------------|-------------------|
| `batch_size` | 批次大小 | 32 | 根据GPU调整 (16-64) |
| `num_epochs` | 训练轮数 | 100 | 50-200 |
| `learning_rate` | 学习率 | 0.001 | 0.0001-0.01 |
| `optimizer` | 优化器 | adam | adam, sgd, rmsprop |
| `weight_decay` | 权重衰减 | 0.0001 | 0-0.001 |

#### 模型参数 / Model Parameters

| 参数 Parameter | 说明 Description | 可选值 Options |
|---------------|-----------------|---------------|
| `name` | 模型架构 | resnet18, resnet50, vgg16, simple_cnn |
| `num_classes` | 分类数量 | 根据数据集 (1-1000+) |
| `pretrained` | 预训练权重 | true, false |
| `dropout` | Dropout率 | 0.0-0.8 |

#### 数据增强参数 / Data Augmentation Parameters

```yaml
augmentation:
  horizontal_flip: true          # 水平翻转（常用）
  vertical_flip: false           # 垂直翻转（根据任务）
  rotation: 15                   # 旋转角度 (0-180)
  zoom_range: 0.2                # 缩放范围 (0-1)
  brightness_range: [0.8, 1.2]   # 亮度调整
  contrast_range: [0.8, 1.2]     # 对比度调整
  saturation_range: [0.8, 1.2]   # 饱和度调整
```

---

## 模块详解 / Module Details

### 1. 数据模块 (src/data/) / Data Module

#### ImageDataLoader 类

```python
from src.data import ImageDataLoader

# 初始化数据加载器 / Initialize data loader
loader = ImageDataLoader(
    data_dir='./data',
    batch_size=32,
    img_size=(224, 224)
)

# 获取数据 / Get data
train_loader, val_loader, test_loader = loader.load_data()

# 获取类别名称 / Get class names
class_names = loader.get_class_names()
```

**功能：/ Features:**
- 自动加载训练、验证、测试集
- 支持多种图像格式
- 内置数据增强
- 批量处理

#### DataAugmentation 类

```python
from src.data import DataAugmentation

# 配置增强策略 / Configure augmentation
aug_config = {
    'horizontal_flip': True,
    'rotation': 15,
    'zoom_range': 0.2
}

augmenter = DataAugmentation(aug_config)
augmented_image = augmenter.apply_augmentation(image)
```

### 2. 模型模块 (src/models/) / Model Module

#### 使用预定义模型 / Using Predefined Models

```python
from src.models import ResNetModel, VGGModel, SimpleCNN

# ResNet模型 / ResNet model
model = ResNetModel(
    num_classes=10,
    version='resnet50',
    pretrained=True
)
resnet = model.build()

# VGG模型 / VGG model
model = VGGModel(
    num_classes=10,
    version='vgg16',
    pretrained=True
)
vgg = model.build()

# 简单CNN / Simple CNN
model = SimpleCNN(
    num_classes=10,
    input_shape=(224, 224, 3)
)
cnn = model.build()
```

#### 迁移学习 / Transfer Learning

```python
from src.models import TransferLearning

# 初始化迁移学习 / Initialize transfer learning
tl = TransferLearning(
    base_model=pretrained_model,
    num_classes=10
)

# 冻结基础层 / Freeze base layers
tl.freeze_base_layers(num_layers=100)

# 添加分类头 / Add classification head
model = tl.add_classification_head(num_classes=10)

# 解冻进行微调 / Unfreeze for fine-tuning
tl.unfreeze_base_layers(num_layers=50)
```

### 3. 训练模块 (src/train.py) / Training Module

```python
from src.train import Trainer

# 初始化训练器 / Initialize trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config
)

# 开始训练 / Start training
history = trainer.train(num_epochs=100)

# 保存检查点 / Save checkpoint
trainer.save_checkpoint(epoch=50, metrics=metrics)

# 恢复训练 / Resume training
trainer.load_checkpoint('path/to/checkpoint.pth')
```

### 4. 评估模块 (src/evaluate.py) / Evaluation Module

```python
from src.evaluate import Evaluator

# 初始化评估器 / Initialize evaluator
evaluator = Evaluator(
    model=trained_model,
    test_loader=test_loader
)

# 执行评估 / Perform evaluation
metrics = evaluator.evaluate()

# 生成混淆矩阵 / Generate confusion matrix
cm = evaluator.generate_confusion_matrix(predictions, targets)

# 生成分类报告 / Generate classification report
report = evaluator.generate_classification_report(predictions, targets)
```

### 5. 预测模块 (src/predict.py) / Prediction Module

```python
from src.predict import Predictor

# 初始化预测器 / Initialize predictor
predictor = Predictor(
    model=model,
    model_path='models/saved_models/best_model.pth'
)

# 单张图片预测 / Single image prediction
class_label, confidence = predictor.predict('image.jpg')

# 批量预测 / Batch prediction
predictions = predictor.predict_batch(['img1.jpg', 'img2.jpg'])

# 获取概率分布 / Get probability distribution
proba = predictor.predict_proba('image.jpg')
```

### 6. 可视化模块 (src/visualization/) / Visualization Module

```python
from src.visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_sample_predictions
)

# 绘制训练历史 / Plot training history
plot_training_history(history, save_path='training_curves.png')

# 绘制混淆矩阵 / Plot confusion matrix
plot_confusion_matrix(
    confusion_matrix,
    class_names,
    save_path='confusion_matrix.png'
)

# 绘制样本预测 / Plot sample predictions
plot_sample_predictions(
    images,
    predictions,
    true_labels,
    save_path='predictions.png'
)
```

---

## 扩展开发 / Extension Development

### 添加自定义模型 / Adding Custom Models

1. 在 `src/models/` 创建新文件，例如 `custom_model.py`
2. 继承 `BaseModel` 类
3. 实现 `build()` 方法

```python
from src.models.base_model import BaseModel

class CustomModel(BaseModel):
    def __init__(self, num_classes, input_shape=(224, 224, 3)):
        super().__init__(num_classes, input_shape)
        # 自定义初始化
    
    def build(self):
        # 实现自定义架构
        # Build custom architecture
        pass
```

### 添加自定义数据增强 / Adding Custom Augmentation

在 `src/data/preprocessing.py` 中添加新函数：

```python
def custom_augmentation(image, params):
    """
    自定义数据增强函数
    Custom augmentation function
    """
    # 实现增强逻辑
    return augmented_image
```

### 添加自定义评估指标 / Adding Custom Metrics

在 `src/evaluate.py` 的 `Evaluator` 类中添加方法：

```python
def calculate_custom_metric(self, predictions, targets):
    """
    计算自定义指标
    Calculate custom metric
    """
    # 实现指标计算
    return metric_value
```

---

## 常见问题 / FAQ

### Q1: 如何选择合适的模型？ / How to choose the right model?

**A:** 
- **简单任务 + 小数据集**: SimpleCNN
- **一般任务**: ResNet18/34
- **复杂任务 + 大数据集**: ResNet50/101
- **需要预训练**: 使用 `pretrained=True`

### Q2: 训练时显存不足怎么办？ / What if out of memory during training?

**A:** 
```yaml
# 减小批次大小 / Reduce batch size
batch_size: 16  # 从32改为16

# 减小图像尺寸 / Reduce image size
img_size: [128, 128]  # 从224改为128

# 使用混合精度训练 / Use mixed precision
mixed_precision: true
```

### Q3: 如何恢复中断的训练？ / How to resume interrupted training?

**A:**
```bash
python scripts/train.py --resume models/checkpoints/checkpoint_epoch_50.pth
```

### Q4: 如何调整学习率？ / How to adjust learning rate?

**A:** 在 `config.yaml` 中配置调度器：

```yaml
scheduler:
  type: 'step'        # 阶梯下降
  # type: 'cosine'    # 余弦退火
  # type: 'plateau'   # 性能平台
```

### Q5: 如何导出模型用于部署？ / How to export model for deployment?

**A:**
```python
# 保存完整模型 / Save complete model
torch.save(model, 'model_complete.pth')

# 保存为ONNX格式 / Save as ONNX
torch.onnx.export(model, dummy_input, 'model.onnx')
```

### Q6: 如何处理类别不平衡？ / How to handle class imbalance?

**A:**
- 使用加权损失函数 / Use weighted loss
- 过采样少数类 / Oversample minority class
- 欠采样多数类 / Undersample majority class
- 使用数据增强 / Use data augmentation

### Q7: 训练速度太慢怎么办？ / Training too slow?

**A:**
- 增加 `num_workers` 参数
- 使用更小的模型
- 使用混合精度训练
- 使用多GPU训练

### Q8: 如何进行模型集成？ / How to ensemble models?

**A:**
```python
# 训练多个模型
model1 = train_model(config1)
model2 = train_model(config2)
model3 = train_model(config3)

# 集成预测
pred1 = model1.predict(image)
pred2 = model2.predict(image)
pred3 = model3.predict(image)

# 投票或平均
final_pred = (pred1 + pred2 + pred3) / 3
```

---

## 最佳实践 / Best Practices

### 1. 数据准备 / Data Preparation
- ✅ 确保数据质量，删除损坏图片
- ✅ 数据均衡性检查
- ✅ 合理划分训练/验证/测试集
- ✅ 使用数据增强提高泛化能力

### 2. 模型选择 / Model Selection
- ✅ 从简单模型开始
- ✅ 使用预训练权重
- ✅ 根据任务复杂度选择模型大小

### 3. 训练策略 / Training Strategy
- ✅ 设置合理的学习率
- ✅ 使用学习率调度器
- ✅ 早停策略防止过拟合
- ✅ 定期保存检查点

### 4. 调试技巧 / Debugging Tips
- ✅ 先在小数据集上验证代码
- ✅ 监控训练和验证损失曲线
- ✅ 使用TensorBoard可视化
- ✅ 记录所有实验参数

### 5. 性能优化 / Performance Optimization
- ✅ 使用GPU加速
- ✅ 批量大小根据GPU内存调整
- ✅ 多进程数据加载
- ✅ 混合精度训练

---

## 项目示例 / Project Examples

### 示例 1：猫狗分类 / Example 1: Cat vs Dog Classification

```bash
# 1. 准备数据
python scripts/prepare_data.py --data_dir raw_data/cats_dogs

# 2. 配置（2个类别）
# Edit config.yaml: num_classes: 2

# 3. 训练
python scripts/train.py

# 4. 评估
python scripts/evaluate.py --model_path models/saved_models/best_model.pth

# 5. 预测新图片
python scripts/predict.py --image_path test_cat.jpg
```

### 示例 2：CIFAR-10分类 / Example 2: CIFAR-10 Classification

```yaml
# config.yaml
model:
  name: 'resnet18'
  num_classes: 10
  pretrained: false

data:
  img_size: [32, 32]
  
training:
  batch_size: 128
  num_epochs: 200
  learning_rate: 0.1
```

### 示例 3：迁移学习 / Example 3: Transfer Learning

```bash
# 使用预训练ResNet50
# config.yaml: model.pretrained: true

# 先冻结训练
python scripts/train.py --freeze_backbone --epochs 10

# 再微调
python scripts/train.py --unfreeze_backbone --epochs 50
```

---

## 技术支持 / Technical Support

### 资源链接 / Resources
- 📖 PyTorch文档: https://pytorch.org/docs/
- 📖 TensorFlow文档: https://www.tensorflow.org/
- 📖 项目GitHub: https://github.com/jurky123/image_classfication

### 常见错误解决 / Common Error Solutions

1. **CUDA out of memory**
   - 减小batch_size
   - 减小图像尺寸
   - 使用梯度累积

2. **模型不收敛**
   - 检查学习率（可能太大或太小）
   - 检查数据标签是否正确
   - 尝试不同的优化器

3. **过拟合**
   - 增加数据增强
   - 增加Dropout
   - 使用正则化
   - 减小模型复杂度

4. **欠拟合**
   - 增加模型复杂度
   - 增加训练轮数
   - 调整学习率
   - 检查数据质量

---

## 更新日志 / Changelog

### Version 0.1.0 (2026-02-10)
- ✅ 初始项目框架
- ✅ 基础模块实现
- ✅ 配置系统
- ✅ 文档完善

### 计划功能 / Planned Features
- [ ] 混合精度训练支持
- [ ] 分布式训练支持
- [ ] 模型压缩和量化
- [ ] Web界面演示
- [ ] 更多预训练模型

---

## 贡献指南 / Contributing

欢迎贡献！请遵循以下步骤：
Welcome to contribute! Please follow these steps:

1. Fork 项目 / Fork the project
2. 创建特性分支 / Create feature branch
3. 提交更改 / Commit changes
4. 推送到分支 / Push to branch
5. 创建 Pull Request / Create Pull Request

---

## 许可证 / License

MIT License - 详见 LICENSE 文件

---

**最后更新 / Last Updated**: 2026-02-10

**作者 / Author**: Image Classification Project Team

**联系方式 / Contact**: GitHub Issues
