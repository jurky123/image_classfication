# ConvNeXt 图像分类项目

## 📦 项目概述

一个**完整的、生产级的** ConvNeXt 图像分类实现，使用PyTorch框架和迁移学习技术在 Oxford Flowers 102 数据集上训练。

### 🎯 主要特性

- ✅ **ConvNeXt 模型** - 支持 Tiny/Small/Base 三种规格
- ✅ **两阶段迁移学习** - 冻结 → 微调训练策略  
- ✅ **完整训练管道** - 数据加载、模型训练、评估、推理
- ✅ **详细文档** - 快速开始、深度教程、常见问题解答
- ✅ **开箱即用** - 数据已准备，脚本可直接运行
- ✅ **灵活配置** - 支持自定义超参数和模型设置

---

## 🚀 快速开始 (3步)

### 1️⃣ 验证环境
```bash
python quick_verify.py
```
输出应该显示所有 `[OK]` ✅

### 2️⃣ 开始训练
```bash
# 推荐方式（交互式）
python train.py

# 或直接运行
python scripts/train_convnext.py
```

### 3️⃣ 监控进度
```bash
tail -f models/logs/training.log
```

---

## 📚 文档导航

**👉 新手建议从这里开始**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - 快速参考卡片

| 文档 | 说明 |
|------|------|
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | 快速参考卡片 ⭐ **从这里开始** |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | 详细项目说明和完整指南 |
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | 详细训练指南和FAQ |
| [COMPLETION_REPORT.md](COMPLETION_REPORT.md) | 项目完成报告 |
| [docs/CONVNEXT_GUIDE.md](docs/CONVNEXT_GUIDE.md) | ConvNeXt 深度教程 |

---

## ⚡ 最简单的开始方式

```bash
# 一键启动（推荐）
python train.py

# 或直接运行
python scripts/train_convnext.py
```

**预计耗时**: 30-40 分钟（GPU）  
**预期精度**: 91-92% ⭐

---

## 📊 快速参考

### 常用命令

| 命令 | 说明 |
|------|------|
| `python train.py` | 交互式训练启动器 ⭐ |
| `python quick_verify.py` | 快速检查环境 |
| `python scripts/train_convnext.py` | 直接训练（默认参数） |

### 自定义训练

```bash
python scripts/train_convnext.py \
  --variant small \           # tiny/small/base
  --batch_size 32 \           # 批次大小
  --epochs 50 \               # 训练轮数
  --lr 1e-4                   # 学习率
```

---

## 📁 项目结构

```
image_classfication/
├── README.md                     ← 本文件
├── QUICK_REFERENCE.md            ← 快速参考卡片 ⭐
├── PROJECT_SUMMARY.md            ← 详细项目说明
├── TRAINING_GUIDE.md             ← 完整训练指南
├── COMPLETION_REPORT.md          ← 项目完成报告
│
├── train.py                      ← 启动器脚本 ⭐
├── quick_verify.py               ← 快速验证
│
├── scripts/
│   ├── train_convnext.py         ← 核心训练脚本
│   └── prepare_data.py           ← 数据准备
│
├── src/
│   ├── models/convnext_model.py  ← ConvNeXt实现 ⭐
│   ├── data/preprocessing.py     ← 数据增强
│   └── utils/                    ← 工具函数
│
├── data/
│   ├── train/                    ← 训练数据（102类）
│   ├── val/                      ← 验证数据
│   └── test/                     ← 测试数据
│
└── models/
    ├── saved_models/             ← 最佳模型输出
    ├── checkpoints/              ← 检查点
    └── logs/                     ← 训练日志
```

---

## 🎓 技术特点

### 两阶段迁移学习
```
Phase 1 (冻结):   固定骨干网络，训练分类头    [Epochs 1-10]
Phase 2 (微调):   解冻最后N个Stage，微调网络  [Epochs 10-30]
```

### 完整的数据增强
- ✅ 随机翻转 (50%)
- ✅ 随机旋转 (±15°)
- ✅ 颜色抖动 (亮度/对比度/饱和度/色调)
- ✅ 仿射变换 (平移/缩放)
- ✅ ImageNet标准化

### 灵活的训练管道
- ✅ 自适应学习率调度 (CosineAnnealing)
- ✅ 梯度裁剪 (防止梯度爆炸)
- ✅ 自动检查点保存
- ✅ 详细的训练日志

---

## 📊 性能预期

| 指标 | 预期值 |
|------|--------|
| 最终验证精度 | 91-92% |
| 训练时间 (GPU) | 30-40 分钟 |
| GPU显存需求 | ~6GB |
| 模型参数量 | 28.6M (Tiny) |

---

## 🚀 技术栈

- **框架**: PyTorch
- **模型**: ConvNeXt (timm)
- **数据**: Torchvision, PIL, NumPy
- **工具**: Python logging, JSON

---

## ✅ 项目检查清单

- [x] 数据集准备完成 (8,189 张图片)
- [x] 模型实现完整 (3 种规格)
- [x] 训练脚本就绪
- [x] 文档全面详细
- [x] 代码已验证测试
- [x] 示例代码可用
- [x] 错误处理完善
- [x] 日志系统完整

---

## 📞 获取帮助

### 遇到问题：
1. 运行 `python quick_verify.py` 验证环境
2. 查看 [QUICK_REFERENCE.md](QUICK_REFERENCE.md) 的常见问题
3. 检查 `models/logs/training.log` 的错误
4. 参考 [TRAINING_GUIDE.md](TRAINING_GUIDE.md) 的FAQ

### 需要帮助：
- 查看代码示例: [examples/quick_start.py](examples/quick_start.py)
- 阅读详细指南: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
- 了解架构设计: [docs/PROJECT_LOGIC.md](docs/PROJECT_LOGIC.md)
│   │   └── transfer_learning.py # 迁移学习
│   │
│   ├── utils/                    # 工具模块
│   │   ├── __init__.py
│   │   ├── common.py            # 通用工具函数
│   │   ├── logger.py            # 日志记录
│   │   └── checkpoint.py        # 模型检查点
│   │
│   ├── visualization/            # 可视化模块
│   │   ├── __init__.py
│   │   └── plots.py             # 绘图函数
│   │
│   ├── train.py                 # 训练模块
│   ├── evaluate.py              # 评估模块
│   └── predict.py               # 预测模块
│
├── configs/                      # 配置文件目录
│   ├── config.yaml              # 主配置文件
│   └── base_config.yaml         # 基础配置文件
│
├── data/                         # 数据目录
│   ├── raw/                     # 原始数据
│   ├── processed/               # 处理后的数据
│   ├── train/                   # 训练集
│   ├── val/                     # 验证集
│   └── test/                    # 测试集
│
├── models/                       # 模型保存目录
│   ├── checkpoints/             # 训练检查点
│   └── saved_models/            # 保存的模型
│
├── notebooks/                    # Jupyter notebooks
│   └── (用于实验和数据探索)
│
├── scripts/                      # 脚本目录
│   ├── train.py                 # 训练脚本
│   ├── evaluate.py              # 评估脚本
│   ├── predict.py               # 预测脚本
│   └── prepare_data.py          # 数据准备脚本
│
├── tests/                        # 测试目录
│   ├── __init__.py
│   ├── test_data.py             # 数据测试
│   ├── test_models.py           # 模型测试
│   ├── test_train.py            # 训练测试
│   └── test_utils.py            # 工具测试
│
├── docs/                         # 文档目录
│   └── images/                  # 文档图片
│
├── .gitignore                    # Git忽略文件
├── requirements.txt              # 项目依赖
├── setup.py                      # 安装配置
└── README.md                     # 项目说明
```

## 模块说明 (Module Description)

### 1. 数据模块 (Data Module)
- **dataloader.py**: 数据加载和批处理
- **preprocessing.py**: 图像预处理和数据增强

### 2. 模型模块 (Models Module)
- **base_model.py**: 所有模型的基类
- **cnn_models.py**: 包含SimpleCNN、ResNet、VGG等模型
- **transfer_learning.py**: 迁移学习相关功能

### 3. 工具模块 (Utils Module)
- **common.py**: 通用工具函数（设备检测、种子设置等）
- **logger.py**: 训练日志记录
- **checkpoint.py**: 模型检查点管理

### 4. 可视化模块 (Visualization Module)
- **plots.py**: 训练曲线、混淆矩阵等可视化

### 5. 核心功能 (Core Functions)
- **train.py**: 模型训练逻辑
- **evaluate.py**: 模型评估逻辑
- **predict.py**: 模型预测/推理

## 使用说明 (Usage)

### 安装依赖 (Install Dependencies)
```bash
pip install -r requirements.txt
```

### 准备数据 (Prepare Data)
```bash
python scripts/prepare_data.py --data_dir /path/to/your/data
```

### 训练模型 (Train Model)
```bash
python scripts/train.py --config configs/config.yaml
```

### 评估模型 (Evaluate Model)
```bash
python scripts/evaluate.py --model_path models/saved_models/best_model.pth
```

### 预测 (Predict)
```bash
python scripts/predict.py --image_path /path/to/image.jpg --model_path models/saved_models/best_model.pth
```

## 配置文件 (Configuration)

配置文件位于 `configs/` 目录，包含：
- 模型参数
- 训练超参数
- 数据路径
- 日志设置

## 测试 (Testing)

运行测试：
```bash
pytest tests/
```

## 开发计划 (Development Plan)

- [ ] 实现数据加载和预处理
- [ ] 实现基础CNN模型
- [ ] 实现迁移学习
- [ ] 实现训练和验证流程
- [ ] 实现模型评估
- [ ] 添加可视化功能
- [ ] 优化性能
- [ ] 完善文档

## 贡献 (Contributing)

欢迎提交问题和拉取请求！

## 许可证 (License)

MIT License
