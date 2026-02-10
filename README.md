# 图像分类项目 (Image Classification Project)

一个结构清晰、易于扩展的图像分类深度学习项目框架。

## 项目结构 (Project Structure)

```
image_classfication/
│
├── src/                          # 源代码目录
│   ├── __init__.py
│   ├── data/                     # 数据处理模块
│   │   ├── __init__.py
│   │   ├── dataloader.py        # 数据加载器
│   │   └── preprocessing.py     # 数据预处理
│   │
│   ├── models/                   # 模型模块
│   │   ├── __init__.py
│   │   ├── base_model.py        # 基础模型类
│   │   ├── cnn_models.py        # CNN模型架构
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
