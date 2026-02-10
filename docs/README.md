# 项目文档 / Project Documentation

## 文档导航 / Documentation Navigation

### 📘 核心文档 / Core Documentation

1. **[完整使用指南 / Complete Usage Guide](USAGE_GUIDE.md)**
   - 中英双语完整文档
   - 详细的API说明和使用示例
   - 配置参数完整说明
   - FAQ和最佳实践
   - Bilingual comprehensive documentation
   - Detailed API description and examples
   - Complete configuration parameters
   - FAQ and best practices

2. **[项目逻辑详解 / Project Logic Explanation](PROJECT_LOGIC.md)**
   - 中文详细讲解
   - 各模块工作原理
   - 完整使用流程
   - 参数调优指南
   - 常见问题诊断
   - Detailed Chinese explanation
   - Module working principles
   - Complete workflow
   - Parameter tuning guide
   - Problem diagnosis

### 🚀 快速开始 / Quick Start

```bash
# 1. 安装依赖 / Install dependencies
pip install -r requirements.txt

# 2. 准备数据 / Prepare data
python scripts/prepare_data.py --data_dir /path/to/data

# 3. 配置模型 / Configure model
# 编辑 configs/config.yaml

# 4. 开始训练 / Start training
python scripts/train.py --config configs/config.yaml

# 5. 评估模型 / Evaluate model
python scripts/evaluate.py --model_path models/saved_models/best_model.pth

# 6. 预测图片 / Predict image
python scripts/predict.py --image_path test.jpg --model_path models/saved_models/best_model.pth
```

## 项目架构 / Project Architecture

### 核心模块 / Core Modules

- **数据模块 (src/data/)**: 数据加载、预处理、增强
  - Data loading, preprocessing, augmentation
  
- **模型模块 (src/models/)**: CNN架构、迁移学习
  - CNN architectures, transfer learning
  
- **训练模块 (src/train.py)**: 训练循环、优化器
  - Training loop, optimizer
  
- **评估模块 (src/evaluate.py)**: 性能指标、可视化
  - Performance metrics, visualization
  
- **推理模块 (src/predict.py)**: 模型推理、预测
  - Model inference, prediction

### 工具模块 / Utility Modules

- **utils/common.py**: 通用工具函数
- **utils/logger.py**: 日志记录系统
- **utils/checkpoint.py**: 模型检查点管理
- **visualization/plots.py**: 训练可视化

## 扩展开发 / Extension Development

### 添加自定义模型 / Adding Custom Models

```python
from src.models.base_model import BaseModel

class CustomModel(BaseModel):
    def __init__(self, num_classes, input_shape=(224, 224, 3)):
        super().__init__(num_classes, input_shape)
    
    def build(self):
        # 实现自定义架构
        pass
```

### 添加自定义数据增强 / Adding Custom Augmentation

在 `src/data/preprocessing.py` 中添加新函数

### 添加自定义评估指标 / Adding Custom Metrics

在 `src/evaluate.py` 的 `Evaluator` 类中添加方法

## 学习路径 / Learning Path

1. ✅ 阅读 [PROJECT_LOGIC.md](PROJECT_LOGIC.md) 理解项目整体逻辑
2. ✅ 参考 [USAGE_GUIDE.md](USAGE_GUIDE.md) 学习详细使用方法
3. ✅ 在小数据集上运行完整流程
4. ✅ 阅读源码理解实现细节
5. ✅ 根据需求定制和扩展

## 技术栈 / Tech Stack

- **深度学习框架 / DL Framework**: PyTorch
- **数据处理 / Data Processing**: NumPy, Pandas, OpenCV
- **可视化 / Visualization**: Matplotlib, Seaborn, TensorBoard
- **数据增强 / Augmentation**: Albumentations
- **实验跟踪 / Experiment Tracking**: WandB, TensorBoard

## 贡献指南 / Contributing

欢迎贡献！请遵循以下步骤：
1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

请确保代码符合项目结构，并添加相应的测试。

## 支持 / Support

- 📖 详细文档: 查看 [USAGE_GUIDE.md](USAGE_GUIDE.md)
- 🐛 问题报告: GitHub Issues
- 💬 讨论交流: GitHub Discussions

## 许可证 / License

MIT License - 详见根目录 LICENSE 文件
