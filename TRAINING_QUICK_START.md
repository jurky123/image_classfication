# 🚀 ConvNeXt 迁移学习 - 3 分钟快速开始

## ⚡ 30 秒开始训练

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据（已完成）
python scripts/prepare_data.py

# 3. 开始训练！
python scripts/train_convnext.py
```

**就这么简单！** 模型会自动开始训练。

---

## 📋 训练过程说明

### 训练流程
```
第 1-10 个 epoch:  冻结骨干网络 → 只训练分类头 → 快速收敛
第 10+ 个 epoch:   解冻最后 2 个阶段 → 联合训练 → 精度提升
```

### 输出文件
```
models/saved_models/best_model.pth    ← 最佳模型（最重要！）
models/checkpoints/                   ← 定期检查点
logs/training.log                     ← 训练日志
logs/training_history_*.json          ← 训练指标
```

---

## 🎯 常用命令

### 基础训练
```bash
# 默认配置（推荐）
python scripts/train_convnext.py

# 使用更强大的 Small 模型
python scripts/train_convnext.py --variant small
```

### 自定义参数
```bash
# 更多 epoch，更大批大小
python scripts/train_convnext.py --epochs 50 --batch-size 64

# 更小的学习率
python scripts/train_convnext.py --lr 5e-5

# 在 epoch 15 解冻
python scripts/train_convnext.py --unfreeze-at 15
```

### 完整示例
```bash
# 高质量训练（推荐用于最终模型）
python scripts/train_convnext.py \
    --variant small \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4
```

---

## 📊 预期结果

| 模型 | 精度 | 时间 |
|------|------|------|
| Tiny | 88-90% | 1-2 小时 |
| Small | 90-93% | 2-3 小时 |
| Base | 92-95% | 4-6 小时 |

---

## 💡 快速提示

### GPU 显存不足？
```bash
# 减小批大小
python scripts/train_convnext.py --batch-size 16
```

### 想要更好的精度？
```bash
# 使用 Small 模型，多训练几个 epoch
python scripts/train_convnext.py --variant small --epochs 50
```

### 想要更快的训练？
```bash
# 使用 Tiny 模型，较少 epoch
python scripts/train_convnext.py --variant tiny --epochs 20
```

---

## 🔍 监控训练

### 查看日志
```bash
# 实时查看
tail -f logs/training.log

# 查看最后 20 行
tail -20 logs/training.log
```

### 查看最佳模型
```bash
# 模型文件大小和时间戳
ls -lh models/saved_models/best_model.pth
```

---

## 🎓 Python 代码示例

### 加载和使用最佳模型
```python
import torch
from src.models.convnext_model import ConvNeXtClassifier

# 创建模型
model = ConvNeXtClassifier(num_classes=102, variant='tiny')

# 加载最佳权重
model.load_weights('models/saved_models/best_model.pth')

# 用于推理
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# 预测
with torch.no_grad():
    output = model(input_image)
    pred_class = output.argmax(dim=1)
```

### 从头开始完整训练
```python
from src.models.convnext_model import ConvNeXtClassifier
import torch.nn as nn
import torch.optim as optim

# 1. 创建模型
model = ConvNeXtClassifier(num_classes=102, variant='tiny', pretrained=True)

# 2. 冻结骨干网络
model.freeze_backbone(freeze=True)

# 3. 设置训练工具
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# 4. 训练...
for epoch in range(30):
    # 训练代码
    pass
    
    # 在 epoch 10 解冻
    if epoch == 10:
        model.unfreeze_backbone(num_stages_to_unfreeze=2)
```

---

## 🎓 学习资源

- 📖 **详细文档**: [docs/CONVNEXT_GUIDE.md](../docs/CONVNEXT_GUIDE.md)
- 🔬 **数据探索**: `jupyter notebook notebooks/data_exploration.ipynb`
- 📊 **项目架构**: [docs/PROJECT_LOGIC.md](../docs/PROJECT_LOGIC.md)

---

## ✅ 检查清单

- [ ] `pip install -r requirements.txt` 安装依赖
- [ ] `python scripts/prepare_data.py` 准备数据
- [ ] `python examples/quick_start.py` 运行示例（可选）
- [ ] `python scripts/train_convnext.py` 开始训练
- [ ] 监控 `logs/training.log` 查看进度
- [ ] 检查 `models/saved_models/best_model.pth` 获取最佳模型

---

## 🎉 成功指标

- ✅ 训练/验证损失逐渐下降
- ✅ 验证准确率 > 85%（第 30 个 epoch）
- ✅ 生成了 `best_model.pth` 文件

**现在就开始训练吧！** 🚀

如有问题，查看 [CONVNEXT_GUIDE.md](../docs/CONVNEXT_GUIDE.md) 了解更多细节。
