# 🚀 ConvNeXt 项目 - 快速参考卡片

## ⚡ 30秒快速开始

```bash
# 1. 验证环境
python quick_verify.py

# 2. 开始训练 (推荐方式)
python train.py

# 或直接运行
python scripts/train_convnext.py --variant tiny --epochs 30
```

---

## 📁 关键文件位置

| 用途 | 文件 | 说明 |
|------|------|------|
| **开始训练** | `train.py` | 交互式训练启动器 ⭐ |
| **快速验证** | `quick_verify.py` | 一键检查 |
| **训练脚本** | `scripts/train_convnext.py` | 底层训练脚本 |
| **模型代码** | `src/models/convnext_model.py` | ConvNeXt实现 |
| **数据处理** | `src/data/preprocessing.py` | 数据增强 |
| **完整指南** | `PROJECT_SUMMARY.md` | 详细说明文档 |
| **训练指南** | `TRAINING_GUIDE.md` | 训练详解 |

---

## 🎯 常用命令

### 基础训练
```bash
python scripts/train_convnext.py
```

### 自定义参数
```bash
python scripts/train_convnext.py \
  --variant small \
  --batch_size 16 \
  --epochs 50 \
  --lr 5e-5
```

### CPU 训练
```bash
python scripts/train_convnext.py --device cpu
```

### 使用启动器
```bash
python train.py  # 交互式选择参数
```

---

## 📊 模型对比

| 模型 | 参数量 | 速度 | 精度 |
|------|--------|------|------|
| **Tiny** | 28.6M | ⚡ 快 | ⭐⭐⭐⭐ |
| **Small** | 50.2M | ⚠️ 中 | ⭐⭐⭐⭐⭐ |
| **Base** | 88.6M | 🐌 慢 | ⭐⭐⭐⭐⭐⭐ |

---

## ⚙️ 常见参数

```
--variant      模型大小 (tiny/small/base)    [默认: tiny]
--batch_size   批次大小                     [默认: 32]
--epochs       训练轮数                     [默认: 30]
--lr           学习率                       [默认: 1e-4]
--device       计算设备 (cuda/cpu)          [默认: cuda]
--num_workers  数据加载进程                 [默认: 4]
--unfreeze_ep  第几个epoch解冻骨干          [默认: 10]
```

---

## 📈 预期时间和精度

### 训练时间 (30 epochs, 批大小32)
- **GPU RTX 3090**: 30-40 分钟
- **GPU RTX 4080**: 20-25 分钟
- **CPU 单核**: 3-4 小时

### 预期精度
```
冻结阶段 (Epochs 1-10)  : ~85% 验证精度
微调阶段 (Epochs 10-30) : ~91-92% 验证精度
```

---

## 🔍 监控训练

### 查看实时日志
```bash
# Linux/Mac
tail -f models/logs/training.log

# Windows PowerShell
Get-Content models/logs/training.log -Wait
```

### 监控GPU使用
```bash
nvidia-smi -l 1  # 每秒刷新一次
```

### 分析训练结果
```bash
# JSON 格式的详细历史
cat models/logs/training_history_*.json

# 或用Python分析
python
>>> import json
>>> with open('models/logs/training_history_*.json') as f:
...     h = json.load(f)
...     print(f"最佳精度: {max(h['val_acc']):.2%}")
```

---

## 🛠️ 故障排除

| 问题 | 解决方案 |
|------|---------|
| CUDA 内存溢出 | 减小 `--batch_size` 或用 `--variant tiny` |
| 训练缓慢 | 增加 `--num_workers` 或使用 `--batch_size 64` |
| 损失不下降 | 降低 `--lr 5e-5` 或增加 `--epochs 50` |
| 导入错误 | 运行 `python quick_verify.py` |
| 数据缺失 | 运行 `python scripts/prepare_data.py` |

---

## 💾 输出文件

训练完成后会生成：

```
models/
├── saved_models/
│   └── best_model.pth          最佳模型 ⭐
├── checkpoints/
│   ├── epoch_0.pth
│   ├── epoch_5.pth
│   └── ...
└── logs/
    ├── training.log            实时日志
    └── training_history_*.json 详细数据
```

---

## 🎓 使用训练好的模型

```python
import torch
from src.models.convnext_model import ConvNeXtClassifier
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image

# 加载模型
model = ConvNeXtClassifier(num_classes=102, variant='tiny', pretrained=False)
ckpt = torch.load('models/saved_models/best_model.pth')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# 加载图片
image = Image.open('flower.jpg')
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
x = transform(image).unsqueeze(0)

# 推理
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    class_idx = probs.argmax(dim=1).item()
    confidence = probs.max().item()

print(f"类别: {class_idx}, 置信度: {confidence:.1%}")
```

---

## 📚 更多资源

- 详细指南: `PROJECT_SUMMARY.md`
- 训练教程: `TRAINING_GUIDE.md`
- 模型指南: `docs/CONVNEXT_GUIDE.md`
- 代码示例: `examples/quick_start.py`
- 完成报告: `COMPLETION_REPORT.md`

---

## ✨ 项目特点

✅ **开箱即用** - 数据已准备，模型已实现
✅ **文档完善** - 详细的指南和示例
✅ **灵活配置** - 支持多种参数自定义
✅ **高效训练** - 两阶段迁移学习策略
✅ **实时监控** - 自动日志和检查点保存

---

## 🚀 立即开始

```bash
# 最简单的方式
python train.py

# 或直接运行
python scripts/train_convnext.py
```

**预计训练时间**: 30-40 分钟
**预期最终精度**: 91-92% ⭐

祝您训练顺利！🎉
