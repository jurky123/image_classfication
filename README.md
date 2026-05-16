# ConvNeXt Image Classification

ConvNeXt-based transfer learning for Oxford Flowers 102 classification (102 flower species). Best validation accuracy: **91.08%**.

## Setup

```bash
pip install -r requirements.txt
```

## Data Preparation

Download the [Oxford Flowers 102 dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) and place the `.mat` files and `jpg/` folder under `data/raw/`.

```bash
python prepare_data.py --raw-dir data/raw --data-dir data
```

This organizes images into `data/{train,val,test}/class_NNN/` in ImageFolder format.

## Training

```bash
# Default training
python train.py

# Custom settings
python train.py \
  --variant tiny \
  --batch-size 64 \
  --epochs 50 \
  --lr 1.4e-4 \
  --augment strong \
  --unfreeze-at 15 \
  --device cuda

# Resume from best checkpoint
python train.py --resume-best
```

## Evaluation

Interactive evaluation: random test images with predictions displayed.

```bash
python evaluate.py --model-path models/saved_models/best_model.pth
```

Keys: N/Space/Right = next image, Q/Esc = quit.

## Inference

```bash
# Single image
python predict.py --model models/saved_models/best_model.pth --image photo.jpg

# Directory batch
python predict.py --model models/saved_models/best_model.pth --dir photos/ --top-k 3

# JSON output
python predict.py --model models/saved_models/best_model.pth --image photo.jpg --json
```

## Project Structure

```
image_classfication/
├── train.py              # Training script
├── evaluate.py           # Interactive evaluation
├── predict.py            # Inference script
├── prepare_data.py       # Oxford Flowers 102 data preparation
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── trainer.py        # TrainerConfig + ConvNeXtTrainer
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py # Base class for models
│   │   └── convnext.py   # ConvNeXtClassifier
│   └── data/
│       ├── __init__.py
│       └── transforms.py # Data transforms and augmentations
├── data/                 # Dataset (runtime)
├── logs/                 # Training history (runtime)
└── models/               # Checkpoints and saved models (runtime)
```

## Model Variants

| Variant | Params | ImageNet Acc | Description |
|---------|--------|-------------|-------------|
| tiny    | 28.6M  | 82.1%       | Lightweight, fast |
| small   | 50.2M  | 83.6%       | Balanced |
| base    | 88.6M  | 84.4%       | High performance |

## License

MIT
