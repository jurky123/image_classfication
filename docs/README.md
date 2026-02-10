# Documentation

## Project Overview
This is an image classification framework built with PyTorch/TensorFlow.

## Quick Start Guide
1. Install dependencies: `pip install -r requirements.txt`
2. Prepare your dataset in the `data/` directory
3. Configure training in `configs/config.yaml`
4. Run training: `python scripts/train.py`

## Architecture
- **Data Pipeline**: Handles loading, preprocessing, and augmentation
- **Model Zoo**: Various CNN architectures and transfer learning
- **Training Pipeline**: Full training loop with checkpointing
- **Evaluation**: Comprehensive metrics and visualization

## Adding Custom Models
Create a new model class in `src/models/` that inherits from `BaseModel`.

## Contributing
Please follow the existing code structure and add tests for new features.
