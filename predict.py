#!/usr/bin/env python3
"""Run inference on images using a trained model."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.datasets import ImageFolder

from src.models.convnext import ConvNeXtClassifier
from src.data.transforms import get_val_transforms


def load_class_names(data_dir):
    dataset = ImageFolder(data_dir, transform=get_val_transforms())
    return dataset.classes


def predict_image(model, image_path, transform, device, top_k=5):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)

    topk_probs, topk_indices = torch.topk(probs, k=min(top_k, probs.size(1)), dim=1)
    return topk_indices[0].tolist(), topk_probs[0].tolist()


def main():
    parser = argparse.ArgumentParser(description='Image classification inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--image', type=str, default=None,
                        help='Single image file to classify')
    parser.add_argument('--dir', type=str, default=None,
                        help='Directory of images to classify')
    parser.add_argument('--variant', type=str, default='tiny',
                        choices=['tiny', 'small', 'base'])
    parser.add_argument('--num-classes', type=int, default=102)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--data-dir', type=str, default='data/val',
                        help='ImageFolder directory for loading class names')
    parser.add_argument('--json', action='store_true',
                        help='Output results as JSON')

    args = parser.parse_args()

    if args.image is None and args.dir is None:
        parser.error("Must specify --image or --dir")

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    try:
        class_names = load_class_names(args.data_dir)
    except Exception:
        class_names = [f'class_{i:03d}' for i in range(args.num_classes)]

    model = ConvNeXtClassifier(
        num_classes=args.num_classes,
        variant=args.variant,
        pretrained=False,
        freeze_backbone_init=False
    ).to(device)

    checkpoint = torch.load(args.model, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    transform = get_val_transforms()

    image_paths = []
    if args.image:
        image_paths = [args.image]
    if args.dir:
        dir_path = Path(args.dir)
        image_paths.extend(sorted([
            str(p) for p in dir_path.iterdir()
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png')
        ]))

    results = []
    for img_path in image_paths:
        indices, confs = predict_image(model, img_path, transform, device, args.top_k)
        predictions = [
            {'rank': i + 1, 'class': class_names[idx], 'confidence': conf}
            for i, (idx, conf) in enumerate(zip(indices, confs))
        ]
        results.append({'image': img_path, 'predictions': predictions})

        if not args.json:
            print(f"\nImage: {img_path}")
            for p in predictions:
                print(f"  {p['rank']}. {p['class']} ({p['confidence']:.4f})")

    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
