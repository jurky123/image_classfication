#!/usr/bin/env python3
"""Interactive model evaluation with matplotlib visualization."""

import argparse
import random
from pathlib import Path
import time

import torch
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt

from src.models.convnext import ConvNeXtClassifier
from src.data.transforms import get_val_transforms


def main():
    parser = argparse.ArgumentParser(description='Interactive model evaluation')
    parser.add_argument('--data-dir', type=str, default='data/test',
                        help='Test dataset directory (ImageFolder format)')
    parser.add_argument('--model-path', type=str,
                        default='models/saved_models/best_model.pth')
    parser.add_argument('--variant', type=str, default='tiny',
                        choices=['tiny', 'small', 'base'])
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display image window')
    parser.add_argument('--interval', type=float, default=0.0,
                        help='Delay between predictions in seconds')

    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        fallback = Path('data') / 'test'
        if fallback.exists():
            data_dir = fallback
        else:
            raise FileNotFoundError(f"Test dataset not found: {args.data_dir}")

    transforms = get_val_transforms(image_size=args.image_size)
    dataset = ImageFolder(str(data_dir), transform=transforms)
    if len(dataset) == 0:
        raise ValueError(f"No images found in {data_dir}")

    class_names = dataset.classes

    model = ConvNeXtClassifier(
        num_classes=len(class_names),
        variant=args.variant,
        pretrained=False,
        freeze_backbone_init=False
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    def wait_for_next(fig):
        state = {'action': None}

        def on_key(event):
            if event.key in ['n', 'right', 'space']:
                state['action'] = 'next'
            elif event.key in ['q', 'escape']:
                state['action'] = 'quit'

        fig.canvas.mpl_connect('key_press_event', on_key)
        while state['action'] is None:
            plt.pause(0.05)

        return state['action']

    try:
        while True:
            image_path, true_idx = random.choice(dataset.samples)
            true_label = class_names[true_idx]

            image = Image.open(image_path).convert('RGB')
            input_tensor = transforms(image).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_idx = int(torch.argmax(probs, dim=1).item())
                confidence = float(probs[0, pred_idx].item())

            pred_label = class_names[pred_idx]

            print(f"Image: {image_path}")
            print(f"True:  {true_label}")
            print(f"Pred:  {pred_label} (conf={confidence:.4f})")

            if not args.no_show:
                fig = plt.figure(figsize=(6, 6))
                plt.imshow(image)
                plt.axis('off')
                plt.title(
                    f"Pred: {pred_label} ({confidence:.2f})\nTrue: {true_label}"
                    "\nPress N/Space/Right for next, Q/Esc to quit"
                )
                plt.tight_layout()
                plt.show(block=False)
                action = wait_for_next(fig)
                plt.close(fig)
                if action == 'quit':
                    break

            if args.interval > 0:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
