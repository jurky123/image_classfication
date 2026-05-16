#!/usr/bin/env python3
"""Train a ConvNeXt image classifier."""

import argparse
from src.trainer import TrainerConfig, ConvNeXtTrainer


def main():
    parser = argparse.ArgumentParser(description='ConvNeXt transfer learning training')
    parser.add_argument('--variant', type=str, default='tiny',
                        choices=['tiny', 'small', 'base'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1.4e-4)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--unfreeze-at', type=int, default=15,
                        help='Epoch to unfreeze backbone')
    parser.add_argument('--augment', type=str, default='strong',
                        choices=['none', 'basic', 'strong'])
    parser.add_argument('--resume-best', action='store_true',
                        help='Resume from saved best model')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    config = TrainerConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.augment = args.augment
    if args.device != 'auto':
        config.device = args.device

    trainer = ConvNeXtTrainer(config)
    trainer.build_model(variant=args.variant, pretrained=True)

    if args.resume_best:
        trainer.load_best_checkpoint()

    trainer.load_data(data_dir=args.data_dir)
    trainer.setup_training()
    trainer.train(num_epochs=config.num_epochs, unfreeze_at_epoch=args.unfreeze_at)


if __name__ == '__main__':
    main()
