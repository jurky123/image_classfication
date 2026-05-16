import os
import json
from datetime import datetime
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from .models.convnext import ConvNeXtClassifier, print_model_info
from .data.transforms import get_train_transforms, get_val_transforms


@dataclass
class TrainerConfig:
    num_classes: int = 102
    num_epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1.6e-4
    weight_decay: float = 1.2e-5
    warmup_epochs: int = 2
    num_workers: int = 4
    augment: str = 'strong'
    device: str = 'cuda'
    checkpoint_dir: str = 'models/checkpoints'
    model_save_dir: str = 'models/saved_models'
    log_dir: str = 'logs'


class ConvNeXtTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.model_save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        self.best_acc = 0.0
        self.best_loss = float('inf')
        self.resume_from_best = False

    def build_model(self, variant='tiny', pretrained=True):
        print(f"Building ConvNeXt-{variant} model...")
        print_model_info(variant)

        self.model = ConvNeXtClassifier(
            num_classes=self.config.num_classes,
            variant=variant,
            pretrained=pretrained,
            freeze_backbone_init=True
        )

        self.model = self.model.to(self.device)
        self.model.summary()

        print(f"Model built")
        print(f"Total params: {self.model.get_total_params():,}")
        print(f"Trainable params: {self.model.get_trainable_params():,}")

    def load_best_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                self.config.model_save_dir, 'best_model.pth')

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return False

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            self.best_acc = checkpoint.get('best_acc', 0.0)
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.resume_from_best = True
        print(f"Loaded best model: {checkpoint_path}")
        return True

    def load_data(self, data_dir='data', image_size=224):
        print("Loading dataset...")

        train_transforms = get_train_transforms(
            image_size=image_size, augment=self.config.augment)
        val_transforms = get_val_transforms(image_size=image_size)

        train_dataset = ImageFolder(
            os.path.join(data_dir, 'train'), transform=train_transforms)
        val_dataset = ImageFolder(
            os.path.join(data_dir, 'val'), transform=val_transforms)

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
            shuffle=True, num_workers=self.config.num_workers, pin_memory=True)

        self.val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=self.config.num_workers, pin_memory=True)

        print(f"Train set: {len(train_dataset)} images")
        print(f"Val set: {len(val_dataset)} images")

    def setup_training(self):
        print("Setting up training...")

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.config.num_epochs, eta_min=1e-6)

        print(f"Optimizer: AdamW (lr={self.config.learning_rate})")
        print(f"Loss: CrossEntropyLoss")
        print(f"Scheduler: CosineAnnealingLR (T_max={self.config.num_epochs})")

    def train_epoch(self, epoch):
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [TRAIN]")

        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            acc = 100. * correct / total
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100. * correct / total

        return avg_loss, avg_acc

    def validate(self, epoch):
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [VAL]")

            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                acc = 100. * correct / total
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})

        avg_loss = total_loss / len(self.val_loader)
        avg_acc = 100. * correct / total

        return avg_loss, avg_acc

    def unfreeze_backbone(self, epoch, unfreeze_at_epoch=10):
        if epoch == unfreeze_at_epoch:
            print(f"Unfreezing backbone at epoch {epoch+1}")
            self.model.unfreeze_backbone(num_stages_to_unfreeze=2)

            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate * 0.1,
                weight_decay=self.config.weight_decay
            )

    def train(self, num_epochs=None, unfreeze_at_epoch=10):
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        print("=" * 70)
        print("Starting training")
        print("=" * 70)

        if self.resume_from_best:
            print("Resumed from best model, immediately unfreezing backbone")
            self.model.unfreeze_backbone(num_stages_to_unfreeze=2)

        training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        for epoch in range(num_epochs):
            self.unfreeze_backbone(epoch, unfreeze_at_epoch)

            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            self.scheduler.step()

            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_loss'].append(val_loss)
            training_history['val_acc'].append(val_acc)

            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            )

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
                print(f"Best model saved (acc: {val_acc:.2f}%)")

            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)

        self.save_training_history(training_history)

        print("=" * 70)
        print(f"Training complete! Best val accuracy: {self.best_acc:.2f}%")
        print("=" * 70)

        return training_history

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
        }

        if is_best:
            save_path = os.path.join(self.config.model_save_dir, 'best_model.pth')
        else:
            save_path = os.path.join(
                self.config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')

        torch.save(checkpoint, save_path)

    def save_training_history(self, history):
        save_path = os.path.join(
            self.config.log_dir,
            f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

        with open(save_path, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"Training history saved: {save_path}")
