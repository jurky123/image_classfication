#!/usr/bin/env python3
"""
ConvNeXt 迁移学习训练脚本
完整的训练管道示例
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import sys
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.convnext_model import ConvNeXtClassifier, print_model_info
from src.data.preprocessing import get_train_transforms, get_val_transforms
from src.utils.logger import setup_logger


class TrainerConfig:
    """训练配置"""
    def __init__(self):
        self.num_classes = 102
        self.num_epochs = 30
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.warmup_epochs = 2
        self.num_workers = 4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.checkpoint_dir = 'models/checkpoints'
        self.model_save_dir = 'models/saved_models'
        self.log_dir = 'logs'


class ConvNeXtTrainer:
    """ConvNeXt 训练器"""
    
    def __init__(self, config):
        """
        初始化训练器
        
        Args:
            config: 训练配置对象
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # 创建输出目录
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.model_save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # 初始化日志
        self.logger = setup_logger('ConvNeXtTrainer', 
                                  os.path.join(config.log_dir, 'training.log'))
        
        # 初始化模型
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # 最佳指标
        self.best_acc = 0.0
        self.best_loss = float('inf')
    
    def build_model(self, variant='tiny', pretrained=True):
        """
        构建模型
        
        Args:
            variant: 模型变体
            pretrained: 是否使用预训练权重
        """
        self.logger.info(f"构建 ConvNeXt-{variant} 模型...")
        print_model_info(variant)
        
        self.model = ConvNeXtClassifier(
            num_classes=self.config.num_classes,
            variant=variant,
            pretrained=pretrained,
            freeze_backbone=True
        )
        
        self.model = self.model.to(self.device)
        self.model.summary()
        
        self.logger.info(f"模型构建完成")
        self.logger.info(f"总参数: {self.model.get_total_params():,}")
        self.logger.info(f"可训练参数: {self.model.get_trainable_params():,}")
    
    def load_data(self, data_dir='data', image_size=224):
        """
        加载数据集
        
        Args:
            data_dir: 数据目录
            image_size: 图片大小
        """
        self.logger.info(f"加载数据集...")
        
        # 获取数据转换
        train_transforms = get_train_transforms(image_size=image_size, augment=True)
        val_transforms = get_val_transforms(image_size=image_size)
        
        # 创建数据集
        train_dataset = ImageFolder(
            os.path.join(data_dir, 'train'),
            transform=train_transforms
        )
        val_dataset = ImageFolder(
            os.path.join(data_dir, 'val'),
            transform=val_transforms
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        self.logger.info(f"训练集: {len(train_dataset)} 张图片")
        self.logger.info(f"验证集: {len(val_dataset)} 张图片")
    
    def setup_training(self):
        """
        设置优化器和损失函数
        """
        self.logger.info("设置训练参数...")
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器 - 使用 AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # 学习率调度器
        total_steps = len(self.train_loader) * self.config.num_epochs
        warmup_steps = len(self.train_loader) * self.config.warmup_epochs
        
        from torch.optim.lr_scheduler import CosineAnnealingLR
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs,
            eta_min=1e-6
        )
        
        self.logger.info(f"优化器: AdamW (lr={self.config.learning_rate})")
        self.logger.info(f"损失函数: CrossEntropyLoss")
        self.logger.info(f"学习率调度: CosineAnnealingLR (T_max={self.config.num_epochs})")
    
    def train_epoch(self, epoch):
        """
        训练一个 epoch
        
        Args:
            epoch: 当前 epoch
            
        Returns:
            平均损失和准确率
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [TRAIN]")
        
        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # 更新进度条
            acc = 100. * correct / total
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def validate(self, epoch):
        """
        验证
        
        Args:
            epoch: 当前 epoch
            
        Returns:
            平均损失和准确率
        """
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
        """
        在指定 epoch 解冻骨干网络
        
        Args:
            epoch: 当前 epoch
            unfreeze_at_epoch: 何时解冻
        """
        if epoch == unfreeze_at_epoch:
            self.logger.info(f"在 epoch {epoch+1} 解冻骨干网络")
            self.model.unfreeze_backbone(num_stages_to_unfreeze=2)
            
            # 更新优化器
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate * 0.1,  # 较小学习率
                weight_decay=self.config.weight_decay
            )
    
    def train(self, num_epochs=None, unfreeze_at_epoch=10):
        """
        完整训练流程
        
        Args:
            num_epochs: 训练 epoch 数
            unfreeze_at_epoch: 何时解冻骨干网络
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        self.logger.info("="*70)
        self.logger.info("开始训练")
        self.logger.info("="*70)
        
        training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(num_epochs):
            # 解冻骨干网络
            self.unfreeze_backbone(epoch, unfreeze_at_epoch)
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate(epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录历史
            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_loss'].append(val_loss)
            training_history['val_acc'].append(val_acc)
            
            # 日志
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            )
            
            # 保存最佳模型
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
                self.logger.info(f"✓ 最佳模型已保存 (准确率: {val_acc:.2f}%)")
            
            # 定期保存检查点
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)
        
        # 保存训练历史
        self.save_training_history(training_history)
        
        self.logger.info("="*70)
        self.logger.info(f"训练完成！最佳验证准确率: {self.best_acc:.2f}%")
        self.logger.info("="*70)
        
        return training_history
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        保存检查点
        
        Args:
            epoch: 当前 epoch
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
        }
        
        if is_best:
            save_path = os.path.join(
                self.config.model_save_dir,
                'best_model.pth'
            )
        else:
            save_path = os.path.join(
                self.config.checkpoint_dir,
                f'checkpoint_epoch_{epoch+1}.pth'
            )
        
        torch.save(checkpoint, save_path)
    
    def save_training_history(self, history):
        """
        保存训练历史
        
        Args:
            history: 训练历史字典
        """
        save_path = os.path.join(
            self.config.log_dir,
            f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(save_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.logger.info(f"训练历史已保存: {save_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ConvNeXt 迁移学习训练')
    parser.add_argument('--variant', type=str, default='tiny',
                       choices=['tiny', 'small', 'base'],
                       help='ConvNeXt 模型变体')
    parser.add_argument('--epochs', type=int, default=30,
                       help='训练 epoch 数')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='数据目录')
    parser.add_argument('--unfreeze-at', type=int, default=10,
                       help='在第几个 epoch 解冻骨干网络')
    parser.add_argument('--device', type=str, default='auto',
                       help='运行设备')
    
    args = parser.parse_args()
    
    # 设置配置
    config = TrainerConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    
    if args.device != 'auto':
        config.device = args.device
    
    # 创建训练器
    trainer = ConvNeXtTrainer(config)
    
    # 构建模型
    trainer.build_model(variant=args.variant, pretrained=True)
    
    # 加载数据
    trainer.load_data(data_dir=args.data_dir)
    
    # 设置训练
    trainer.setup_training()
    
    # 开始训练
    history = trainer.train(num_epochs=config.num_epochs, unfreeze_at_epoch=args.unfreeze_at)


if __name__ == '__main__':
    main()
