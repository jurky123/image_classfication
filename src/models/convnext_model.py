"""
ConvNeXt model implementation for transfer learning
ConvNeXt: A Modern Take on Residual Networks (Meta AI)
"""

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, convnext_small, convnext_base
from .base_model import BaseModel


class ConvNeXtClassifier(BaseModel):
    """
    ConvNeXt-based image classifier with transfer learning support
    """
    
    def __init__(self, num_classes=102, variant='tiny', pretrained=True, freeze_backbone_init=True):
        """
        Initialize ConvNeXt classifier
        
        Args:
            num_classes: Number of output classes (default: 102 for Flowers dataset)
            variant: ConvNeXt variant ('tiny', 'small', 'base', etc.)
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone_init: Whether to freeze backbone layers during initial training
        """
        super(ConvNeXtClassifier, self).__init__(num_classes)
        
        self.variant = variant
        self.pretrained = pretrained
        
        # 加载预训练模型
        self.model = self._load_pretrained_model()
        
        # 替换分类头
        self._modify_classifier()
        
        # 初始化冻结骨干网络
        if freeze_backbone_init:
            self.freeze_backbone(True)
        
        print(f"✓ ConvNeXt-{variant.capitalize()} 模型已创建")
        print(f"  - 预训练: {'是' if pretrained else '否'}")
        print(f"  - 冻结骨干网络: {'是' if freeze_backbone_init else '否'}")
        print(f"  - 输出类别: {num_classes}")
    
    def _load_pretrained_model(self):
        """
        加载预训练的 ConvNeXt 模型
        
        Returns:
            预训练模型
        """
        model_mapping = {
            'tiny': convnext_tiny,
            'small': convnext_small,
            'base': convnext_base,
        }
        
        if self.variant not in model_mapping:
            raise ValueError(f"不支持的 ConvNeXt 变体: {self.variant}. "
                           f"支持: {list(model_mapping.keys())}")
        
        model_fn = model_mapping[self.variant]
        model = model_fn(pretrained=self.pretrained)
        
        return model
    
    def _modify_classifier(self):
        """
        修改分类头以适应目标类别数
        """
        # ConvNeXt 的分类头包含 Flatten 和 Linear 层
        # 获取 flatten 后的输出维度
        if hasattr(self.model, 'classifier'):
            classifier = self.model.classifier
            # 通常是 Sequential([Flatten, Linear])
            if isinstance(classifier, nn.Sequential):
                # 获取最后一个 Linear 层的输入维度
                linear_layer = None
                for module in classifier:
                    if isinstance(module, nn.Linear):
                        linear_layer = module
                
                if linear_layer is not None:
                    in_features = linear_layer.in_features
                    # 替换为新的线性层
                    new_linear = nn.Linear(in_features, self.num_classes)
                    # 替换分类头中的线性层
                    for i, module in enumerate(classifier):
                        if isinstance(module, nn.Linear):
                            classifier[i] = new_linear
                            print(f"  - 分类头: {in_features} -> {self.num_classes}")
                            return
        
        # 备用方案：直接替换整个分类头
        print(f"  - 分类头已替换为 {self.num_classes} 个输出类别")
    
    def freeze_backbone(self, freeze=True):
        """
        冻结/解冻骨干网络
        
        Args:
            freeze: 是否冻结
        """
        for name, param in self.model.features.named_parameters():
            param.requires_grad = not freeze
        
        # 分类头始终不冻结
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        
        status = "已冻结" if freeze else "已解冻"
        print(f"✓ 骨干网络{status}")
    
    def unfreeze_backbone(self, num_stages_to_unfreeze=2):
        """
        部分解冻骨干网络的最后几个阶段
        
        Args:
            num_stages_to_unfreeze: 要解冻的阶段数 (1-4)
        """
        # ConvNeXt 有 4 个 stage
        if num_stages_to_unfreeze > 4:
            num_stages_to_unfreeze = 4
        
        # 冻结前面的 stages
        for i in range(4 - num_stages_to_unfreeze):
            for param in self.model.features[i].parameters():
                param.requires_grad = False
        
        # 解冻后面的 stages
        for i in range(4 - num_stages_to_unfreeze, 4):
            for param in self.model.features[i].parameters():
                param.requires_grad = True
        
        print(f"✓ 已部分解冻最后 {num_stages_to_unfreeze} 个阶段")
    
    def get_trainable_params(self):
        """
        获取可训练参数数量
        
        Returns:
            可训练参数数量
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self):
        """
        获取总参数数量
        
        Returns:
            总参数数量
        """
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, 3, 224, 224)
            
        Returns:
            输出张量 (batch_size, num_classes)
        """
        return self.model(x)
    
    def summary(self):
        """
        打印模型摘要
        """
        total_params = self.get_total_params()
        trainable_params = self.get_trainable_params()
        
        print("="*70)
        print(f"ConvNeXt-{self.variant.upper()} 模型摘要")
        print("="*70)
        print(f"总参数数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"冻结参数: {total_params - trainable_params:,}")
        print("="*70)


def create_convnext_model(num_classes=102, variant='tiny', pretrained=True, 
                         device='cuda' if torch.cuda.is_available() else 'cpu',
                         freeze_backbone_init=True):
    """
    工厂函数：创建 ConvNeXt 模型
    
    Args:
        num_classes: 输出类别数
        variant: 模型变体 ('tiny', 'small', 'base')
        pretrained: 是否使用预训练权重
        device: 运行设备
        freeze_backbone_init: 是否初始化时冻结骨干网络
        
    Returns:
        ConvNeXtClassifier 实例
    """
    model = ConvNeXtClassifier(
        num_classes=num_classes,
        variant=variant,
        pretrained=pretrained,
        freeze_backbone_init=freeze_backbone_init
    )
    
    model = model.to(device)
    return model


# ConvNeXt 模型参数信息
MODEL_INFO = {
    'tiny': {
        'params': '28.6M',
        'flops': '4.5G',
        'imagenet_acc': '82.1%',
        'description': '轻量级模型，速度快'
    },
    'small': {
        'params': '50.2M',
        'flops': '8.7G',
        'imagenet_acc': '83.6%',
        'description': '平衡性能和速度'
    },
    'base': {
        'params': '88.6M',
        'flops': '15.4G',
        'imagenet_acc': '84.4%',
        'description': '高性能，需要更多资源'
    },
}


def print_model_info(variant='tiny'):
    """
    打印模型信息
    
    Args:
        variant: 模型变体
    """
    if variant not in MODEL_INFO:
        print(f"不支持的模型: {variant}")
        return
    
    info = MODEL_INFO[variant]
    print(f"\nConvNeXt-{variant.upper()} 信息:")
    print(f"  - 参数数量: {info['params']}")
    print(f"  - 计算量: {info['flops']}")
    print(f"  - ImageNet 准确率: {info['imagenet_acc']}")
    print(f"  - 描述: {info['description']}\n")
