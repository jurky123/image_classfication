"""
ConvNeXt model implementation for transfer learning
ConvNeXt: A Modern Take on Residual Networks (Meta AI)
"""

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, convnext_small, convnext_base


class ConvNeXtClassifier(nn.Module):
    def __init__(self, num_classes=102, variant='tiny', pretrained=True, freeze_backbone_init=True):
        super().__init__()
        self.num_classes = num_classes
        
        self.variant = variant
        self.pretrained = pretrained
        
        # 加载预训练模型
        self.model = self._load_pretrained_model()
        
        # 替换分类头
        self._modify_classifier()
        
        # 初始化冻结骨干网络
        if freeze_backbone_init:
            self.freeze_backbone(True)
        
        print(f"[OK] ConvNeXt-{variant.capitalize()} created")
        print(f"  - Pretrained: {'yes' if pretrained else 'no'}")
        print(f"  - Frozen backbone: {'yes' if freeze_backbone_init else 'no'}")
        print(f"  - Output classes: {num_classes}")
    
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
                            print(f"  - Classifier head: {in_features} -> {self.num_classes}")
                            return

        print(f"  - Classifier head replaced: {self.num_classes} output classes")
    
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
        
        status = "frozen" if freeze else "unfrozen"
        print(f"[OK] Backbone {status}")
    
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
        
        print(f"[OK] Unfroze last {num_stages_to_unfreeze} stages")
    
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
        print(f"ConvNeXt-{self.variant.upper()} Summary")
        print("="*70)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Frozen params: {total_params - trainable_params:,}")
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
        'description': 'Lightweight, fast'
    },
    'small': {
        'params': '50.2M',
        'flops': '8.7G',
        'imagenet_acc': '83.6%',
        'description': 'Balanced performance and speed'
    },
    'base': {
        'params': '88.6M',
        'flops': '15.4G',
        'imagenet_acc': '84.4%',
        'description': 'High performance, more resources'
    },
}


def print_model_info(variant='tiny'):
    """
    打印模型信息
    
    Args:
        variant: 模型变体
    """
    if variant not in MODEL_INFO:
        print(f"Unsupported model variant: {variant}")
        return

    info = MODEL_INFO[variant]
    print(f"\nConvNeXt-{variant.upper()} Info:")
    print(f"  - Parameters: {info['params']}")
    print(f"  - FLOPs: {info['flops']}")
    print(f"  - ImageNet accuracy: {info['imagenet_acc']}")
    print(f"  - Description: {info['description']}\n")
