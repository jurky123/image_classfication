"""
Base model class for all image classification models
"""

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    
    def __init__(self, num_classes, input_shape=(224, 224, 3)):
        """
        Initialize base model
        
        Args:
            num_classes: Number of output classes
            input_shape: Input image shape (height, width, channels)
        """
        super(BaseModel, self).__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape
    
    def build(self):
        """
        Build the model architecture
        
        Returns:
            Model instance
        """
        raise NotImplementedError("Subclasses must implement build() method")
    
    def forward(self, x):
        """
        Forward pass
        """
        raise NotImplementedError("Subclasses must implement forward() method")
    
    def summary(self):
        """
        Print model summary
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("="*60)
        print(f"Model Summary")
        print("="*60)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
        print("="*60)
    
    def load_weights(self, weights_path):
        """
        Load model weights from file
        
        Args:
            weights_path: Path to weights file
        """
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            self.load_state_dict(state_dict)
            print(f"✓ 模型权重已加载: {weights_path}")
        except Exception as e:
            print(f"✗ 加载权重失败: {str(e)}")
    
    def save_weights(self, weights_path):
        """
        Save model weights to file
        
        Args:
            weights_path: Path to save weights
        """
        try:
            torch.save(self.state_dict(), weights_path)
            print(f"✓ 模型权重已保存: {weights_path}")
        except Exception as e:
            print(f"✗ 保存权重失败: {str(e)}")
