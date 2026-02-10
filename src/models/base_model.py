"""
Base model class for all image classification models
"""


class BaseModel:
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
        pass
    
    def build(self):
        """
        Build the model architecture
        
        Returns:
            Model instance
        """
        raise NotImplementedError("Subclasses must implement build() method")
    
    def summary(self):
        """
        Print model summary
        """
        pass
    
    def load_weights(self, weights_path):
        """
        Load model weights from file
        
        Args:
            weights_path: Path to weights file
        """
        pass
    
    def save_weights(self, weights_path):
        """
        Save model weights to file
        
        Args:
            weights_path: Path to save weights
        """
        pass
