"""
Model checkpoint utilities
"""


class ModelCheckpoint:
    """
    Save model checkpoints during training
    """
    
    def __init__(self, save_dir, monitor='val_loss', mode='min'):
        """
        Initialize checkpoint manager
        
        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max' for monitoring
        """
        pass
    
    def save(self, model, epoch, metrics):
        """
        Save model checkpoint
        
        Args:
            model: Model to save
            epoch: Current epoch
            metrics: Current metrics
        """
        pass
    
    def load_best(self):
        """
        Load best checkpoint
        
        Returns:
            Best model checkpoint
        """
        pass
