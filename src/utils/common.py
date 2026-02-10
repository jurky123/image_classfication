"""
Utility functions and helpers
"""


def get_device():
    """
    Get available device (CPU/GPU)
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    pass


def set_seed(seed=42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    pass


def count_parameters(model):
    """
    Count total and trainable parameters in model
    
    Args:
        model: Model instance
        
    Returns:
        Dictionary with total and trainable parameter counts
    """
    pass


def save_config(config, save_path):
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    pass


def load_config(config_path):
    """
    Load configuration from file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    pass
