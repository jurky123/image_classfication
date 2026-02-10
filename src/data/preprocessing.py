"""
Dataset preprocessing utilities
"""


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess a single image
    
    Args:
        image_path: Path to image file
        target_size: Target image size (height, width)
        
    Returns:
        Preprocessed image tensor
    """
    pass


def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train, validation and test sets
    
    Args:
        data_dir: Path to dataset directory
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        
    Returns:
        train_paths, val_paths, test_paths
    """
    pass


def normalize_image(image, mean, std):
    """
    Normalize image using mean and standard deviation
    
    Args:
        image: Input image
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        
    Returns:
        Normalized image
    """
    pass
