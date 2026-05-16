"""
Dataset preprocessing utilities for image classification
"""

import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def preprocess_image(image_path, target_size=(224, 224), mode='RGB'):
    """
    预处理单张图片
    
    Args:
        image_path: 图片文件路径
        target_size: 目标图片尺寸 (height, width)
        mode: 图片模式 ('RGB' 或 'L' for grayscale)
        
    Returns:
        处理后的图片数组
    """
    try:
        # 使用 PIL 读取图片
        image = Image.open(image_path).convert(mode)
        
        # 调整大小
        image = image.resize(target_size, Image.BILINEAR)
        
        # 转换为 numpy 数组
        image_array = np.array(image)
        
        return image_array
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {str(e)}")
        return None


def get_train_transforms(image_size=224, augment=True):
    """
    获取训练数据转换
    
    Args:
        image_size: 图片大小
        augment: 是否使用数据增强
        
    Returns:
        torchvision transforms
    """
    if isinstance(augment, bool):
        augment_level = 'basic' if augment else 'none'
    else:
        augment_level = str(augment).lower().strip()

    if augment_level == 'none':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    if augment_level == 'strong':
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.25, contrast=0.25,
                                  saturation=0.25, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15),
                                    scale=(0.9, 1.1), shear=10),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')
        ])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                              saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(image_size=224):
    """
    获取验证/测试数据转换
    
    Args:
        image_size: 图片大小
        
    Returns:
        torchvision transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    将数据集划分为训练、验证和测试集
    
    Args:
        data_dir: 数据集目录路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        
    Returns:
        train_paths, val_paths, test_paths: 路径列表
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "比例之和必须等于1"
    
    all_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append(os.path.join(root, file))
    
    # 打乱数据
    np.random.shuffle(all_files)
    
    # 计算划分索引
    n = len(all_files)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_paths = all_files[:train_end]
    val_paths = all_files[train_end:val_end]
    test_paths = all_files[val_end:]
    
    return train_paths, val_paths, test_paths


def normalize_image(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    使用均值和标准差归一化图片
    
    Args:
        image: 输入图片 (numpy array or tensor)
        mean: 归一化均值
        std: 归一化标准差
        
    Returns:
        归一化后的图片
    """
    if isinstance(image, np.ndarray):
        # numpy array
        image = image.astype(np.float32) / 255.0
        mean = np.array(mean).reshape(1, 1, 3)
        std = np.array(std).reshape(1, 1, 3)
        normalized = (image - mean) / std
        return normalized
    elif torch.is_tensor(image):
        # PyTorch tensor
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
        normalized = (image - mean) / std
        return normalized
    else:
        raise ValueError("图片必须是 numpy array 或 PyTorch tensor")


def denormalize_image(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    反归一化图片用于可视化
    
    Args:
        image: 归一化后的图片 (numpy array or tensor)
        mean: 归一化均值
        std: 归一化标准差
        
    Returns:
        反归一化后的图片
    """
    if isinstance(image, np.ndarray):
        mean = np.array(mean).reshape(1, 1, 3)
        std = np.array(std).reshape(1, 1, 3)
        denormalized = (image * std + mean) * 255.0
        return np.clip(denormalized, 0, 255).astype(np.uint8)
    elif torch.is_tensor(image):
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
        denormalized = (image * std + mean) * 255.0
        return torch.clamp(denormalized, 0, 255).byte()
    else:
        raise ValueError("图片必须是 numpy array 或 PyTorch tensor")


def calculate_dataset_statistics(data_dir):
    """
    Compute dataset mean and std for normalization.

    Args:
        data_dir: Dataset directory path

    Returns:
        mean, std: Per-channel mean and standard deviation
    """
    import cv2
    print("Computing dataset statistics...")

    all_images = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img.astype(np.float32) / 255.0
                    all_images.append(img)

    all_pixels = np.concatenate([img.reshape(-1, 3) for img in all_images])
    mean = np.mean(all_pixels, axis=0)
    std = np.std(all_pixels, axis=0)

    print(f"Mean: {mean}")
    print(f"Std: {std}")

    return mean, std
