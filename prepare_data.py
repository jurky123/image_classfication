#!/usr/bin/env python3
"""
Data preparation script for Oxford Flowers 102 dataset
Run this script to prepare and organize your dataset
"""

import os
import shutil
import argparse
from pathlib import Path
from scipy.io import loadmat
from tqdm import tqdm


def load_mat_files(raw_dir):
    """
    加载 .mat 文件获取标签和数据集划分信息
    
    Args:
        raw_dir: 原始数据目录
        
    Returns:
        labels: 图片标签字典 {image_id: label}
        splits: 数据集划分字典 {'train': [...], 'val': [...], 'test': [...]}
    """
    print("正在加载标签和数据集划分信息...")
    
    # 加载标签
    labels_mat = loadmat(os.path.join(raw_dir, 'imagelabels.mat'))
    labels = labels_mat['labels'][0]  # 标签从1到102
    
    # 加载数据集划分
    setid_mat = loadmat(os.path.join(raw_dir, 'setid.mat'))
    train_ids = setid_mat['trnid'][0]
    val_ids = setid_mat['valid'][0]
    test_ids = setid_mat['tstid'][0]
    
    # 创建标签字典 (image_id: label)
    labels_dict = {i+1: int(label) for i, label in enumerate(labels)}
    
    # 创建划分字典
    splits = {
        'train': train_ids.tolist(),
        'val': val_ids.tolist(),
        'test': test_ids.tolist()
    }
    
    print(f"  - 训练集: {len(train_ids)} 张图片")
    print(f"  - 验证集: {len(val_ids)} 张图片")
    print(f"  - 测试集: {len(test_ids)} 张图片")
    print(f"  - 类别数: {len(set(labels))}")
    
    return labels_dict, splits


def organize_dataset(raw_dir, data_dir, labels_dict, splits):
    """
    将图片组织到对应的目录结构中
    
    Args:
        raw_dir: 原始数据目录
        data_dir: 目标数据目录
        labels_dict: 标签字典
        splits: 数据集划分
    """
    jpg_dir = os.path.join(raw_dir, 'jpg')
    
    # 处理每个数据集分割
    for split_name, image_ids in splits.items():
        print(f"\n正在组织 {split_name} 数据集...")
        split_dir = os.path.join(data_dir, split_name)
        
        # 遍历每张图片
        for img_id in tqdm(image_ids, desc=f"处理 {split_name}"):
            # 获取图片文件名和标签
            src_img = os.path.join(jpg_dir, f'image_{img_id:05d}.jpg')
            label = labels_dict[img_id]
            
            # 创建类别目录
            class_dir = os.path.join(split_dir, f'class_{label:03d}')
            os.makedirs(class_dir, exist_ok=True)
            
            # 复制图片到目标目录
            dst_img = os.path.join(class_dir, f'image_{img_id:05d}.jpg')
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
            else:
                print(f"警告: 找不到图片 {src_img}")


def generate_statistics(data_dir, splits):
    """
    生成数据集统计信息
    
    Args:
        data_dir: 数据目录
        splits: 数据集划分
    """
    print("\n" + "="*60)
    print("数据集统计信息")
    print("="*60)
    
    for split_name in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split_name)
        if os.path.exists(split_dir):
            classes = sorted([d for d in os.listdir(split_dir) 
                            if os.path.isdir(os.path.join(split_dir, d))])
            total_images = sum([len(os.listdir(os.path.join(split_dir, c))) 
                              for c in classes])
            print(f"\n{split_name.upper()} 集:")
            print(f"  - 类别数: {len(classes)}")
            print(f"  - 图片总数: {total_images}")
            if len(classes) > 0:
                images_per_class = [len(os.listdir(os.path.join(split_dir, c))) 
                                  for c in classes]
                print(f"  - 每类图片数: 最小={min(images_per_class)}, "
                      f"最大={max(images_per_class)}, "
                      f"平均={sum(images_per_class)/len(images_per_class):.1f}")


def main():
    """
    主函数：准备 Oxford Flowers 102 数据集
    """
    parser = argparse.ArgumentParser(description='准备 Oxford Flowers 102 数据集')
    parser.add_argument('--raw-dir', type=str, 
                       default='data/raw',
                       help='原始数据目录')
    parser.add_argument('--data-dir', type=str, 
                       default='data',
                       help='目标数据目录')
    parser.add_argument('--clean', action='store_true',
                       help='清理已存在的处理后数据')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Oxford Flowers 102 数据集准备")
    print("="*60)
    
    # 检查原始数据目录
    if not os.path.exists(args.raw_dir):
        print(f"错误: 找不到原始数据目录 {args.raw_dir}")
        return
    
    # 清理已存在的数据
    if args.clean:
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(args.data_dir, split)
            if os.path.exists(split_dir):
                print(f"清理目录: {split_dir}")
                shutil.rmtree(split_dir)
    
    # 创建目录
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(args.data_dir, split), exist_ok=True)
    
    # 加载标签和划分信息
    labels_dict, splits = load_mat_files(args.raw_dir)
    
    # 组织数据集
    organize_dataset(args.raw_dir, args.data_dir, labels_dict, splits)
    
    # 生成统计信息
    generate_statistics(args.data_dir, splits)
    
    print("\n" + "="*60)
    print("数据准备完成！")
    print("="*60)
    print(f"\n数据集已组织到以下目录:")
    print(f"  - 训练集: {os.path.join(args.data_dir, 'train')}")
    print(f"  - 验证集: {os.path.join(args.data_dir, 'val')}")
    print(f"  - 测试集: {os.path.join(args.data_dir, 'test')}")


if __name__ == "__main__":
    main()
