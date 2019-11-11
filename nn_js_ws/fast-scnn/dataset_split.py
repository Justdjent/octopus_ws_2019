import argparse
import os

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Script for creating train/val split.')
    parser.add_argument('--dataset_path', '-dp', type=str, default='/home/quantum/Documents/datasets/new-coco',
                        help='Path to dataset directory on disk')
    parser.add_argument('--split_ratio', '-sr', type=float, default='0.8',
                        help='Train/val split ratio')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(seed=42)
    dataset_path = args.dataset_path
    split_ratio = args.split_ratio

    image_names_df = pd.DataFrame(os.listdir(os.path.join(dataset_path, 'images')), columns=['image_name'])
    np.random.shuffle(image_names_df.values)
    train_df = image_names_df.iloc[:int(len(image_names_df) * split_ratio)]
    val_df = image_names_df.iloc[int(len(image_names_df) * split_ratio):]

    train_df.to_csv(os.path.join(dataset_path, 'train_df.csv'), index=False)
    val_df.to_csv(os.path.join(dataset_path, 'val_df.csv'), index=False)
