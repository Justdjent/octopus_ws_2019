from fast_scnn_keras import Fast_SCNN
import keras
from metrics import dice_coef
from losses import cce_dice_loss, bce_dice_loss
from data_generator import DataGenerator
import argparse
import cv2
import numpy as np
import pandas as pd
import yaml

from albumentations import (
    Compose, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,
    ShiftScaleRotate, Resize, RandomCrop, OneOf, RandomRotate90, Flip, Transpose, IAAAdditiveGaussianNoise,
    GaussNoise, MotionBlur, MedianBlur, Blur, Crop
)
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.random.set_random_seed(42)
np.random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description='Script for training Fast-SCNN model.')
    parser.add_argument('--dataset_path', '-dp', type=str, default='/home/quantum/Documents/datasets/coco-test/',
                        help='Path to dataset directory')
    parser.add_argument('--train_df', '-td', type=str, default='/home/quantum/Documents/datasets/coco-test/train_df.csv',
                        help='Path to file with train image names')
    parser.add_argument('--val_df', '-vd', type=str, default='/home/quantum/Documents/datasets/coco-test/val_df.csv',
                        help='Path to file with validation image names')
    parser.add_argument('--config_path', '-cp', type=str, default='config.yml',
                        help='Path to experiment config file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    dataset_path = args.dataset_path
    train_df_path = args.train_df
    val_df_path = args.val_df
    config_path = args.config_path

    with open(config_path) as config:
        data = yaml.load(config, Loader=yaml.SafeLoader)
        train_info = data['train']
        dataset_info = data['dataset']

        input_height = dataset_info['image_height']
        input_width = dataset_info['image_width']

        channels = dataset_info['channels']

        resize = train_info['augmentations']['resize']['size']
        batch_size = train_info['batch']
        epochs = train_info['epochs']
        num_classes = train_info['classes']

    AUGMENTATIONS_TRAIN = Compose([
        Resize(resize, resize, always_apply=True),
        RandomCrop(input_height, input_width, always_apply=True),
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        OneOf([
            RandomContrast(limit=0.2, p=0.5),
            RandomGamma(gamma_limit=(80, 120), p=0.5),
            RandomBrightness(limit=0.2, p=0.5)
        ], p=0.2),
        HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20,
                           val_shift_limit=10, p=.3),
        ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.1,
            rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.3),
    ])

    AUGMENTATIONS_VALID = Compose([
        Resize(resize, resize, always_apply=True),
        RandomCrop(input_height, input_width, always_apply=True),
    ])

    model = Fast_SCNN(input_shape=(input_height, input_width, channels),
                      num_classes=num_classes)\
        .model(show_summary=False, activation='sigmoid')
    optimizer = keras.optimizers.SGD(momentum=0.9, lr=0.045)
    model.compile(loss=bce_dice_loss, optimizer=optimizer, metrics=[dice_coef])

    checkpoint = keras.callbacks.ModelCheckpoint(filepath='model.h5',
                                                 monitor='val_dice_coef',
                                                 verbose=1,
                                                 mode='max',
                                                 save_best_only=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_dice_coef',
                                                  verbose=1,
                                                  min_lr=1e-5,
                                                  patience=20,
                                                  mode='max')
    callbacks = [checkpoint]

    train_df = pd.read_csv(train_df_path, header=0)
    train_gen = DataGenerator(train_df, dataset_path,
                              dim=(input_height, input_width),
                              batch_size=batch_size,
                              n_channels=channels,
                              augmentations=AUGMENTATIONS_TRAIN)

    val_df = pd.read_csv(val_df_path, header=0)
    valid_gen = DataGenerator(val_df, dataset_path,
                              dim=(input_height, input_width),
                              batch_size=batch_size,
                              n_channels=channels,
                              augmentations=AUGMENTATIONS_TRAIN)
    model.fit_generator(
        train_gen,
        epochs=epochs,
        validation_data=valid_gen,
        workers=2,
        use_multiprocessing=False,
        callbacks=callbacks
    )
