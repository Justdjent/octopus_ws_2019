import argparse
import numpy as np
import yaml
import os
import pandas as pd
import cv2
from albumentations import Resize
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from fast_scnn_keras import Fast_SCNN


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def evaluation(model_weights_path, dataset_path, val_df_path, input_height, input_width,
               start_threshold=0.05, end_threshold=0.9, step=0.05):
    val_images = pd.read_csv(val_df_path, header=0)['image_name'].values

    model = Fast_SCNN(input_shape=(input_height, input_width, channels),
                      num_classes=num_classes) \
        .model(show_summary=False, activation='sigmoid')
    model.load_weights(model_weights_path)
    images_path = os.path.join(dataset_path, 'images')
    masks_path = os.path.join(dataset_path, 'masks')
    for threshold in np.arange(start_threshold, end_threshold, step):
        dice_values = []
        print(f'Measuring at threshold {threshold}')
        for filename in tqdm(val_images):
            image = cv2.imread(os.path.join(images_path, filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = np.load(os.path.join(masks_path, "{0}.npy".format(filename.split('.')[0])))
            resize_augm = Resize(input_height, input_width)
            augmented = resize_augm(image=image, mask=mask)

            resized_image = augmented['image'] / 255
            resized_mask = augmented['mask']
            prediction = model.predict(np.expand_dims(resized_image, axis=0))
            thresholded_prediction = prediction > threshold
            dice = dice_coef(y_true=resized_mask, y_pred=thresholded_prediction)
            dice_values.append(dice)
        print(f'Average value of {np.array(dice_values).mean()} at threshold {threshold}')


def parse_args():
    parser = argparse.ArgumentParser(description='Script for training Fast-SCNN model.')
    parser.add_argument('--dataset_path', '-dp', type=str,
                        default='/home/quantum/Documents/datasets/coco-test/',
                        help='Path to dataset directory')
    parser.add_argument('--model_weights_path', '-mwp', type=str,
                        default='/home/quantum/Documents/datasets/coco-test/model.h5',
                        help='Path to file with model weights')
    parser.add_argument('--val_df', '-vd', type=str,
                        default='/home/quantum/Documents/datasets/coco-test/val_df.csv',
                        help='Path to file with validation image names')
    parser.add_argument('--config_path', '-cp', type=str,
                        default='config.yml',
                        help='Path to experiment config file')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    dataset_path = args.dataset_path
    val_df_path = args.val_df
    config_path = args.config_path
    model_weights_path = args.model_weights_path
    with open(config_path) as config:
        data = yaml.load(config, Loader=yaml.SafeLoader)
        train_info = data['train']
        dataset_info = data['dataset']

        input_height = dataset_info['image_height']
        input_width = dataset_info['image_width']
        channels = dataset_info['channels']
        num_classes = train_info['classes']

    evaluation(model_weights_path=model_weights_path,
               val_df_path=val_df_path,
               dataset_path=dataset_path,
               input_height=input_height,
               input_width=input_width)
