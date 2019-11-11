import os
import shutil
import argparse

import numpy as np
from pycocotools.coco import COCO
from pycocotools.mask import decode, frPyObjects
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Script for creating train/val split.')
    parser.add_argument('--dataset_path', '-dp', type=str, default='/home/quantum/Documents/datasets/coco',
                        help='Path to dataset directory on disk')
    parser.add_argument('--save_path', '-sp', type=str, default='/home/quantum/Documents/datasets/coco-filtered',
                        help='Path to where formatted dataset will be stored')
    parser.add_argument('--classes', '-c', nargs='+', default=['cup', 'laptop', 'cell phone'],
                        help='List of classes from coco dataset to be used')
    args = parser.parse_args()
    return args


def get_images_by_cats(cat_ids):
    img_ids = set()
    for val in cat_ids:
        category_img_ids = coco.getImgIds(catIds=val)
        for imgId in category_img_ids:
            img_ids.add(imgId)
    return coco.loadImgs(list(img_ids))


if __name__ == "__main__":
    args = parse_args()
    dataset_path = args.dataset_path
    save_path = args.save_path
    classes = args.classes

    train_val_split_info = {
        'train': {
            'ann_file': 'annotations/instances_train2017.json',
            'initial_images_path': 'train2017'
        },
        'val': {
            'ann_file': 'annotations/instances_val2017.json',
            'initial_images_path': 'val2017'
        }
    }
    for part_of_split in train_val_split_info:
        split_info = train_val_split_info[part_of_split]
        annotation_file = split_info['ann_file']
        initial_images_path = os.path.join(dataset_path, split_info['initial_images_path'])
        images_path = os.path.join(save_path, 'images')
        masks_path = os.path.join(save_path, 'masks')
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(masks_path, exist_ok=True)

        coco = COCO(os.path.join(dataset_path, annotation_file))

        cat_ids = coco.getCatIds(catNms=classes)
        cat_ids_number = {}
        for idx, catId in enumerate(cat_ids):
            cat_ids_number[catId] = idx + 1
        images = get_images_by_cats(cat_ids=cat_ids)

        for image in tqdm(images):
            annIds = coco.getAnnIds(imgIds=image['id'], catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(annIds)
            resulting_mask = np.zeros((image['height'], image['width']), dtype='uint8')
            masked = np.zeros((image['height'], image['width']), dtype='uint8')
            for annotation in anns:
                if type(annotation['segmentation']) == list:
                    # polygons
                    mask = decode(frPyObjects([annotation['segmentation'][0]], image['height'], image['width']))
                else:
                    # rle
                    if type(annotation['segmentation']['counts']) == list:
                        mask = decode(frPyObjects([annotation['segmentation']], image['height'], image['width']))
                    else:
                        mask = decode([annotation['segmentation']])
                mask = mask[:, :, 0]
                resulting_mask += mask * cat_ids_number[annotation['category_id']]
                resulting_mask -= np.logical_and(masked, mask).astype('uint8') * cat_ids_number[
                    annotation['category_id']]
                masked = np.logical_or(masked, mask)
            if resulting_mask.max() > 3:
                print('Something went wrong during mask processing.')
                break
            shutil.copyfile(os.path.join(initial_images_path, image['file_name']),
                            os.path.join(images_path, image['file_name']))
            np.save(os.path.join(masks_path, "{0}.npy".format(image['file_name'].split('.')[0])),
                    resulting_mask)
