"""
Dataset module to generate training data for ResNet23.

ResNet23 data folder structure must be:

main_path
    -> experiment_id
        -> image[number]_crop[number]
        -> ...
    -> ...
"""

from pathlib import Path
import cv2
import numpy as np


class Dataset:
    def __init__(self, main_path, pmap_name="prob_map.png", img_name="image.png"):
        """

        :param main_path: (string|Path) Path to main directory containing images. E.g. ./data/lensfree/
        :param val_split: (float|string)    If float: percent of validation images (e.g. 0.2).
                                            If string: validation image names (e.g. nrk).
        """
        self.main_path = Path(main_path)
        self.prob_maps = np.array(list(self.main_path.rglob(pmap_name)))
        self.images = np.array(list(self.main_path.rglob(img_name)))
        self.len_images = len(self.images)
        self.len_prob_maps = len(self.prob_maps)
        assert self.len_prob_maps == self.len_images,\
            f"Number of probability maps {self.len_prob_maps} does not match number of images {self.len_images}"
        self.val_split = None
        self.train_images = None
        self.train_prob_maps = None
        self.val_prob_maps = None
        self.val_images = None

    def __len__(self):
        return self.len_images

    def generate_train_data(self):
        # TODO add augmentation
        assert (self.train_images is not None) and (self.train_prob_maps is not None),\
            "No training data found. Split dataset before generating training data."

        for img, pm in zip(self.train_images, self.train_prob_maps):
            image, pmap = cv2.imread(str(img)), cv2.imread(str(pm))
            yield image, pmap

    def generate_val_data(self):
        assert (self.val_images is not None) and (self.val_prob_maps is not None), \
            "No validation data found. Split dataset before generating validation data."

        for img, pm in zip(self.val_images, self.val_prob_maps):
            image, pmap = cv2.imread(str(img)), cv2.imread(str(pm))
            yield image, pmap

    def split_data(self, val_split):
        """Split dataset in train/val images."""
        self.val_split = val_split
        ids = list(range(self.__len__()))
        if type(self.val_split) is float:
            assert 0 < self.val_split < 1,\
                f"Invalid validation split size. Must be: 0 < val_split < 1. Given: {self.val_split} "
            number_of_val_images = int(self.val_split * self.len_images)
            val_ids = np.random.choice(list(range(self.len_images)), number_of_val_images, replace=False)
        elif type(self.val_split) is str:
            val_ids = [i for i, img in enumerate(self.images) if self.val_split in img.parts]
        else:
            raise ValueError(f"Invalid val_split type. Expected flaot or str. Given: {type(self.val_split)}")
        train_ids = list(set(ids) - set(val_ids))
        self.train_images = self.images[train_ids]
        self.train_prob_maps = self.prob_maps[train_ids]
        self.val_prob_maps = self.images[val_ids]
        self.val_images = self.images[val_ids]
        return train_ids, val_ids
