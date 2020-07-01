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
    """Dataset class used to train ResNet23"""
    def __init__(self, main_path, pmap_name="prob_map.tif", img_name="image.tif"):
        """
        :param main_path: (str|Path) Path to main directory containing images (e.g. ./data/lensfree/).
        :param pmap_name: (str) Name of probability map images
        :param img_name: (str) Name of input images
        """
        self.main_path = Path(main_path)
        self.prob_maps = np.array(list(self.main_path.rglob(pmap_name)))
        self.images = np.array(list(self.main_path.rglob(img_name)))
        self.len_images = len(self.images)
        self.len_prob_maps = len(self.prob_maps)
        assert self.len_prob_maps == self.len_images,\
            f"Number of probability maps {self.len_prob_maps} does not match number of images {self.len_images}"
        self.val_split = None
        self.train_ids = None
        self.val_ids = None
        self.train_images = None
        self.train_prob_maps = None
        self.val_prob_maps = None
        self.val_images = None

    def __len__(self):
        return self.len_images

    def __getitem__(self, item):
        img = cv2.imread(str(self.images[item]), -1)
        pm = cv2.imread(str(self.prob_maps[item]), -1)
        return img, pm

    def generate_train_data(self, batch_size=1):
        """Data generator on training data."""
        # TODO add augmentation
        assert (self.train_images is not None) and (self.train_prob_maps is not None),\
            "No training data found. Split dataset before generating training data."

        images, pmaps = [], []
        for img, pm in zip(self.train_images, self.train_prob_maps):
            image, pmap = cv2.imread(str(img), -1), cv2.imread(str(pm), -1)
            if len(image.shape) == 2:
                image = np.repeat(image[..., np.newaxis], 3, axis=2) # Input must be of shape (None, W, H, 3)
            images.append(image)
            pmaps.append(pmap[..., np.newaxis])  # Target must be of shape (None, W, H, 1)
            if (len(images) == batch_size) and (len(pmaps) == batch_size):
                yield np.array(images), np.array(pmaps)
                images, pmaps = [], []

    def generate_val_data(self, batch_size=1):
        """Data generator on validation data."""
        assert (self.val_images is not None) and (self.val_prob_maps is not None), \
            "No validation data found. Split dataset before generating validation data."

        images, pmaps = [], []
        for img, pm in zip(self.val_images, self.val_prob_maps):
            image, pmap = cv2.imread(str(img), -1), cv2.imread(str(pm), -1)
            if len(image.shape) == 2:
                image = np.repeat(image[..., np.newaxis], 3, axis=2)
            images.append(image)
            pmaps.append(pmap[..., np.newaxis])
            if (len(images) == batch_size) and (len(pmaps) == batch_size):
                yield np.array(images), np.array(pmaps)
                images, pmaps = [], []

    def split_data(self, val_split):
        """
        Split dataset in train/val images.
        :param val_split: (float|string)    If float: percent of validation images (e.g. 0.2).
                                            If string: validation image names (e.g. nrk).
        :return: (tuple) training and validation indices.
        """

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
        self.val_images = self.images[val_ids]
        self.val_prob_maps = self.prob_maps[val_ids]
        self.train_ids = train_ids
        self.val_ids = val_ids
        return train_ids, val_ids
