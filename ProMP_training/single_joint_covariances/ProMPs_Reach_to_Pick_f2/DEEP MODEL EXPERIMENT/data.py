"""
This module contains all the functions and classes necessary to load the data into a dataset.
Each function is used to load a specific type of data.
The class Dataset is the base class on which all datasets are built and its most important
features are the Dataset.samples attribute and the Dataset.prepare_data() method.
All the other classes inheriting from Dataset in this module are abstract classes, each one
loading a different part of the data. They can be combined into complex datasets which can be
found in the datasets module and can be instantiated and used in the experiments.
"""

import json
import os
from abc import ABC, abstractmethod
import numpy as np
from natsort import natsorted
from skimage.io import imread
from skimage.transform import resize

def normalize_negative_one(img, twod=True):
    normalized_input = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    if twod:
        return 2*normalized_input - 1
    else:
        return 2*normalized_input - 1, np.amin(img), np.amax(img)

def reverse_normalize_negative_one(img, min, max):
    po = (img + 1) / 2
    reversed_input = ((max - min) * po) + min
    return reversed_input


def load_image(img_path, resize_shape=(256, 256, 3), normalize=False):
    '''
    LOAD ORIGINAL RGB IMAGE
    '''
    img = imread(img_path, pilmode='RGB')
    if resize_shape:
        img = resize(img, resize_shape)
    if normalize:
        img = normalize_negative_one(img)
    return img.astype('float64')

def load_images(ids,img_path, resize_shape=(256, 256, 3), normalize=False):
    '''
    LOAD A SET OF RGB IMAGES (the ones listed in images ids)
    '''
    return np.array([load_image(os.path.join(img_path, id + ".png"), resize_shape, normalize) for id in ids])

def load_depth(img_path, resize_shape=(256, 256, 3)):
    '''
    LOAD ORIGINAL RGB IMAGE
    '''
    d = np.load(img_path, allow_pickle=True)
    depth = np.zeros(shape=(480, 640, 3))
    for i in range(3):
        depth[:, :, i] = d
    depth = resize(depth, resize_shape)
    return depth.astype('float64')

def load_depth_images(ids,img_path, resize_shape=(256, 256, 3)):
    '''
    LOAD A SET OF RGB IMAGES (the ones listed in images ids)
    '''
    return np.array([load_depth(os.path.join(img_path, id + ".npy"), resize_shape) for id in ids])


def load_encoded_image(img_path, encoder, resize_shape=(256, 256, 3), normalize=False):
    '''
    LOAD ENCODED IMAGES
    '''
    return np.squeeze(encoder(np.expand_dims(load_image(img_path, resize_shape=resize_shape, normalize=normalize), axis=0))).astype('float64')

def load_encoded_images(ids, img_dir, encoder, resize_shape=(256, 256, 3), normalize=False):
    '''
    LOAD A SET OF ENCODED IMAGES (the ones listed in images ids)
    '''
    return np.array([load_encoded_image(os.path.join(img_dir, id + ".png"), encoder, resize_shape, normalize) for id in ids])

def get_id(filename):
    '''
    Get the id of a .json file
    '''
    return filename.split(sep='/')[-1].strip('.png').split(sep='_')[0]
def get_ids(data_dir):
    '''
    Get a list of ids of the .json files
    '''
    out=np.array(natsorted([get_id(filename) for filename in os.listdir(data_dir)]))
    out=out[:-1]
    return out
def get_images_id(filename):
    '''
    Get the single id of an image.
    '''
    return filename.split(sep='/')[-1].strip('.png')
def get_images_ids(data_dir):
    '''
    Get the list of ids of images in a directory
    '''
    out=np.array(natsorted([get_images_id(filename) for filename in os.listdir(data_dir)]))
    out=out[:-1]
    return out

def load_weights_mean(json_path, mean_key="mean_weights"):
    '''
    Loads the mean weights saved in a .json files as a dictionary with key mean_key
    '''
    with open(json_path, 'r') as fp:
        annotation = json.load(fp)
        return np.asarray(annotation[mean_key]).round(16).astype('float64')
def load_batch_weights_mean(dataset_path, ids):
    '''
    loads a batch of mean of weights
    '''
    return np.asarray([load_weights_mean(os.path.join(dataset_path, id + ".json")) for id in ids], dtype=object)   #(506, 56, 1)

def load_weights_L(json_path, L_key="L"):
    '''
    Loads the non zero values of the Cholesky decomposition of the covariance matrix saved in a .json files as a dictionary with key L.
    '''
    with open(json_path, 'r') as fp:
        annotation = json.load(fp)
        return np.asarray(annotation[L_key]).round(16).astype('float64')

def load_batch_weights_L(dataset_path, ids):
    '''
    loads a batch of non zero values of the Cholesky decomposition of the covariance matrix
    '''
    return np.asarray([load_weights_L(os.path.join(dataset_path, id + ".json")) for id in ids], dtype=object)         #(506, 1596)

class Dataset(ABC):
    """
    The base dataset only loads the IDs of the files and the mean and L of the ProMPs weights
    """
    def __init__(self, dataset_dir,rgb_dir):
        self.samples = {}
        self.dataset_dir = dataset_dir
        self.rgb_dir = rgb_dir
        # Load all IDs, images IDs,mean of weights and fake L elements
        self.samples["id"] = get_ids(self.rgb_dir)                                                   # 506
        self.samples["images_id"] = get_images_ids(self.rgb_dir)                                     # 506
        self.samples["mean_weights"] = load_batch_weights_mean(self.dataset_dir, self.samples["id"]) # (506, 56, 1)
        self.samples["L"] = load_batch_weights_L(self.dataset_dir, self.samples["id"])               # (506, 1596 )
    @abstractmethod
    def _split_train_test_val(self):
        """Return the list of indexes corresponding to training, validation and testing."""
        raise NotImplementedError()

    @abstractmethod
    def prepare_data(self):
        """Prepare the data from self.samples to self.data in the format (X_train, y_train), (X_val, y_val), (X_test, y_test)."""
        raise NotImplementedError()

class DatasetOrdinary(Dataset, ABC):

    def __init__(self, rgb_dir,dataset_dir, **kwargs):
        super().__init__(rgb_dir=rgb_dir,dataset_dir=dataset_dir, **kwargs)

    def _split_train_test_val(self, val_frac=0.0, N_test=0, random_state=None,use_val_in_train=False):
        rng = np.random.default_rng(random_state)
        # Explicit cast to int type since default is float and indices must be integers.
        idx_train = np.array([], dtype=np.int64)
        idx_all = np.arange(0, self.samples["id"].shape[0])
        # Extract N_test test samples.
        idx_test = rng.choice(idx_all, N_test, replace=False)
        idx_train = np.append(idx_train, np.setdiff1d(idx_all, idx_test))
        # Extract N_val validation samples from the whole dataset.
        N_val = int(np.ceil(len(idx_train) * val_frac))
        idx_val = rng.choice(idx_train, N_val, replace=False)
        # N_train samples from the whole dataset are what is left.
        if use_val_in_train==True:
         idx_train=idx_train
        else:
         idx_train = np.setdiff1d(idx_train, idx_val)
        return idx_train, idx_val, idx_test

class DatasetRGB(Dataset, ABC):
    """
    RGB datasets load the RGB or encoded images in the samples.
    """
    def __init__(self, dataset_dir, rgb_dir, depth_dir, **kwargs):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir

        super().__init__(dataset_dir=dataset_dir,rgb_dir=rgb_dir, **kwargs)

        # LOAD FULL IMAGES
        self.samples["img_enc"] = load_images(self.samples["images_id"], self.rgb_dir)
        self.samples["img_enc_D"] = load_depth_images(self.samples["images_id"], self.depth_dir)


