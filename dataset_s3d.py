from os import listdir, mkdir
from os.path import isfile, isdir, join
import numpy as np
import scipy.misc
from scipy.ndimage import gaussian_filter
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from random import shuffle
from sklearn.model_selection import train_test_split
import math
from tensorflow.keras import layers
vgg_mean = np.array([123, 116, 103])

def get_3D_data():
    s3d_dir = './data/'

    left_image_data, right_image_data, label_data = import_train_data(s3d_dir, 'left', 'right','dense_left', use_cache=False)

    num_val = int(math.floor(len(left_image_data)*0.1))
    val_idx = np.random.choice(len(left_image_data), num_val, replace=False)
    train_idx = np.setdiff1d(range(len(left_image_data)), val_idx)

    X_train = [left_image_data[train_idx], right_image_data[train_idx]]
    Y_train = label_data[train_idx]

    X_val = [left_image_data[val_idx], right_image_data[val_idx]]
    Y_val = label_data[val_idx]
    return X_train, Y_train, X_val, Y_val


#assuming that the dataset structure is as follows
#dataset_root
#--> stimuli_dir
#--> fixation_maps_dir

def import_train_data(dataset_root, left_stimuli_dir, right_stimuli_dir, fixation_maps_dir, use_cache=True):

    if use_cache:
        cache_dir = join(dataset_root, '__cache')
        if not isdir(cache_dir):
            mkdir(cache_dir)

        if isfile(join(cache_dir, 'data.npy')):
            left_image_data, right_image_data, label_data = np.load(join(cache_dir, 'data.npy'))
            return left_image_data, right_image_data, label_data

    image_names = [f for f in listdir(join(dataset_root, left_stimuli_dir)) if isfile(join(dataset_root, left_stimuli_dir , f))]

    left_image_data = np.zeros((len(image_names), 224, 224, 3))
    right_image_data = np.zeros((len(image_names),224, 224, 3))
    label_data = np.zeros((len(image_names), 45, 45, 1))

    for i in range(len(image_names)):
        img_left = img_to_array(load_img(join(dataset_root, left_stimuli_dir, image_names[i]),
            grayscale=False,
            target_size=(224, 224),
            interpolation='nearest'))
        img_right = img_to_array(load_img(join(dataset_root, right_stimuli_dir, image_names[i]),
            grayscale=False,
            target_size=(224, 224),
            interpolation='nearest'))
        label = img_to_array(load_img(join(dataset_root, fixation_maps_dir, image_names[i]),
            grayscale=True,
            target_size=(45, 45),
            interpolation='nearest'))

        img_left -= vgg_mean
        img_right -= vgg_mean

        left_image_data[i] = img_left[None,:]/255
        right_image_data[i] = img_right[None, :]/255
        label_data[i] = label[None, :]/255

    if use_cache:
        np.save(join(cache_dir, 'data.npy'), (left_image_data, right_image_data, label_data))

    return left_image_data, right_image_data, label_data
