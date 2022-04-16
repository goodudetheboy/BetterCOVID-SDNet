import os

import numpy as np

import cv2

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras

from glob import glob
from tqdm import tqdm

SEGMENTATION_MODEL_DIR = os.path.join("..", "input")

SEGMENTATION_MODEL = os.path.join(SEGMENTATION_MODEL_DIR, "unet_lung_seg.hdf5")
OUTPUT_DIR = "cleaned-data"
OUTPUT_DIR_N = os.path.join(OUTPUT_DIR, "N")
OUTPUT_DIR_P = os.path.join(OUTPUT_DIR, "P")

INPUT_DIR = "Data"
INPUT_DIR_N = os.path.join(INPUT_DIR, "N")
INPUT_DIR_P = os.path.join(INPUT_DIR, "P")

def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

segmentation_model = load_model(SEGMENTATION_MODEL, \
                                custom_objects={'dice_coef_loss': dice_coef_loss, \
                                                'dice_coef': dice_coef})


def image_to_train(img):
    npy = img / 255
    npy = np.reshape(npy, npy.shape + (1,))
    npy = np.reshape(npy,(1,) + npy.shape)
    return npy

def train_to_image(npy):
    img = (npy[0,:, :, 0] * 255.).astype(np.uint8)
    return img

# offset is roughly 2.5% of 512
# CROP_OFFSET = 0/
CROP_OFFSET = 13

def segment_image(pid, img, save_to):
    img = cv2.resize(img, (512, 512))
    segm_ret = segmentation_model.predict(image_to_train(img), \
                                          verbose=0)
    segm_s = np.squeeze(segm_ret)
    segm_s = np.where(segm_s > 0.5, 1, 0)
    
    up = left = 512
    down = right = 0

    for i in range(len(segm_s)):
        for j in range(len(segm_s[i])):
            if segm_s[i][j] == 1:
                up = min(up, i)
                down = max(down, i)
                left = min(left, j)
                right = max(right, j)

    # apply offset to cropping
    up = max(0, up-CROP_OFFSET)
    left = max(0, left-CROP_OFFSET)
    down = min(512, down+CROP_OFFSET)
    right = min(512, right+CROP_OFFSET)

    h_crop = down - up    
    w_crop = right - left

    cropped_image = img[up:up+h_crop, left:left+w_crop]
    # plt.imshow(cropped_image, cmap='gray')
    cv2.imwrite(os.path.join(save_to, "%s.jpg" % pid), cropped_image)


for filename in tqdm(glob(os.path.join(INPUT_DIR_N, "*.jpg"))):
    pid, fileext = os.path.splitext(os.path.basename(filename))
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    segment_image(pid, img, OUTPUT_DIR_N)

for filename in tqdm(glob(os.path.join(INPUT_DIR_P, "*.jpg"))):
    pid, fileext = os.path.splitext(os.path.basename(filename))
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    segment_image(pid, img, OUTPUT_DIR_P)
