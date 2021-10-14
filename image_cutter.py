import hpacellseg.cellsegmentator as cellsegmentator
from hpacellseg.utils import label_cell, label_nuclei
import glob
import os
import imageio
import pandas as pd
import pickle
import random

import numpy as np
import multiprocessing
import multiprocessing as mp
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

import base64
from pycocotools import _mask as coco_mask
import typing as t
import zlib
import ast
import cv2
import argparse

IMG_SIZE = 512

def get_paddings(x_len, y_len):

    pad_left, pad_right, pad_up, pad_down = 0, 0, 0, 0

    if x_len > y_len:
        pad_left = (x_len-y_len)//2
        pad_right = x_len-y_len - pad_left
    else:
        pad_up = (y_len-x_len)//2
        pad_down = y_len-x_len - pad_up

    return pad_left, pad_right, pad_up, pad_down

def load_this_image(image_path):
    image = np.array(cv2.imread(image_path, cv2.IMREAD_UNCHANGED))
    
    if image.dtype == np.uint8:
        image = image.astype(np.uint16)
        image *= 257
    
    return image

def get_cropped_channel_image(image, top_left, bottom_right, mask, cell_id):
    
    image = image[top_left[0]:bottom_right[0]+1,top_left[1]:bottom_right[1]+1].copy()

    image[(mask != 0) * (mask != cell_id)] = 0

    pad_left, pad_right, pad_up, pad_down = get_paddings(image.shape[0], image.shape[1])
    
    image = np.pad(image, [(pad_up, pad_down), (pad_left, pad_right)], mode='constant')

    return image

def get_mask_info(mask):
    true_points = np.argwhere(mask)
    
    if not true_points.any():
        return np.array([0, 0]), np.array([0, 0])
    
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)

    return top_left, bottom_right

def solve(name, read_dir, save_dir, mask_dir):
    
    ID = name.replace('_red.png','').split('/')[-1]
    
    cell_mask = np.load(mask_dir + ID + '.npy')
    max_id = cell_mask.max()

    image_per_channel = {}

    for channel in ['_red', '_blue', '_green', '_yellow']:

        image_filename = ID + channel + '.png'
        image = load_this_image(read_dir + image_filename)

        image_per_channel[channel] = image
    
    for curr_id in range(1, max_id+1):
        
        cell_info = get_mask_info(cell_mask == curr_id)
        
        top_left, bottom_right = cell_info[0], cell_info[1]
        
        dim1 = bottom_right[0]+1 - top_left[0]
        dim2 = bottom_right[1]+1 - top_left[1]

        pad_left, pad_right, pad_up, pad_down = get_paddings(dim1, dim2)

        top_left[1] -= pad_left + 16
        bottom_right[1] += pad_right + 16
        top_left[0] -= pad_up + 16
        bottom_right[0] += pad_down + 16

        top_left[0] = max(0, top_left[0])
        top_left[1] = max(0, top_left[1])
        bottom_right[0] = min(cell_mask.shape[0], bottom_right[0])
        bottom_right[1] = min(cell_mask.shape[1], bottom_right[1])
        
        for channel in ['_red', '_blue', '_green']:#, '_yellow']:
        
            save_path = save_dir + ID + '_' + str(curr_id-1) + channel + '.png'

            big_image = image_per_channel[channel]
            image = get_cropped_channel_image(big_image, cell_info[0], cell_info[1], cell_mask[top_left[0]:bottom_right[0]+1,top_left[1]:bottom_right[1]+1], curr_id)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(save_path, image)
            
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', help='input folder', action='store')
    parser.add_argument('-m', '--masks_folder', help='masks folder', action='store')
    parser.add_argument('-o', '--output_folder', help='output folder', action='store')
    args = vars(parser.parse_args())

    if not args['input_folder'] or not args['output_folder'] or not args['masks_folder']:
        raise Exception('Please specify input folder, output folder and masks folder!')

    read_dir = args['input_folder'] + '/'
    save_dir = args['output_folder'] + '/'
    mask_dir = args['masks_folder'] + '/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #if not os.path.exists(mask_dir):
        #os.makedirs(mask_dir)

    mt = glob.glob(read_dir + '*_red.png')

    pool = mp.Pool(mp.cpu_count())

    for x in mt:
        pool.apply_async(solve, args=(x,read_dir, save_dir, mask_dir,))

    pool.close()    
    pool.join()
