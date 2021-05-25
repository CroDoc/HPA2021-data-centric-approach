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
import argparse

import torch

def solve(cell_segmentation, nuc_segmentation, name, cell_save_dir, nuclei_save_dir):
    nuclei_mask, cell_mask = label_cell(nuc_segmentation, cell_segmentation)
    
    ID = name.replace('_red.png','').split('/')[-1]
    if cell_mask.max() < 256:
        np.save(nuclei_save_dir + ID + '.npy', nuclei_mask.astype(np.uint8))
        np.save(cell_save_dir + ID + '.npy', cell_mask.astype(np.uint8))
    else:
        np.save(nuclei_save_dir + ID + '.npy', nuclei_mask)
        np.save(cell_save_dir + ID + '.npy', cell_mask)
            
    return

def run_segmentator(read_dir, cell_save_dir, nuclei_save_dir, batch):

    if not os.path.exists(cell_save_dir):
        os.makedirs(cell_save_dir)

    if not os.path.exists(nuclei_save_dir):
        os.makedirs(nuclei_save_dir)

    NUC_MODEL = "./segmentation/dpn_unet_nuclei_v1.pth"
    CELL_MODEL = "./segmentation/dpn_unet_cell_3ch_v1.pth"

    segmentator = cellsegmentator.CellSegmentator(
        NUC_MODEL,
        CELL_MODEL,
        scale_factor=0.25,
        device="cuda",
        padding=True,
        multi_channel_model=True,
    )

    pos = 0

    mt = glob.glob(read_dir + '*_red.png')
    er = [f.replace('red', 'yellow') for f in mt]
    nu = [f.replace('red', 'blue') for f in mt]
    pos = 0

    while pos < len(mt):

        images = [mt[pos:pos+batch], er[pos:pos+batch], nu[pos:pos+batch]]
        pos += batch

        nuc_segmentations = segmentator.pred_nuclei(images[2])
        cell_segmentations = segmentator.pred_cells(images)

        pool = mp.Pool(mp.cpu_count())

        for i in range(len(cell_segmentations)):
            name = images[0][i]

            pool.apply_async(solve, args=(cell_segmentations[i], nuc_segmentations[i], name, cell_save_dir, nuclei_save_dir))

        pool.close()    
        pool.join()
        print(pos, end = ' ')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', help='input folder', action='store')
    parser.add_argument('-c', '--cells_folder', help='cells save folder', action='store')
    parser.add_argument('-n', '--nuclei_folder', help='nuclei save folder', action='store')
    parser.add_argument('-b', '--batch_size', help='batch size', action='store', type=int)
    args = vars(parser.parse_args())
    
    if not args['input_folder'] or not args['cells_folder'] or not args['nuclei_folder']:
        raise Exception('Please specify input folder, cells folder and nuclei folder!')

    batch_size = 256
    if args['batch_size']:
        batch_size = args['batch_size']

    run_segmentator(args['input_folder'] + '/', args['cells_folder'] + '/', args['nuclei_folder'] + '/', batch_size)