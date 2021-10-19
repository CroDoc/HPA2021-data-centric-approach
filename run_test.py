import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import csv
import argparse
import cv2
import numpy as np

from code.dataset_generator import get_test_loader
from code.networks import get_efficient_net
from code.data_cleaner import DataCleaner
from code.solution_values import SolutionValues, merge_solution_values, get_fixed_weights, get_custom_weights

import multiprocessing
import multiprocessing as mp
import glob
from tqdm import tqdm

CPU_COUNT = mp.cpu_count()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate_net(net, test_loader, dataset):

    net.eval()

    solution_values = SolutionValues()

    for data in tqdm(test_loader):

        inputs, indices = data
        inputs = inputs.to(device)
        outputs = net(inputs)

        for batch_index in range(len(indices)):
            dataset_index = indices[batch_index].item()
            metadata = dataset.image_metadata[dataset_index]

            image_name = metadata.image_name
            splitter = image_name.rsplit('_', 1)
            ID, num = splitter[0], int(splitter[1])

            for label in range(18):

                value = outputs[batch_index][label].item()
                solution_values.add_value(ID, num, label, value)

    return solution_values

def get_b4_ensemble(test_loader, dataset):

    print('B4 ensemble - 4 checkpoints:')

    net = get_efficient_net('efficientnet-b4', 'final_models/b4-e12.pt', device)
    solution_values1 = evaluate_net(net, test_loader, dataset)

    net = get_efficient_net('efficientnet-b4', 'final_models/b4-e13.pt', device)
    solution_values2 = evaluate_net(net, test_loader, dataset)

    net = get_efficient_net('efficientnet-b4', 'final_models/b4-e15.pt', device)
    solution_values3 = evaluate_net(net, test_loader, dataset)

    net = get_efficient_net('efficientnet-b4', 'final_models/b4-e16.pt', device)
    solution_values4 = evaluate_net(net, test_loader, dataset)

    solution_values_b4_ensemble = merge_solution_values([solution_values1, solution_values2, solution_values3, solution_values4], [0.15, 0.35, 0.35, 0.15])

    return solution_values_b4_ensemble

def get_b0_resize_ensemble(test_loader, dataset):

    print('B0 resize ensemble - 4 checkpoints:')

    net = get_efficient_net('efficientnet-b0', 'final_models/b0-resize-e12.pt', device)
    solution_values1 = evaluate_net(net, test_loader, dataset)

    net = get_efficient_net('efficientnet-b0', 'final_models/b0-resize-e13.pt', device)
    solution_values2 = evaluate_net(net, test_loader, dataset)

    net = get_efficient_net('efficientnet-b0', 'final_models/b0-resize-e14.pt', device)
    solution_values3 = evaluate_net(net, test_loader, dataset)

    net = get_efficient_net('efficientnet-b0', 'final_models/b0-resize-e15.pt', device)
    solution_values4 = evaluate_net(net, test_loader, dataset)

    solution_values_resize_ensemble = merge_solution_values([solution_values1, solution_values2, solution_values3, solution_values4], [0.20, 0.30, 0.30, 0.20])

    return solution_values_resize_ensemble

def get_b0_resize_and_pad_ensemble(test_loader, dataset):

    print('B0 resize and pad ensemble - 2 checkpoints:')

    net = get_efficient_net('efficientnet-b0', 'final_models/b0-resize-and-pad-e12.pt', device)
    solution_values1 = evaluate_net(net, test_loader, dataset)

    net = get_efficient_net('efficientnet-b0', 'final_models/b0-resize-and-pad-e13.pt', device)
    solution_values2 = evaluate_net(net, test_loader, dataset)

    solution_values_resize_and_pad_ensemble = merge_solution_values([solution_values1, solution_values2], [0.5, 0.5])

    return solution_values_resize_and_pad_ensemble

def get_solo_b0(test_loader, dataset):

    print('B0 solo:')

    net = get_efficient_net('efficientnet-b0', 'final_models/b0-solo.pt', device)
    solution_values_solo_b0 = evaluate_net(net, test_loader, dataset)

    return solution_values_solo_b0

def get_best_b0(test_loader, dataset):

    print('B0 best:')

    net = get_efficient_net('efficientnet-b0', 'final_models/b0-resize-and-pad-e13.pt', device)
    solution_values_best_b0 = evaluate_net(net, test_loader, dataset)

    return solution_values_best_b0

def solve_data_cleaner(image_name, nuclei_masks_folder, cell_masks_folder, dataset_folder):

    dc = DataCleaner(nuclei_masks_folder, cell_masks_folder, dataset_folder)
    value = dc.clean(image_name)

    return image_name, value

def run(opt):

    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)

    _, dataset = get_test_loader(opt.images_folder, opt.batch_size, opt.workers)
    image_names = [x.image_name for x in dataset.image_metadata]
    image_values = []

    pool = mp.Pool(CPU_COUNT)

    for image_name in image_names:
        pool.apply_async(solve_data_cleaner, args=(image_name, opt.nuclei_masks, opt.cell_masks, opt.images_folder,), callback=image_values.append)
    pool.close()
    pool.join()

    test_loader, dataset = get_test_loader(opt.dataset_folder, opt.batch_size, opt.workers)

    border_and_garbage_value = {}
    for ID, vals in image_values:

        for i in range(len(vals)):
            border_and_garbage_value[ID + '_' + str(i)] = vals[i]

    print('RUNNING ' + opt.models + ':')

    if opt.models == 'b0':
        solution_values = get_best_b0(test_loader, dataset)
    elif opt.models == 'final_ensemble_1' or opt.models == 'final_ensemble_2':
        if opt.models == 'final_ensemble_1':
            solution_values_b4_ensemble = get_b4_ensemble(test_loader, dataset)

        solution_values_resize_ensemble = get_b0_resize_ensemble(test_loader, dataset)
        solution_values_resize_and_pad_ensemble = get_b0_resize_and_pad_ensemble(test_loader, dataset)
        solution_values_solo_b0 = get_solo_b0(test_loader, dataset)

        if opt.models == 'final_ensemble_1':
            solution_values = merge_solution_values([solution_values_b4_ensemble, solution_values_resize_ensemble, solution_values_resize_and_pad_ensemble, solution_values_solo_b0], [0.30, 0.30, 0.30, 0.10])
        else:
            solution_values = merge_solution_values([solution_values_resize_ensemble, solution_values_resize_and_pad_ensemble, solution_values_solo_b0], [0.4, 0.4, 0.2])
    else:
        raise Exception('invalid models!')

    if not opt.no_postprocessing:
        if opt.models != 'final_ensemble_1':
            cell_weights, image_weights = get_fixed_weights()
        else:
            cell_weights, image_weights = get_custom_weights()

        solution_values.calculate_negatives()
        solution_values.weight_border_and_garbage_images(border_and_garbage_value)
        solution_values.weight_cells_per_image(cell_weights, image_weights, border_and_garbage_value)

    output_filename = opt.output_folder + opt.output_filename + '.xlsx'
    output_sheet = opt.models
    if opt.no_postprocessing:
        output_sheet += '_np'

    solution_values.to_output_table(output_filename, output_sheet, opt.append, opt.short)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images_folder', help='images folder', action='store', required=True)
    parser.add_argument('-d', '--dataset_folder', help='dataset folder', action='store', required=True)
    parser.add_argument('-o', '--output_filename', help='output filename', action='store', required=True)
    parser.add_argument('-f', '--output_folder', help='output folder', action='store', required=True)
    parser.add_argument('-c', '--cell_masks', help='cell masks folder', action='store', required=True)
    parser.add_argument('-n', '--nuclei_masks', help='nuclei masks folder', action='store', required=True)

    parser.add_argument('-m', '--models', default='b0', help='b0, final_ensemble_1 or final_ensemble_2', action='store')
    parser.add_argument('-b', '--batch_size', default=8, help='batch size', action='store', type=int)
    parser.add_argument('-w', '--workers', default=16, help='number of workers', action='store', type=int)
    parser.add_argument('--no_postprocessing', help='used for getting gramdcam scores for b0', action='store_true')
    parser.add_argument('-a', '--append', help='append to output file', action='store_true')
    parser.add_argument('-s', '--short', default=None, help='store x decimal places', action='store', type=int)

    opt = parser.parse_known_args()[0]

    if opt.images_folder[-1] != '/':
        opt.images_folder += '/'

    if opt.dataset_folder[-1] != '/':
        opt.dataset_folder += '/'

    if opt.output_folder[-1] != '/':
        opt.output_folder += '/'

    if opt.cell_masks[-1] != '/':
        opt.cell_masks += '/'

    if opt.nuclei_masks[-1] != '/':
        opt.nuclei_masks += '/'

    run(opt)
