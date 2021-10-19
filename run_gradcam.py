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

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run(opt):

    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)

    cam_loader, dataset = get_test_loader(opt.input_folder, opt.batch_size, opt.workers)
    net = get_efficient_net(opt.network, opt.network_weights, device)

    if opt.gradcam_layer == 'last_block':
        target_layer = net._blocks[-1]
    elif opt.gradcam_layer == 'conv_head':
        target_layer = net._conv_head
    elif opt.gradcam_layer == 'bn1':
        target_layer = net._bn1
    else:
        raise Exception('Invalid gradcam layer!')

    cam = GradCAM(model=net, target_layer=target_layer, use_cuda=(device=="cuda:0"))

    for i, data in enumerate(cam_loader, 0):

        images, indices = data
        inputs = images.to(device)

        grayscale_cams = cam(input_tensor=inputs, aug_smooth=True)

        for batch_index in range(len(indices)):
            dataset_index = indices[batch_index].item()
            metadata = dataset.image_metadata[dataset_index]

            image_filename = opt.output_folder + '/' + metadata.image_name + '.png'

            grayscale_cam = grayscale_cams[batch_index, :]
            rgb_image = np.array(images[batch_index,:3])
            rgb_image = rgb_image.transpose(1, 2, 0)
            rgb_image = rgb_image[...,[2,1,0]]

            cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=False)#, colormap=cv2.COLORMAP_TURBO)

            cv2.imwrite(image_filename, cam_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', help='input folder', action='store', required=True)
    parser.add_argument('-o', '--output_folder', default = 'CAM_output', help='output folder', action='store')
    parser.add_argument('-n', '--network_weights', default='final_models/b0-resize-and-pad-e13.pt', help='weights location for EfficientNet network', action='store')
    parser.add_argument('--network', default='efficientnet-b0', help='EfficientNet network name - change if using b4', action='store')
    parser.add_argument('-b', '--batch_size', default=8, help='batch size', action='store', type=int)
    parser.add_argument('-w', '--workers', default=16, help='number of workers', action='store', type=int)
    parser.add_argument('--gradcam_layer', default='bn1', help='layer for grad_cam - last_block, conv_head or bn1', action='store')

    opt = parser.parse_known_args()[0]

    if opt.input_folder[-1] != '/':
        opt.input_folder += '/'

    run(opt)
