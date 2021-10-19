import numpy as np
import cv2

from code.utils import get_cropped_mask

"""
# use for train dataset generating
def ratio_to_value_multiplier(ratio):
    if ratio > 0.01:
        return 1.0

    return 0.001
"""

def ratio_to_value_multiplier(ratio):
    if ratio > 0.3:
        return 1.0

    if ratio > 0.01:
        return (ratio * 3.33) ** 0.1

    return 0.001

def no_nuclei_and_border_cleaner(nuclei_mask, cell_mask, red, blue, yellow, values):
    for i in range(len(values)):
        index = i+1

        if blue[nuclei_mask == index].sum() == 0:
            values[i] *= 0.01
            continue

        single_nuclei_mask = get_cropped_mask(nuclei_mask, index)

        lr = single_nuclei_mask.shape[0]
        ud = single_nuclei_mask.shape[1]

        up = single_nuclei_mask[0].sum()
        down = single_nuclei_mask[-1].sum()
        left = single_nuclei_mask[:, -0].sum()
        right = single_nuclei_mask[:, -1].sum()

        ud_ratio = (ud - max(up, down)) / ud
        lr_ratio = (lr - max(left, right)) / lr

        values[i] *= ratio_to_value_multiplier(ud_ratio) * ratio_to_value_multiplier(lr_ratio)

    return values

def no_blue_yellow_red_cleaner(nuclei_mask, cell_mask, red, blue, yellow, values):

    red_sums = []
    blue_yellow_products = []
    valid_candidates = 0

    for i in range(len(values)):
        index = i+1

        if values[i] <= 0.01:
            red_sums.append(0)
            blue_yellow_products.append(0)
            continue

        valid_candidates += 1

        red_sum = red[cell_mask == index].sum()
        blue_sum = blue[cell_mask == index].sum()
        yellow_sum = yellow[cell_mask == index].sum()

        red_sums.append(red_sum)
        blue_yellow_products.append(blue_sum * yellow_sum)

    red_minimum = sum(red_sums) / valid_candidates * 0.077
    blue_yellow_products_minimum = sum(blue_yellow_products) / valid_candidates * 0.05

    for i in range(len(values)):
        index = i+1

        if values[i] <= 0.01:
            continue

        if red_sums[i] < red_minimum or blue_yellow_products[i] < blue_yellow_products_minimum:
            values[i] *= 0.001

    return values

class DataCleaner():

    def __init__(self, nuclei_masks_folder, cell_masks_folder, dataset_folder):
        self.cleaners = [no_nuclei_and_border_cleaner, no_blue_yellow_red_cleaner]
        self.nuclei_masks_folder = nuclei_masks_folder
        self.cell_masks_folder = cell_masks_folder
        self.dataset_folder = dataset_folder

    def load_image(self, image_path):
        image = np.array(cv2.imread(image_path, cv2.IMREAD_UNCHANGED))

        if image.dtype == np.uint8:
            image = image.astype(np.uint16)
            image *= 257

        image = image.astype(np.float32)
        image /= 65535

        return image

    def clean(self, image_id):
        nuclei_mask = np.load(self.nuclei_masks_folder + image_id + '.npy')
        cell_mask = np.load(self.nuclei_masks_folder + image_id + '.npy')
        image_path = self.dataset_folder + image_id
        red = self.load_image(image_path + '_red.png')
        blue = self.load_image(image_path + '_blue.png')
        yellow = self.load_image(image_path + '_yellow.png')

        values = [1.0] * cell_mask.max()

        for cleaner in self.cleaners:
            values = cleaner(nuclei_mask, cell_mask, red, blue, yellow, values)

        return values
