import csv
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms
import numpy as np
import pickle
from code.utils import get_labels_to_indices_map

tsv_file = 'data/kaggle_2021.tsv'
csv_file = 'data/train.csv'

LABEL_COUNT = 19

def get_cam_loader(input_folder, batch_size, num_workers):
    cam_cells = sorted(set(x.rsplit('_', 1)[0] for x in os.listdir(input_folder)))
    cam_metadata = []

    for cam_cell in cam_cells:
        cam_metadata.append(ImageMetadata(cam_cell, input_folder, image_labels = None, metric_labels = None))

    dataset = HPACellDataset(cam_metadata, label_map = None, transforms = transforms.Compose([transforms.ToTensor()]), is_test = True)
    cam_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return cam_loader, dataset

def get_test_loader(test_dir, batch_size = 32, num_workers = 16):
    image_id_to_image_name = generate_image_id_to_image_names_dictionary(test_dir)

    test_images = [row[0] for row in csv.reader(open('data/sample_submission.csv', 'r')) if row[0] != 'ID']

    test_metadata = []

    for test_image in test_images:
        for test_cell in image_id_to_image_name[test_image]:
            test_metadata.append(ImageMetadata(test_cell, test_dir, image_labels = None, metric_labels = None))

    print("TM:", len(test_metadata))
    dataset = HPACellDataset(test_metadata, label_map = None, transforms = transforms.Compose([transforms.ToTensor()]), is_test = True)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_loader, dataset

def parse_label_set(classes):
    if not classes:
        return []

    return set(map(int, classes.split('|')))

class ImageMetadata():
    def __init__(self, image_name, data_directory, image_labels, metric_labels = None):
        self.image_name = image_name
        self.image_path = data_directory + image_name
        self.image_labels = image_labels

        # used for validation
        if metric_labels is None:
            self.metric_labels = set()
        else:
            self.metric_labels = metric_labels

        self.loss = 0

        self.relabel = {}

    def __repr__(self):
        return self.image_path + ' - ' + str(self.image_labels)

def generate_image_id_to_label_dictionary():

    label_dictionary = {row[0].split('/')[-1] : parse_label_set(row[4]) for row in csv.reader(open(tsv_file, 'r')) if row[0] != 'Image' and row[3] == 'False'}
    label_dictionary.update({row[0] : parse_label_set(row[1]) for row in csv.reader(open(csv_file, 'r')) if row[0] != 'ID'})

    return label_dictionary

def generate_image_name_to_label_dictionary(data_directory):
    image_id_to_label_dictionary = generate_image_id_to_label_dictionary()
    image_names = set(image.rsplit('_',1)[0] for image in os.listdir(data_directory))

    image_name_to_label_dictionary = {image_name : image_id_to_label_dictionary[image_name.rsplit('_', 1)[0]] for image_name in image_names}

    return image_name_to_label_dictionary

def generate_image_id_to_image_names_dictionary(data_directory):

    image_id_to_image_names_dictionary = {}

    image_names = set(image.rsplit('_',1)[0] for image in os.listdir(data_directory))

    for image_name in image_names:
        image_id = image_name.rsplit('_', 1)[0]

        image_id_list = image_id_to_image_names_dictionary.get(image_id, [])
        image_id_list.append(image_name)
        image_id_to_image_names_dictionary[image_id] = image_id_list

    return image_id_to_image_names_dictionary

def generate_image_metadata(image_id_list, image_id_to_image_names_dictionary, image_name_to_label_dictionary, data_directory):
    image_metadata = []

    missing = 0

    for image_id in image_id_list:
        if not image_id in image_id_to_image_names_dictionary:
            missing += 1
            continue
        for image_name in image_id_to_image_names_dictionary[image_id]:
            labels = image_name_to_label_dictionary[image_name]
            image_metadata.append(ImageMetadata(image_name, data_directory, labels.copy(), None))

    print("MISSING FROM DATASET:", missing)

    return image_metadata

def get_public_data(image_id_to_image_names_dictionary, image_name_to_label_dictionary, data_directory):
    image_id_list = [row[0].split('/')[-1] for row in csv.reader(open(tsv_file, 'r')) if row[0] != 'Image' and row[3] == 'False']
    return generate_image_metadata(image_id_list, image_id_to_image_names_dictionary, image_name_to_label_dictionary, data_directory)

def get_train_data(image_id_to_image_names_dictionary, image_name_to_label_dictionary, data_directory):
    image_id_list = [row[0] for row in csv.reader(open(csv_file, 'r')) if row[0] != 'ID']
    return generate_image_metadata(image_id_list, image_id_to_image_names_dictionary, image_name_to_label_dictionary, data_directory)

class TrainValidationSplitter():
    def __init__(self, percent, active_labels, random_seed):
        self.percent = percent
        self.active_labels = set(active_labels)
        self.random_seed = random_seed

    def split(self, image_metadata):
        random.seed(self.random_seed)
        random.shuffle(image_metadata)

        split = len(image_metadata) * self.percent // 100
        validation_metadata = image_metadata[:split]
        train_metadata = image_metadata[split:]

        return train_metadata, validation_metadata

class HPACellDataset(Dataset):

    def __init__(self, image_metadata, label_map, transforms, is_test=False, channels_num=4):
        self.image_metadata = image_metadata
        self.label_map = label_map
        self.transforms = transforms
        self.is_test = is_test

        if channels_num == 3:
            self.channels = ['_red', '_green', '_blue']
        elif channels_num == 4:
            self.channels = ['_red', '_green', '_blue', '_yellow']
        else:
            raise Exception('unsupported number of channels')

    def __len__(self):
        return len(self.image_metadata)

    def generate_label_tensor(self, image_labels, relabel):

        indices = [self.label_map[label] for label in image_labels if label in self.label_map]

        label_tensor = torch.zeros(len(self.label_map))
        label_tensor[indices] = 1.0

        for label in relabel:
            if label in self.label_map:
                index = self.label_map[label]

                label_tensor[index] = relabel[label]

        return label_tensor

    def __getitem__(self, index):

        image_metadata = self.image_metadata[index]

        channels = []

        for channel in self.channels:

            image = cv2.imread(image_metadata.image_path + channel + '.png', cv2.IMREAD_UNCHANGED)
            image = image.astype(np.float32) / 65535.0
            channels.append(image)

        image = np.dstack(channels)

        if self.is_test:
            return self.transforms(image), index

        image = self.transforms(image)

        return image, self.generate_label_tensor(image_metadata.image_labels, image_metadata.relabel), index

class FairSampler():
    def __init__(self, image_metadata, sample_size):
        self.image_metadata = image_metadata
        self.sample_size = sample_size
        self.index = 0

    def sample(self):

        current_sample = []
        missing = self.sample_size

        while missing > 0:

            if self.index == 0:
                random.seed()
                random.shuffle(self.image_metadata)

            candidate = self.image_metadata[self.index]
            label_cnt = len(candidate.image_labels)
            if label_cnt == 0:
                label_cnt = 1

            if random.random() < (1.0 / label_cnt) ** 0.5:
                current_sample.append(candidate)
                missing -= 1

            self.index += 1
            if self.index == len(self.image_metadata):
                self.index = 0

        return current_sample

class ClassicSampler(FairSampler):
    def __init__(self, image_metadata, sample_size):
        self.image_metadata = image_metadata
        self.sample_size = sample_size
        self.index = 0

class AntiLabelSampler(FairSampler):
    def __init__(self, image_metadata, sample_size, antilabel):
        self.image_metadata = image_metadata
        self.sample_size = sample_size
        self.index = 0

        self.image_metadata.extend([x for x in self.image_metadata if not antilabel in x.image_labels])
        self.image_metadata.extend([x for x in self.image_metadata if not antilabel in x.image_labels])

# 25% negatives 25% similar negatives 50% mostly good positives
class MitoticSpindleSampler():
    def __init__(self, image_metadata, sample_size, label=11):
        self.image_metadata = image_metadata
        self.sample_size = sample_size
        self.index = 0

        best =[x for x in self.image_metadata if x.relabel[label] >= 0.99]
        self.image_metadata.extend(best)
        self.image_metadata.extend(best)

        positives = [x for x in self.image_metadata if x.relabel[label] > 0.0]

        for i in range(40):
            self.image_metadata.extend(positives)

    def sample(self):

        current_sample = []
        missing = self.sample_size

        while missing > 0:

            if self.index == 0:
                random.seed()
                random.shuffle(self.image_metadata)

            current_sample.extend(self.image_metadata[self.index : self.index + missing])

            self.index += missing
            if self.index >= len(self.image_metadata):
                self.index = 0

            missing = self.sample_size - len(current_sample)

        return current_sample

class UpSampleSampler(FairSampler):
    def __init__(self, image_metadata, sample_size, label):
        self.image_metadata = image_metadata
        self.sample_size = sample_size
        self.index = 0

        #self.image_metadata.extend([x for x in self.image_metadata if x.relabel.get(label, 0) >= 0.5])
        #self.image_metadata.extend([x for x in self.image_metadata if x.relabel.get(label, 0) >= 0.7])
        self.image_metadata.extend([x for x in self.image_metadata if x.relabel.get(label, 0) >= 0.9])

        if label == 12:
            self.image_metadata.extend([x for x in self.image_metadata if x.relabel.get(label, 0) >= 0.7])
            #self.image_metadata.extend([x for x in self.image_metadata if x.relabel.get(label, 0) >= 0.7])
            self.image_metadata.extend([x for x in self.image_metadata if x.relabel.get(label, 0) >= 0.9])
            self.image_metadata.extend([x for x in self.image_metadata if x.relabel.get(label, 0) >= 0.9])

            for x in self.image_metadata:
                if x.relabel.get(12, 0) == 0.69:
                    x.relabel[12] = 1.0

        #self.image_metadata.extend([x for x in self.image_metadata if x.relabel.get(label, 0) >= 0.99])

class ClassicSamplerWithLBalancer():
    def __init__(self, image_metadata, sample_size):
        self.image_metadata = image_metadata
        self.sample_size = sample_size
        self.index = 0

        print("LEN0:", len(self.image_metadata))

        extra_data = [x for x in self.image_metadata if x.relabel.get(11,0) == 1.0]
        extra_data.extend([x for x in self.image_metadata if x.relabel.get(11,0) == 1.0])
        extra_data.extend([x for x in self.image_metadata if x.relabel.get(15,0) == 1.0])

        extra_data.extend(x for x in image_metadata if x.metric_labels)
        extra_data.extend(x for x in image_metadata if x.metric_labels)
        extra_data.extend(x for x in image_metadata if x.metric_labels)

        self.image_metadata.extend(extra_data)
        print("LEN:", len(self.image_metadata))
        self.sample_size = len(self.image_metadata)

    def sample(self):

        current_sample = []
        missing = self.sample_size

        while missing > 0:

            if self.index == 0:
                random.seed()
                random.shuffle(self.image_metadata)

            current_sample.extend(self.image_metadata[self.index : self.index + missing])

            self.index += missing
            if self.index >= len(self.image_metadata):
                self.index = 0

            missing = self.sample_size - len(current_sample)

        return current_sample

class OverfitUnderfitSampler():
    def __init__(self, image_metadata, sample_size, swap_best = 8, swap_worst=12, swap_random=5, reset_chance = 0.2):
        self.image_metadata = image_metadata
        self.sample_size = sample_size

        self.swap_best = swap_best
        self.swap_worst = swap_worst
        self.swap_random = swap_random
        self.reset_chance = reset_chance

        self.classic_sampler = ClassicSampler(self.image_metadata, self.sample_size)
        self.selected = self.classic_sampler.sample()

    def sample(self):

        if random.random() < self.reset_chance:
            self.selected = self.classic_sampler.sample()
            return self.selected

        swap_best_count = self.sample_size * self.swap_best // 100
        swap_worst_count = self.sample_size * self.swap_worst // 100
        swap_random_count = self.sample_size * self.swap_random // 100

        # best first
        self.selected.sort(key = lambda x : x.loss)
        self.selected = self.selected[swap_best_count:-swap_worst_count]
        random.seed()
        random.shuffle(self.selected)
        self.selected = self.selected[:swap_random_count]

        while len(self.selected) < self.sample_size:
            rand_val = random.randint(0, len(self.image_metadata)-1)
            self.selected.append(self.image_metadata[rand_val])

        return self.selected

class ParetoSampler():
    def __init__(self, image_metadata, sample_size, swap_best = 20, swap_random=5):

        self.swap_best = swap_best
        self.swap_random = swap_random

        if sample_size > len(image_metadata):
            raise Exception('Not enough images for ParetoSampler!')

        self.sample_size = sample_size

        random.seed()
        random.shuffle(image_metadata)

        self.selected = image_metadata[:self.sample_size]
        self.pool = image_metadata[self.sample_size:]

    def move_images(self, move_from, move_to, size, shuffle_from=False):

        if shuffle_from:
            random.seed()
            random.shuffle(move_from)

        move_to.extend(move_from[:size])
        move_from = move_from[size:]

        return move_from, move_to

    def sample(self):

        swap_best_count = self.sample_size * self.swap_best // 100
        swap_random_count = self.sample_size * self.swap_random // 100

        # best first
        self.selected.sort(key = lambda x : x.loss)
        self.selected, self.pool = self.move_images(self.selected, self.pool, swap_best_count, shuffle_from=False)
        self.selected, self.pool = self.move_images(self.selected, self.pool, swap_random_count, shuffle_from=True)
        self.pool, self.selected = self.move_images(self.pool, self.selected, swap_best_count + swap_random_count, shuffle_from=True)

        if len(self.selected) != self.sample_size:
            raise Exception('ParetoSampler sample size invalid!')

        return self.selected

class TrainImbalanceDataLoaderGenerator():
    def __init__(self, train_image_metadata, active_labels, sample_size, batch_size, image_transforms=None, num_workers=16, all_together = True):
        self.train_image_metadata = train_image_metadata

        #print("TRAIN IMAGES: " + str(len(self.train_image_metadata)))

        self.sample_size = sample_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.last_dataset = None
        self.all_together = all_together

        if image_transforms is None:
            self.transforms = transforms.Compose([transforms.ToTensor()])
        else:
            self.transforms = image_transforms

        self.set_active_labels_and_reset(active_labels)

    def set_active_labels_and_reset(self, active_labels):
        self.active_labels = set(active_labels)
        self.reset()

    def reset(self):

        self.last_dataset = None
        self.label_map = get_labels_to_indices_map(self.active_labels)

        print(self.label_map)

        if self.all_together:
            self.samplers = [ClassicSamplerWithLBalancer(self.train_image_metadata, self.sample_size)]
        else:
            images_per_label_map = {label : [] for label in self.active_labels}
            negative_list = []

            for image_metadata in self.train_image_metadata:
                is_positive = False

                for label in image_metadata.image_labels:
                    if label in images_per_label_map:
                        label_list = images_per_label_map[label]
                        label_list.append(image_metadata)
                        is_positive = True

                if not is_positive:
                    negative_list.append(image_metadata)

            self.samplers = []

            for label in self.active_labels:
                if label in [1, 3, 4, 6, 12]:
                    self.samplers.append(UpSampleSampler(images_per_label_map[label], self.sample_size // 2, label))
                elif label in [8, 9, 10]:
                    self.samplers.append(UpSampleSampler(images_per_label_map[label], self.sample_size // 2, label))
                elif label in [13, 17]:
                    self.samplers.append(UpSampleSampler(images_per_label_map[label], self.sample_size * 1, label))
                elif label in [0, 16]:
                    self.samplers.append(UpSampleSampler(images_per_label_map[label], self.sample_size * 2, label))
                elif label in [11]:
                    self.samplers.append(MitoticSpindleSampler(images_per_label_map[label], self.sample_size // 3))
                elif label in [15]:
                    self.samplers.append(UpSampleSampler(images_per_label_map[label], self.sample_size // 2, label))
                else:
                    self.samplers.append(UpSampleSampler(images_per_label_map[label], self.sample_size, label))
            self.samplers.append(ClassicSampler(negative_list, self.sample_size))

    def generate_next_data_loader(self):

        image_metadata = []

        for sampler in self.samplers:
            image_metadata.extend(sampler.sample())

        random.seed()
        random.shuffle(image_metadata)

        dataset = HPACellDataset(image_metadata, self.label_map, self.transforms)
        self.last_dataset = dataset
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

class ValidationDataLoaderGenerator():
    def __init__(self, validation_image_metadata, batch_size, image_transforms=None, num_workers=16):
        self.validation_image_metadata = validation_image_metadata
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.last_dataset = None

        if image_transforms is None:
            self.transforms = transforms.Compose([transforms.ToTensor()])
        else:
            self.transforms = image_transforms

    def generate_data_loader(self, label_ids):
        label_map = get_labels_to_indices_map(label_ids)

        dataset = HPACellDataset(self.validation_image_metadata, label_map, self.transforms)
        self.last_dataset = dataset

        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

def generate_train_public_data(train_data_directory, extra_data_directory = None):

    image_name_to_label_dictionary = generate_image_name_to_label_dictionary(train_data_directory)
    image_id_to_image_names_dictionary = generate_image_id_to_image_names_dictionary(train_data_directory)

    if extra_data_directory:
        image_name_to_label_dictionary.update(generate_image_name_to_label_dictionary(extra_data_directory))
        image_id_to_image_names_dictionary.update(generate_image_id_to_image_names_dictionary(extra_data_directory))
    else:
        extra_data_directory = train_data_directory

    train_image_metadata = get_train_data(image_id_to_image_names_dictionary, image_name_to_label_dictionary, train_data_directory)
    public_image_metadata = get_public_data(image_id_to_image_names_dictionary, image_name_to_label_dictionary, extra_data_directory)

    return train_image_metadata, public_image_metadata
