from code.data_prep import get_train_validation_data
from code.validation import MagicValidator
from code.dataset_generator import HPACellDataset
from code.utils import get_labels_to_indices_map

import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet

import random
import cv2
import argparse
import os

LOSS_ITER_PRINT = 1000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_efficient_net_b0(state_dict_path):
    net = EfficientNet.from_name('efficientnet-b0', in_channels=4, num_classes=18, image_size=512)
    net._fc = nn.Sequential(nn.Linear(1280, 18), nn.Sigmoid())

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(state_dict_path))
        net.to(device)
    else:
        net.load_state_dict(torch.load(state_dict_path ,map_location=device))

    net.eval()
    
    return net

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class RandomResize(object):

    def __call__(self, image):

        if random.random() < 0.4:
            
            SIZE = random.randint(256, 384)
            image = cv2.resize(image, (SIZE, SIZE))

        return image

class RandomResizeAndPad(object):

    def __call__(self, image):

        if random.random() < 0.3:
            
            SIZE = random.randint(256, 384)
            image = cv2.resize(image, (SIZE, SIZE))
        
        if random.random() < 0.3:
            PAD = [random.randint(0, 200),random.randint(0, 200),random.randint(0, 200),random.randint(0, 200)] 
            image = cv2.copyMakeBorder(image,PAD[0],PAD[1],PAD[2],PAD[3],cv2.BORDER_CONSTANT)

        return image

def update_validation_metadata(train_metadata, validation_metadata, sample_size, label, validation_multipler):
    image_metadata = validation_metadata

    image_metadata = [x for x in image_metadata if label in x.image_labels]

    if label == 6:
        extra_data = [x for x in image_metadata if x.relabel.get(6, 0) == 1.0]
        image_metadata.extend(extra_data)
    
    if label in [11, 15]:
        for x in train_metadata:
            if label in x.image_labels and x.relabel.get(label, 0) >= 0.9:
                image_metadata.append(x)

    extra_data = []

    for i in range(validation_multipler):
        for x in image_metadata:
            if label in x.image_labels and label in x.metric_labels:
                value = x.relabel.get(label, 1.1)
                if value in [0.01, 0.51, 0.99, 1.1]:
                    continue
                
                extra_data.append(x)
    
    if label == 17:
        random.shuffle(image_metadata)
        image_metadata = image_metadata[:len(image_metadata) // 4]

    image_metadata.extend(extra_data)

    if label == 11:
        image_metadata.extend(image_metadata)
        image_metadata.extend(image_metadata)

    print(label, len(image_metadata))
    return image_metadata
    

class ValidationSampler():
    def __init__(self, train_metadata, validation_metadata, sample_size, label, validation_multipler):
        self.image_metadata = validation_metadata
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

def train(net, train_loader, criterion, optimizer, scheduler = None, name='unnamed', save_dir='finetuned/'):
    
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        net.train()

        inputs, labels, indices = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss

        if (i+1) % LOSS_ITER_PRINT == 0:
            print('[%d,%d] l: %f' % (epoch + 1, i + 1, running_loss / LOSS_ITER_PRINT))
            running_loss = 0.0

    torch.save(net.state_dict(), save_dir + name)

def finetune_me(model_location, batch_size = 32, criterion = 'focal', save_dir = 'finetuned/', random_resize = False, random_resize_and_pad = False):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_metadata, validation_metadata = get_train_validation_data()

    validation_scalers = {0:0, 1:2, 2:3, 3:2, 4:5, 5:5, 6:4, 7:7, 8:2, 9:2, 10:3, 11:0, 12:5, 13:0, 14:0, 15:4, 16:0, 17:7, 18:0}
    tot = 0

    all_metadata = []
    for label in range(19):
        vs = update_validation_metadata(train_metadata, validation_metadata, 20000, label, validation_multipler=validation_scalers[label])
        all_metadata.extend(vs)

    random.seed()
    random.shuffle(all_metadata)
    print("LEN ALL:", len(all_metadata))
    active_labels = [x for x in range(18)]
    label_map = get_labels_to_indices_map(active_labels)

    if random_resize_and_pad:
        finetune_transforms = transforms.Compose([RandomResizeAndPad(), transforms.ToTensor(), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation((-90,90)), transforms.Resize((512,512))])
    elif random_resize:
        finetune_transforms = transforms.Compose([RandomResize(), transforms.ToTensor(), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation((-90,90)), transforms.Resize((512,512))])
    else:
        finetune_transforms = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation((-90,90))])

    dataset = HPACellDataset(all_metadata, label_map, finetune_transforms)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=16)

    if criterion == 'focal':
        criterion = FocalLoss(reduce=True)
    elif criterion == 'bce':
        criterion = nn.BCELoss()
    else:
        raise Exception('Unsupported loss!')

    net = get_efficient_net_b0(model_location)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-5)
    name = model_location.split('/')[-1]
    train(net, train_loader, criterion, optimizer, scheduler = None, name=name, save_dir=save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', help='batch size', action='store', type=int)
    parser.add_argument('-c', '--criterion', help='criterion - bce or focal', action='store')
    parser.add_argument('-l', '--location', help='model location', action='store')
    parser.add_argument('-s', '--save_dir', help='directory for saving models', action='store')
    
    parser.add_argument('-r', '--random_resize', help='random resize', action='store_true')
    parser.add_argument('-p', '--random_resize_and_pad', help='random resize and pad', action='store_true')

    args = vars(parser.parse_args())
    
    batch_size = 32
    if args['batch_size']:
        batch_size = args['batch_size']

    criterion = 'focal'
    if args['criterion']:
        criterion = args['criterion']
    
    if args['location']:
        model_location = args['location']
    else:
        raise Exception('need model location')

    save_dir = 'finetuned/'
    if args['save_dir']:
        save_dir = args['save_dir'] + '/'

    finetune_me(model_location, batch_size, criterion, save_dir, args['random_resize'], args['random_resize_and_pad'])