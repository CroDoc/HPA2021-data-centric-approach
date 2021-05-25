import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import random
import cv2
import argparse

from code.run import Trainer
from code.validation import ModelSaver
from code.networks import CustomResNet18
from code.utils import get_border_and_garbage_images, relabel_image_metadata, get_latest_relabel_old, get_latest_antibody_relabel, relabel_me

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

def run(model_name = 'effnet-b0', batch_size = 32, criterion = 'focal', save_dir = 'models/', random_resize = False, random_resize_and_pad = False):

    active_labels = [x for x in range(18)]
    trainer = Trainer(all_together = False, active_labels = active_labels)

    trainer.epochs = 16
    trainer.sample_size = 20000
    trainer.LOSS_ITER_PRINT = 1000
    trainer.batch_size = batch_size

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    trainer.model_saver = ModelSaver(0.0, model_name + '-map', save_dir = save_dir, bigger=True)
    trainer.classic_model_saver = ModelSaver(200.0, model_name + '-loss', save_dir = save_dir)

    if random_resize_and_pad:
        trainer.train_transforms = transforms.Compose([RandomResizeAndPad(), transforms.ToTensor(), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation((-90,90)), transforms.Resize((512,512))])
    elif random_resize:
        trainer.train_transforms = transforms.Compose([RandomResize(), transforms.ToTensor(), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation((-90,90)), transforms.Resize((512,512))])
    else:
        trainer.train_transforms = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation((-90,90))])


    if criterion == 'focal':
        trainer.criterion = FocalLoss(reduce=False)
    elif criterion != 'bce':
        raise Exception('Unsupported loss!')

    trainer.setup()
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', help='batch size', action='store', type=int)
    parser.add_argument('-c', '--criterion', help='criterion - bce or focal', action='store')
    parser.add_argument('-n', '--name', help='model name', action='store')
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
    
    model_name = 'effet-b0'
    if args['name']:
        model_name = args['name']

    save_dir = 'models/'
    if args['save_dir']:
        save_dir = args['save_dir'] + '/'

    run(model_name, batch_size, criterion, save_dir, args['random_resize'], args['random_resize_and_pad'])