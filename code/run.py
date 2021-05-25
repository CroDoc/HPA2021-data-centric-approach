import torch
import pickle

from code.dataset_generator import ValidationDataLoaderGenerator, TrainImbalanceDataLoaderGenerator
from code.validation import MagicValidator, ModelSaver
from code.networks import CustomResNet18, CustomResNet34, CustomResNet50, EfficientNetWrapper, EfficientNetWrapper4
from code.utils import train_validation_print
from code.data_prep import get_train_validation_data
import torch.nn as nn
import random
from code.utils import train_count

class Trainer():
    def __init__(self, active_labels, print_me = True, all_together = True):

        self.active_labels = active_labels

        self.all_together = all_together

        self.train_metadata, self.validation_metadata = get_train_validation_data()

        self.print_me = print_me

        if print_me:
            train_validation_print(self.train_metadata, self.validation_metadata)

        self.epochs = 20
        self.max_steps = 100000000000
        self.LOSS_ITER_PRINT = 1000
        self.VALIDATION_ITER_PRINT = 100000000000
        self.batch_size = 64
        # sample size per class - negatives treated as separate class
        self.sample_size = 20000
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # criterion with no reduction
        self.criterion = nn.BCELoss(reduction='none')

        self.model_saver = None

        self.train_loader_generator = None
        self.validator = None
        self.network_wrapper = None

        self.optimizer = None
        self.scheduler = None

        self.train_transforms = None
        
    # call when someting important changed
    def setup(self):
        if self.network_wrapper is None:
            self.network_wrapper = EfficientNetWrapper(self.active_labels, use_cuda=True, network = None, input_channels = 4)
        else:
            self.network_wrapper.set_output_layer(self.active_labels)

        #self.optimizer = torch.optim.AdamW(self.network_wrapper.network.parameters(), lr=7e-4, weight_decay=1e-1)
        self.optimizer = torch.optim.Adam(self.network_wrapper.network.parameters(), lr=3e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=15, eta_min=1e-5)
        #self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=1e-7, max_lr=1e-4, step_size_up=1000, cycle_momentum=False)#, mode='triangular2')
        
        if not self.model_saver:
            self.model_saver = ModelSaver(0.0, 'last_network', bigger = True)
            
        validation_loader_generator = ValidationDataLoaderGenerator(self.validation_metadata, self.batch_size)
        validation_loader = validation_loader_generator.generate_data_loader(self.active_labels)
        dataset = validation_loader_generator.last_dataset

        self.validator = MagicValidator(dataset, validation_loader, self.criterion, self.device)
        self.train_loader_generator = TrainImbalanceDataLoaderGenerator(self.train_metadata, self.active_labels, self.sample_size, self.batch_size, image_transforms = self.train_transforms, all_together = self.all_together)

    def check_for_none(self):
        if self.active_labels is None:
            raise Exception('No active_labels!')
        
        if self.train_loader_generator is None:
            raise Exception('No train loader generator!')

        if self.validator is None:
            raise Exception('No validator!')

        if self.model_saver is None:
            raise Exception('No model saver!')

        if self.network_wrapper is None:
            raise Exception('No network wrapper!')

        if self.criterion is None:
            raise Exception('No criterion!')

        if self.device is None:
            raise Exception('No device!')

    def train(self):

        if self.print_me:
            print('--- FINAL ---')
            train_validation_print(self.train_metadata, self.validation_metadata)

        self.check_for_none()

        steps = 0
        net = self.network_wrapper.network

        for epoch in range(self.epochs):

            positive_scores = {}

            for label in self.active_labels:
                positive_scores[label] = {}
            
            train_loader = self.train_loader_generator.generate_next_data_loader()
            dataset = self.train_loader_generator.last_dataset

            print("DATASET", len(dataset))

            running_loss = 0.0

            print("LEARNING RATE: ", self.scheduler.get_last_lr())

            for i, data in enumerate(train_loader, 0):
                net.train()

                inputs, labels, indices = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                steps += labels.size(0)

                self.optimizer.zero_grad()
                outputs = net(inputs)

                loss = self.criterion(outputs, labels).mean(dim=1)

                loss_distance = loss

                loss = loss.mean()
                loss.backward()

                loss = loss.to('cpu')
                loss_distance = loss_distance.to('cpu')

                for pos in range(len(indices)):
                    index = indices[pos]
                    metadata = dataset.image_metadata[index.item()]
                    metadata.loss = loss_distance[pos].item()
                    
                    for label in metadata.image_labels:
                        if label in self.active_labels:
                            positive_scores[label][metadata.image_name] = metadata.loss

                self.optimizer.step()

                running_loss += loss

                if (i+1) % self.LOSS_ITER_PRINT == 0:
                    print('[%d,%d] l: %f' % (epoch, i + 1, running_loss / self.LOSS_ITER_PRINT))
                    running_loss = 0.0

                """
                #used for more frequent validation

                if (i+1) % self.VALIDATION_ITER_PRINT == 0:
                    validation_loss, classic_loss = self.validator.validate(net)
                    self.model_saver.save_model_if_better(self.network_wrapper, validation_loss)
                    self.classic_model_saver.save_model_if_better(self.network_wrapper, classic_loss)
                """

                if steps > self.max_steps:
                    return
            
            for label in self.active_labels:
                vals = positive_scores[label].values()
                score = 0

                if len(vals):
                    score = sum(vals) / len(vals)

            if self.scheduler:
                self.scheduler.step()

            #if epoch <= 10:
                #continue

            print('epoch', epoch, 'loss:', end=' ')
            validation_loss, classic_loss = self.validator.validate(net)
            print(validation_loss, classic_loss)

            self.model_saver.save_model(self.network_wrapper, validation_loss, epoch)