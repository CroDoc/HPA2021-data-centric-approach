import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from code.utils import get_labels_to_indices_map
from abc import ABC, abstractmethod
import pickle
import torch
from efficientnet_pytorch import EfficientNet

class NetworkWrapper():
    def __init__(self, output_labels, use_cuda=True, network = None, input_channels = 4):

        self.use_cuda = use_cuda
        self.input_channels = input_channels

        self.output_labels = self.sort_and_unique(output_labels)
        self.label_to_index = get_labels_to_indices_map(self.output_labels)

        self.output_size = len(self.output_labels)

        if network:
            self.network = network
        else:
            self.network = self.generate_new_network()
            self.set_input_layer()
            self.set_output_layer()
        
        if self.use_cuda:
            self.network.cuda()
        
        print(self.network)
    
    @abstractmethod
    def generate_new_network(self):
        pass

    def sort_and_unique(self, output_labels):
        return sorted(set(output_labels))

    @abstractmethod
    def set_output_layer(self):
        pass

    @abstractmethod
    def set_input_layer(self):
        pass

    def freeze_parameters(self, freeze):
        if freeze <= 0:
            return
        for name, param in self.network.named_parameters():
            print (name)
        for param in self.network.parameters():
            
            param.requires_grad = False

            freeze -= 1
            if freeze == 0:
                break
    
    def unfreeze_parameters(self):
        for param in self.network.parameters():
            param.requires_grad = True

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class CustomResNet50(NetworkWrapper):
    def generate_new_network(self):
        model =  models.resnet50(pretrained=True)

        #model.layer3 = Identity()
        #model.layer4 = Identity()
        
        return model
    
    def set_output_layer(self):
        self.network.fc = nn.Sequential(nn.Linear(2048, self.output_size), nn.Sigmoid())

    def set_input_layer(self):
        layer = self.network.conv1
                
        new_layer = nn.Conv2d(in_channels=self.input_channels, 
                        out_channels=layer.out_channels, 
                        kernel_size=layer.kernel_size, 
                        stride=layer.stride,
                        padding=layer.padding,
                        bias=layer.bias)

        copy_weights = 0

        new_layer.weight[:, :layer.in_channels, :, :] = layer.weight.clone()

        for i in range(self.input_channels - layer.in_channels):
            channel = layer.in_channels + i
            new_layer.weight[:, channel:channel+1, :, :] = layer.weight[:, copy_weights:copy_weights+1, : :].clone()
        new_layer.weight = nn.Parameter(new_layer.weight)

        self.network.conv1 = new_layer

class CustomResNet34(NetworkWrapper):
    def generate_new_network(self):
        model =  models.resnet34(pretrained=True)

        #model.layer3 = Identity()
        #model.layer4 = Identity()
        
        return model
    
    def set_output_layer(self):
        self.network.fc = nn.Sequential(nn.Linear(512, self.output_size), nn.Sigmoid())

    def set_input_layer(self):
        layer = self.network.conv1
                
        new_layer = nn.Conv2d(in_channels=self.input_channels, 
                        out_channels=layer.out_channels, 
                        kernel_size=layer.kernel_size, 
                        stride=layer.stride,
                        padding=layer.padding,
                        bias=layer.bias)

        copy_weights = 0

        new_layer.weight[:, :layer.in_channels, :, :] = layer.weight.clone()

        for i in range(self.input_channels - layer.in_channels):
            channel = layer.in_channels + i
            new_layer.weight[:, channel:channel+1, :, :] = layer.weight[:, copy_weights:copy_weights+1, : :].clone()
        new_layer.weight = nn.Parameter(new_layer.weight)

        self.network.conv1 = new_layer

class CustomResNet18(NetworkWrapper):
    def generate_new_network(self):
        model =  models.resnet18(pretrained=True)

        #model.layer3 = Identity()
        #model.layer4 = Identity()
        
        return model
    
    def set_output_layer(self):
        self.network.fc = nn.Sequential(nn.Linear(512, self.output_size), nn.Sigmoid())

    def set_input_layer(self):
        layer = self.network.conv1
                
        new_layer = nn.Conv2d(in_channels=self.input_channels, 
                        out_channels=layer.out_channels, 
                        kernel_size=layer.kernel_size, 
                        stride=layer.stride,
                        padding=layer.padding,
                        bias=layer.bias)

        copy_weights = 0

        new_layer.weight[:, :layer.in_channels, :, :] = layer.weight.clone()

        for i in range(self.input_channels - layer.in_channels):
            channel = layer.in_channels + i
            new_layer.weight[:, channel:channel+1, :, :] = layer.weight[:, copy_weights:copy_weights+1, : :].clone()
        new_layer.weight = nn.Parameter(new_layer.weight)

        self.network.conv1 = new_layer

class EfficientNetWrapper(NetworkWrapper):
    def generate_new_network(self, model_name='efficientnet-b0'):
        in_channels = self.input_channels
        model = EfficientNet.from_pretrained(model_name, in_channels=in_channels, num_classes=len(self.output_labels), image_size=512)
        return model
    
    def set_output_layer(self):
        self.network._fc = nn.Sequential(nn.Linear(1280, self.output_size), nn.Sigmoid())
        return

    def set_input_layer(self):
        return

class EfficientNetWrapper4(NetworkWrapper):
    def generate_new_network(self, model_name='efficientnet-b4'):
        in_channels = self.input_channels
        model = EfficientNet.from_pretrained(model_name, in_channels=in_channels, num_classes=len(self.output_labels), image_size=512)
        return model
    
    def set_output_layer(self):
        self.network._fc = nn.Sequential(nn.Linear(1792, self.output_size), nn.Sigmoid())
        return

    def set_input_layer(self):
        return

class ResNet18(NetworkWrapper):
    def generate_new_network(self, pretrained, yellow):
        model =  models.resnet18(pretrained=pretrained)
        self.pretrained = pretrained

        if yellow:
            self.add_yellow(model)
        
        return model
    
    def attach_output_layer(self, output_size):
        self.network.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(512, output_size), nn.Sigmoid())

    def add_yellow(self, model):
        new_in_channels = 4
        layer = model.conv1
                
        new_layer = nn.Conv2d(in_channels=new_in_channels, 
                        out_channels=layer.out_channels, 
                        kernel_size=layer.kernel_size, 
                        stride=layer.stride, 
                        padding=layer.padding,
                        bias=layer.bias)

        if not self.pretrained:
            model.conv1 = new_layer
            return

        copy_weights = 0

        new_layer.weight[:, :layer.in_channels, :, :] = layer.weight.clone()

        for i in range(new_in_channels - layer.in_channels):
            channel = layer.in_channels + i
            new_layer.weight[:, channel:channel+1, :, :] = layer.weight[:, copy_weights:copy_weights+1, : :].clone()
        new_layer.weight = nn.Parameter(new_layer.weight)

        model.conv1 = new_layer

class DenseNet121(NetworkWrapper):
    def generate_new_network(self, pretrained, yellow):
        model =  models.densenet121(pretrained=pretrained)

        if yellow:
            self.add_yellow(model)
        
        return model
    
    def attach_output_layer(self, output_size):
        self.network.classifier = nn.Sequential(nn.Linear(1024, output_size), nn.Sigmoid())

    def add_yellow(self, model):
        new_in_channels = 4
        layer = model.features[0]
                
        new_layer = nn.Conv2d(in_channels=new_in_channels, 
                        out_channels=layer.out_channels, 
                        kernel_size=layer.kernel_size, 
                        stride=layer.stride, 
                        padding=layer.padding,
                        bias=layer.bias)

        copy_weights = 0

        new_layer.weight[:, :layer.in_channels, :, :] = layer.weight.clone()

        for i in range(new_in_channels - layer.in_channels):
            channel = layer.in_channels + i
            new_layer.weight[:, channel:channel+1, :, :] = layer.weight[:, copy_weights:copy_weights+1, : :].clone()
        new_layer.weight = nn.Parameter(new_layer.weight)

        model.features[0] = new_layer