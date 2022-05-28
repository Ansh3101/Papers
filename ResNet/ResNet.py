import torch
from torch import nn
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2
from glob import glob
import os
import tqdm

device='cpu'

# Training Loop Embedded Into Model

'''
Training Modules
'''

class ClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)          
        return loss, acc

    def validation_step(self, batch):
        images, labels = batch 
        images = images.to(device)
        labels = labels.to(device)
        out = self(images)                    
        loss = F.cross_entropy(out, labels)  
        acc = accuracy(out, labels)          
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()    
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

# Accuracy & Validation Functions
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)       

    
# Inception Block
    
class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.expansion = 4
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
    def forward(self, x):
        identity = x
        
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.relu(self.norm3(self.conv3(x)))
        
        if self.identity_downsample:
            identity = self.identity_downsample(identity)
            
       # x += identity
        x = self.relu(x)
        
        return x

    
'''
Model Class
'''

class ResNet(ClassificationBase):
    '''
    Accepts Only RGB Images Of Height, Width: (224, 224)
    
    Input Size : 3, 224, 224
    Output Size : output_classes
    '''
    def __init__(self, block, layers, image_channels=3, output_classes=10):
        super(ClassificationBase, self).__init__()
        
        self.relu = nn.ReLU()
        self.in_channels = 64
    
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=64, kernel_size=(7, 7), stride=2, padding=3)
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.norm1 = nn.BatchNorm2d(64)
        
        self.conv2 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.conv3 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.conv4 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.conv5 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.l1 = nn.Linear(2048, 1000)
        self.l2 = nn.Linear(1000, output_classes)
        
    def _make_layer(self, block, n_residual_blocks, out_channels, stride):
        
        identity_downsample = None
        layers = []
        if stride != 1 or self.in_channels != out_channels*4: # Whenever It's Not The Output Channels / Original Height & Width Aren't Being Reduced
            identity_sample = nn.Sequential( 
                nn.Conv2d(self.in_channels, out_channels=out_channels*4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels*4))
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels*4
        
        for i in range(n_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.norm1(self.conv1(x))
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        
        x = self.l1(x)
        x = self.l2(x)
        return x