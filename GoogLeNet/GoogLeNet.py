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


# Convolutional Block 
        
class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()

        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))        

    
# Inception Block
    
class InceptionBlock(nn.Module):
    
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(InceptionBlock, self).__init__()
        
        self.branch1 = ConvBlock(in_channels=in_channels, out_channels=out_1x1, kernel_size=(1, 1))
        
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=red_3x3, kernel_size=(1, 1)),
            ConvBlock(in_channels=red_3x3, out_channels=out_3x3, kernel_size=(3, 3), padding=(1, 1))) 
        
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=red_5x5, kernel_size=(1, 1)),
            ConvBlock(in_channels=red_5x5, out_channels=out_5x5, kernel_size=(5, 5), padding=(2, 2)))
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ConvBlock(in_channels=in_channels, out_channels=out_1x1pool, kernel_size=(1, 1)))
            
    def forward(self, x):
        x = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)
        return x

    
'''
Model Class
'''

class GoogLeNet(ClassificationBase):
    '''
    Accepts Only RGB Images Of Height, Width: (224, 224)
    
    Input Size : 3, 224, 224
    Output Size : output_classes
    '''
    def __init__(self, in_channels=3, output_classes=10):
        super(ClassificationBase, self).__init__()
        
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=64, kernel_size=(7, 7), padding=(3, 3), stride=(2, 2))
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))

        self.conv2 = ConvBlock(in_channels=64, out_channels=192, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))
        
        # Starting With Inception Blocks
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        
        self.pool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 114, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        
        self.avgpool = nn.AvgPool2d(kernel_size=(7, 7), stride=(1, 1))
        
        self.dropout = nn.Dropout2d()
        
        self.l1 = nn.Linear(1024, 1000)
        self.l2 = nn.Linear(1000, output_classes)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        
        x = self.inception3a(x)
        x = self.inception3b(x)

        x = self.pool4(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        
        x = self.pool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.avgpool(x)
        x = self.dropout(x)
        
        x = x.reshape(x.shape[0], 1024)
        
        x = self.l1(x)
        x = self.l2(x)
        
        return x