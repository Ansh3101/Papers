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


'''
Model Class
'''

class LeNet(ClassificationBase):
    '''
    Accepts Only Black & White Images Of Height, Width: (32, 32)

    Input Size : 1, 32, 32
    Output Size : output_classes
    '''
    def __init__(self, output_classes):
        super(ClassificationBase, self).__init__()
        
        self.output_classes = output_classes
        self.relu = nn.ReLU() # If Value Is Negative, Clips It To 0; No Change To Positive Values
        self.avgpool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)) # Returns Average Of Numbers / Kernel Iteration (Size Reduction)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)) # Uses Weights Vector To Change Values, Outputs Multiple Channels
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.l1 = nn.Linear(120, 84) # Sum Of Matrix Multiplications (Input . Weight) Vectors
        self.l2 = nn.Linear(84, output_classes)
        
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.avgpool(x)
        x = self.relu(self.conv2(x))
        x = self.avgpool(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1) # To ReShape (batch_size, 120, 1, 1) into (batch_size, 120)
        x = self.l1(x)
        x = self.l2(x)
        return x