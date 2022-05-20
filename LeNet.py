from torch import nn

class LeNet(nn.Module):
    '''
    Accepts Only Black & White Images Of Height, Width: (32, 32)

    Input Size : 1, 32, 32
    Output Size : output_classes
    '''
    def __init__(self, output_classes):
        super(LeNet, self).__init__()
        
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