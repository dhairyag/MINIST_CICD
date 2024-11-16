import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 conv1_channels=16, 
                 conv2_channels=32, 
                 fc1_size=128, 
                 num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1)
        
        # Calculate the size after convolutions and pooling
        self.flat_size = conv2_channels * 7 * 7
        
        self.fc1 = nn.Linear(self.flat_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, self.flat_size)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x