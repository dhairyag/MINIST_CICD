import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, 
                 in_channels=1,
                 conv1_channels=8,
                 conv2_channels=16,
                 fc1_size=24,
                 num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout2d(0.1)
        self.dropout2 = nn.Dropout(0.1)
        
        # Calculate the size after convolutions and pooling
        self.flat_size = conv2_channels * 7 * 7
        
        self.fc1 = nn.Linear(self.flat_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(conv1_channels)
        self.batchnorm2 = nn.BatchNorm2d(conv2_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = x.view(-1, self.flat_size)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x