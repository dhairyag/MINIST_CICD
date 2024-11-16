import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.network import SimpleCNN
from datetime import datetime
import os
from utils.augmentation import MNISTAugmentation

def train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up augmentations
    augmentation = MNISTAugmentation()
    
    # Load MNIST dataset with augmentations
    train_dataset = datasets.MNIST(
        './data', 
        train=True, 
        download=True, 
        transform=None  # No transform here as we'll create visualizations
    )
    
    # Create augmentation visualizations
    augmentation.visualize_augmentations(train_dataset)
    
    # Now load the actual training dataset with augmentations
    train_dataset = datasets.MNIST(
        './data', 
        train=True, 
        download=True, 
        transform=augmentation.train_transform
    )
    
    test_dataset = datasets.MNIST(
        './data', 
        train=False, 
        transform=augmentation.test_transform
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for 1 epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), f'models/model_{timestamp}.pth')
    
if __name__ == "__main__":
    train() 