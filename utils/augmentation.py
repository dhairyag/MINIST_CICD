import torch
import torchvision.transforms as T
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np

class MNISTAugmentation:
    def __init__(self):
        # Carefully chosen augmentations that preserve digit readability
        self.train_transform = T.Compose([
            T.RandomRotation(15),  # Slight rotation (±15 degrees)
            T.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # Small translations
                scale=(0.9, 1.1),      # Slight scaling
                shear=10               # Slight shearing
            ),
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))
        ])
        
        self.test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,))
        ])

        # Transform for visualization (without normalization)
        self.viz_transform = T.Compose([
            T.RandomRotation(15),
            T.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=10
            ),
            T.ToTensor()
        ])

    def visualize_augmentations(self, dataset, num_samples=10, num_aug=4):
        """
        Visualize augmentations for given samples
        """
        # Create output directory if it doesn't exist
        os.makedirs('augmentation_samples', exist_ok=True)
        
        fig, axes = plt.subplots(num_samples, num_aug + 1, 
                                figsize=(2*(num_aug + 1), 2*num_samples))
        
        to_tensor = T.ToTensor()
        
        for i in range(num_samples):
            image, label = dataset[i]
            # Convert PIL Image to tensor for visualization
            img_tensor = to_tensor(image)
            axes[i, 0].imshow(img_tensor.squeeze(), cmap='gray')
            axes[i, 0].set_title(f'Original\nLabel: {label}')
            
            for j in range(num_aug):
                # Use viz_transform instead of train_transform to avoid normalization
                aug_image = self.viz_transform(image)
                axes[i, j+1].imshow(aug_image.squeeze(), cmap='gray')
                axes[i, j+1].set_title(f'Aug {j+1}\nLabel: {label}')
            
        for ax in axes.flat:
            ax.axis('off')
            
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'augmentation_samples/augmentations_{timestamp}.png')
        plt.close() 