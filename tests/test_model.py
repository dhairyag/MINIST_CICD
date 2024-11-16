import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from model.network import SimpleCNN
from torchvision import datasets, transforms
import glob
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_parameters():
    model = SimpleCNN()
    param_count = count_parameters(model)
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 25000"

def test_input_output_shape():
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"

def test_model_accuracy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    
    # Load the latest model with weights_only=True for security
    import glob
    import os
    model_files = glob.glob('models/model_*.pth')
    latest_model = max(model_files, key=os.path.getctime)
    model.load_state_dict(torch.load(latest_model, weights_only=True))
    
    # Test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 95, f"Accuracy is {accuracy}%, should be > 95%" 

def test_model_confidence():
    """Test if model predictions have reasonable confidence scores"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    
    # Load latest model
    model_files = glob.glob('models/model_*.pth')
    latest_model = max(model_files, key=os.path.getctime)
    model.load_state_dict(torch.load(latest_model, weights_only=True))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)
    
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data.to(device)
        outputs = torch.softmax(model(data), dim=1)
        max_probs, _ = torch.max(outputs, dim=1)
        
        # Check if average confidence of predictions is reasonable
        avg_confidence = max_probs.mean().item()
        assert 0.7 <= avg_confidence <= 1.0, \
            f"Average confidence {avg_confidence:.2f} is outside reasonable range [0.7, 1.0]"

def test_class_distribution():
    """Test if model predictions are reasonably distributed across all digits"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    
    model_files = glob.glob('models/model_*.pth')
    latest_model = max(model_files, key=os.path.getctime)
    model.load_state_dict(torch.load(latest_model, weights_only=True))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    model.eval()
    predictions = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
    
    # Count predictions for each class
    class_counts = torch.bincount(torch.tensor(predictions))
    class_distribution = class_counts / len(predictions)
    
    # Check if each class has at least 5% of predictions
    min_class_ratio = class_distribution.min().item()
    assert min_class_ratio >= 0.05, \
        f"Minimum class ratio {min_class_ratio:.3f} is below 0.05, suggesting poor distribution"

def test_adversarial_robustness():
    """Test basic robustness to input noise"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    
    model_files = glob.glob('models/model_*.pth')
    latest_model = max(model_files, key=os.path.getctime)
    model.load_state_dict(torch.load(latest_model, weights_only=True))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)
    
    model.eval()
    correct_original = 0
    correct_noisy = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Original prediction
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            correct_original += (predicted == target).sum().item()
            
            # Add Gaussian noise
            noise = torch.randn_like(data) * 0.1
            noisy_data = data + noise
            
            # Prediction with noise
            noisy_outputs = model(noisy_data)
            _, noisy_predicted = torch.max(noisy_outputs.data, 1)
            correct_noisy += (noisy_predicted == target).sum().item()
            
            total += target.size(0)
    
    original_accuracy = 100 * correct_original / total
    noisy_accuracy = 100 * correct_noisy / total
    accuracy_drop = original_accuracy - noisy_accuracy
    
    assert accuracy_drop < 15, \
        f"Model accuracy drops by {accuracy_drop:.1f}% with noise, should be < 15%"
