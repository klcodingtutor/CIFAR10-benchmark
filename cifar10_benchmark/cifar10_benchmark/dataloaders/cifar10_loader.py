import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def get_cifar10_dataloaders(batch_size):
    """Load CIFAR-10 dataset with data augmentation and normalization."""
    
    # Define data transformations
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Load CIFAR-10 training and test datasets
    train_dataset = datasets.CIFAR10(root=os.path.join(os.getcwd(), 'data'), train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=os.path.join(os.getcwd(), 'data'), train=False, download=True, transform=transform_test)
    
    # Create data loaders
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader