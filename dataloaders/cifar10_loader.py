import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_cifar10_dataloaders(batch_size, data_dir='./data', val_split=0.1):
    # Updated training transformations
    transform_train = transforms.Compose([
        transforms.Resize(256),                # Resize to 256x256 first
        transforms.RandomCrop(224),            # Crop to 224x224 with augmentation
        transforms.RandomHorizontalFlip(),     # Keep data augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Updated testing/validation transformations
    transform_test = transforms.Compose([
        transforms.Resize(224),                # Resize directly to 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load datasets
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    
    # -----------------------------------------------------------------------------------
    # TODO:
    SCALE_FACTOR = 0.1
    print(f"Scaling factor: {SCALE_FACTOR}")

    print(f"Original trainset size: {len(trainset)}")
    print(f"Original testset size: {len(testset)}")
    
    scaled_trainset_size = int(len(trainset) * SCALE_FACTOR)
    scaled_testset_size = int(len(testset) * SCALE_FACTOR)
    print(f"Scaled trainset size: {scaled_trainset_size}")
    print(f"Scaled testset size: {scaled_testset_size}")
    
    trainset, _ = torch.utils.data.random_split(trainset, [scaled_trainset_size, len(trainset) - scaled_trainset_size])
    testset, _ = torch.utils.data.random_split(testset, [scaled_testset_size, len(testset) - scaled_testset_size])
    # -----------------------------------------------------------------------------------
    
    # Split trainset into training and validation sets
    val_size = int(len(trainset) * val_split)
    train_size = len(trainset) - val_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])
    
    # Create dataloaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Final trainset size: {len(trainset)}")
    print(f"Final valset size: {len(valset)}")
    print(f"Final testset size: {len(testset)}")
    
    return trainloader, valloader, testloader