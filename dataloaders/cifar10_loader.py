import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_cifar10_dataloaders(batch_size, data_dir='./data', val_split=0.1):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    
    # -----------------------------------------------------------------------------------
    SCALE_FACTOR = 1
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
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Final trainset size: {len(trainset)}")
    print(f"Final valset size: {len(valset)}")
    print(f"Final testset size: {len(testset)}")
    
    return trainloader, valloader, testloader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(batch_size=64)
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Validation loader batches: {len(val_loader)}")
    print(f"Test loader batches: {len(test_loader)}")