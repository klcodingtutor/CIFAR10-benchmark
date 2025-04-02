import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from .augmentations import get_train_transforms, get_test_transforms

def get_cifar10_dataloaders(data_dir='./data', batch_size=128, val_split=0.1, num_workers=4):
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=get_train_transforms())
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=get_test_transforms())

    num_train = len(train_dataset)
    indices = torch.randperm(num_train).tolist()
    split = int(val_split * num_train)
    train_idx, val_idx = indices[split:], indices[:split]
    test_indices = torch.randperm(num_train).tolist()
    test_idx = test_indices[:len(test_dataset)]


    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)
    test_subset = Subset(test_dataset, range(len(test_dataset)))

    PARTIAL_RATIO = 0.1 # reduce everything to 10% of the original dataset
    partial_train_idx = train_idx[:int(PARTIAL_RATIO * len(train_idx))]
    partial_val_idx = val_idx[:int(PARTIAL_RATIO * len(val_idx))]
    partial_test_idx = test_idx[:int(PARTIAL_RATIO * len(test_idx))]
    print(f"Using partial dataset with ratio {PARTIAL_RATIO}")
    print(f"Full train size: {len(train_idx)} >> Partial train size: {len(partial_train_idx)}")
    print(f"Full val size: {len(val_idx)} >> Partial val size: {len(partial_val_idx)}")
    print(f"Full test size: {len(test_dataset)} >> Partial test size: {len(partial_test_idx)}")

    train_subset = Subset(train_dataset, partial_train_idx)
    val_subset = Subset(train_dataset, partial_val_idx)
    test_subset = Subset(test_dataset, partial_test_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader