import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms

def get_face_dataloaders(data_dir='./data/face', batch_size=128, num_workers=4, task='gender'):
    """
    Load face dataset from CSV for a specific task.
    
    Args:
        data_dir (str): Directory containing images and CSV.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of workers for DataLoader.
        task (str): Task to train on ('gender', 'age_10', 'age_5', 'disease').
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load CSV and split (assuming no explicit val split, we'll create one)
    csv_path = os.path.join(data_dir, 'face_images_path_with_meta_jpg_exist_only.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")
    
    df = pd.read_csv(csv_path)
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    
    # Create a validation split from train (e.g., 10%)
    val_split = 0.1
    train_size = int((1 - val_split) * len(train_df))
    train_df, val_df = train_df.iloc[:train_size], train_df.iloc[train_size:]

    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Custom Dataset class
    class CustomDataset(Dataset):
        def __init__(self, dataframe, task, image_folder, transform=None):
            self.dataframe = dataframe
            self.task = task
            self.transform = transform
            self.image_folder = image_folder
            
            valid_tasks = ['gender', 'age_10', 'age_5', 'disease']
            if task not in valid_tasks:
                raise ValueError(f"Task must be one of {valid_tasks}, got {task}")

            if self.task == 'gender':
                self.label_col = 'gender'
            elif self.task == 'age_10':
                self.label_col = 'age_div_10_round'
            elif self.task == 'age_5':
                self.label_col = 'age_div_5_round'
            elif self.task == 'disease':
                self.label_col = 'disease'
            
            unique_labels = sorted(self.dataframe[self.label_col].unique())
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            print(f"Task: {self.task}, Number of classes: {len(self.label_to_idx)}")

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            img_filename = self.dataframe.iloc[idx]['dest_filename']
            img_path = os.path.join(self.image_folder, img_filename)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found at: {img_path}")
            
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            label = self.label_to_idx[self.dataframe.iloc[idx][self.label_col]]
            return image, label

    # Create datasets
    train_dataset = CustomDataset(train_df, task, data_dir, train_transform)
    val_dataset = CustomDataset(val_df, task, data_dir, test_transform)
    test_dataset = CustomDataset(test_df, task, data_dir, test_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader