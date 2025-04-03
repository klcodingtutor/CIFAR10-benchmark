import os
import torch
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser
from utils.config import load_config
from utils.train_utils import train_epoch, evaluate, save_checkpoint, load_checkpoint
from dataloaders.cifar10_loader import get_cifar10_dataloaders
from dataloaders.get_face_dataloaders import get_face_dataloaders
import sys
from models import get_model
import json

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Parse command-line arguments
    parser = ArgumentParser(description="Model Benchmarking")
    parser.add_argument('--config', type=str, required=True, help="Path to the config YAML file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging to a file
    log_file = f"./checkpoints/{config['model']}_{config['dataset']}_{config['task']}_best.log"
    print(f"Logging to {log_file}")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)  # Create the directory if it doesn't exist
    
    # Create the directory if it doesn't exist
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting training...")

    # Redirect stdout and stderr to the log file
    sys.stdout = open(log_file, 'w+')
    sys.stderr = open(log_file, 'w+')

    # Load configuration again to ensure it's available after setting up logging
    config = load_config(args.config)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataloaders based on dataset
    if config['dataset'].lower() == 'cifar10':
        trainloader, valloader, testloader = get_cifar10_dataloaders(config['batch_size'])
        num_classes = 10  # Fixed for CIFAR-10
    elif config['dataset'].lower() == 'face':
        trainloader, valloader, testloader, train_dataset, val_dataset, test_dataset, df = get_face_dataloaders(
            data_dir='./data/face',
            batch_size=config['batch_size'],
            num_workers=4,
            task=config['task'],
            resize=config.get('resize', 224)  # Use the resize value from config, default to 224
        )
        num_classes = len(train_dataset.label_to_idx)  # Dynamically set from train_dataset
        print(f"Number of classes for task '{config['task']}': {num_classes}")
    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")
    
    # Load model
    model = get_model(
        model_name=config['model'],
        model_family=config['model_family'],
        pretrained=config['pretrained'],
        transfer_learning=config['transfer_learning'],
        num_classes=num_classes
    ).to(device)
    
    # # Load checkpoint if provided from config
    # if config.get('checkpoint'):
    #     print(f"Loading checkpoint from {config['checkpoint']}")
    #     loaded_config = config.copy()
    #     model, loaded_config = load_checkpoint(model, args.checkpoint, device)
    #     print(f"Loaded checkpoint config: {loaded_config}")
    #     config = loaded_config
    
    # Define loss, optimizer
    criterion = nn.CrossEntropyLoss()
    if config['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
    
    # Optional: Add a learning rate scheduler if specified in the config
    scheduler = None
    if config.get('scheduler') == 'step_lr':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training loop
    # -----------------------------------------------------------------------------------
    # TODO:
    best_acc = 0.0
    best_acc_epoch = -1
    best_acc_val = 0.0
    best_acc_val_epoch = -1
    # -----------------------------------------------------------------------------------
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        train_loss, train_acc, val_loss, val_acc = train_epoch(model, trainloader, valloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # -----------------------------------------------------------------------------------
        # TODO:
        # Save best model based on test
        if test_acc > best_acc:
            best_acc = test_acc
            
            if not os.path.exists("./checkpoints"):
                os.makedirs("./checkpoints")
            
            # Save the model checkpoint under the "checkpoints" directory
            filepath = f"./checkpoints/{config['model']}_{config['dataset']}_{config['task']}_best.pth"
            save_checkpoint(model, config, filepath)
            best_acc_epoch = epoch + 1
            
            # Save epoch information to a JSON file
            info_filepath = f"./checkpoints/{config['model']}_{config['dataset']}_{config['task']}_best.json"
            epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc
            }
            with open(info_filepath, "w") as f:
                json.dump(epoch_info, f, indent=4)
        # -----------------------------------------------------------------------------------
    
        # Save best model based on val
        if val_acc > best_acc_val:
            best_acc_val = val_acc
            filepath = f"./checkpoints/{config['model']}_{config['dataset']}_{config['task']}_val_best.pth"
            save_checkpoint(model, config, filepath)
            best_acc_val_epoch = epoch + 1

            # Save epoch information to a JSON file
            info_filepath = f"./checkpoints/{config['model']}_{config['dataset']}_{config['task']}_val_best.json"
            epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc
            }
            with open(info_filepath, "w") as f:
                json.dump(epoch_info, f, indent=4)
        
        # Step the scheduler if it exists
        if scheduler:
            scheduler.step()
    
    print(f"Training completed. Best Test Acc: {best_acc:.2f}% at epoch {best_acc_epoch}, "
          f"Best Val Acc: {best_acc_val:.2f}% at epoch {best_acc_val_epoch}")

if __name__ == "__main__":
    main()