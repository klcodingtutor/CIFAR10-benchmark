import torch
import torch.nn as nn
import torch.optim as optim
from argparse import ArgumentParser
from utils.config import load_config
from utils.train_utils import train_epoch, evaluate, save_checkpoint
from dataloaders.cifar10_loader import get_cifar10_dataloaders
from models import get_model
import json

def main():
    # Parse command-line arguments
    parser = ArgumentParser(description="CIFAR-10 Model Benchmarking")
    parser.add_argument('--config', type=str, required=True, help="Path to the config YAML file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataloaders
    trainloader, valloader, testloader = get_cifar10_dataloaders(config['batch_size'])


    
    # Load model
    model = get_model(
        model_name=config['model'],
        model_family=config['model_family'],
        pretrained=config['pretrained'],
        transfer_learning=config['transfer_learning'],
        num_classes=10
    ).to(device)
    
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
            filepath = f"{config['model']}_{config['dataset']}_{config['task']}_best.pth"
            save_checkpoint(model, config, filepath)
            best_acc_epoch = epoch + 1
            
            # Save epoch information to a JSON file
            info_filepath = f"{config['model']}_{config['dataset']}_{config['task']}_best.json"
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
            filepath = f"{config['model']}_{config['dataset']}_{config['task']}_val_best.pth"
            save_checkpoint(model, config, filepath)
            best_acc_val_epoch = epoch + 1

            # Save epoch information to a JSON file
            info_filepath = f"{config['model']}_{config['dataset']}_{config['task']}_val_best.json"
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

    print(f"Training completed. Best Val Acc: {best_acc:.2f}%")

if __name__ == "__main__":
    main()