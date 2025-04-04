from models.AttentionMobileNetShallow import AttentionMobileNetShallow
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



# Hard-coded configuration
config = {
    "model": "AttentionMobileNetShallow_single_face",
    "model_family": "AttentionMobileNetShallow",
    "dataset": "face",
    "task": "disease",
    "epochs": 50,
    "batch_size": 64,
    "optimizer": "adam",
    "lr": 0.001,
    "scheduler": None,
    "pretrained": False,
    "transfer_learning": False,
    "resize": 32
}

# Set up logging to a file
log_file = f"./checkpoints/{config['model']}_{config['dataset']}_{config['task']}_{'pretrained'if config['pretrained'] else 'noPretrained'}_{'transferLearning' if config['transfer_learning'] else 'noTransferLearning'}_training.log"
print(f"Logging to {log_file}")   
os.makedirs(os.path.dirname(log_file), exist_ok=True)  # Create the directory if it doesn't exist

# Configure logging (only once, with a FileHandler)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(log_file, mode='a')  # Append mode
file_handler.setLevel(logging.DEBUG)  # Set to DEBUG to capture all messages
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)
logging.info(f"Logging to {log_file}")
logging.info("Starting training...")

# redirects normal print statements to the log file
sys.stdout= open(log_file, 'a')

# Load configuration again to ensure it's available after setting up logging
print("Loading configuration...")
print(config)

save_config_path = f"./checkpoints/{config['model']}_{config['dataset']}_{config['task']}_{'pretrained'if config['pretrained'] else 'noPretrained'}_{'transferLearning' if config['transfer_learning'] else 'noTransferLearning'}_config.json"
os.makedirs("./checkpoints", exist_ok=True)
with open(save_config_path, 'w') as f:
    json.dump(config, f, indent=4)
    
print(f"Copied config file to {save_config_path}")
# calculate hash
import hashlib
from torchsummary import summary
from utils.model_utils import get_actual_size
from utils.train_utils import train_with_CIFAR10
import cv2
import numpy as np
print(f"Hash of the config file: {hashlib.md5(open(save_config_path, 'rb').read()).hexdigest()}")


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataloaders for task
trainloader, valloader, testloader, train_dataset, val_dataset, test_dataset, df = get_face_dataloaders(
    data_dir='./data/face',
    batch_size=config['batch_size'],
    num_workers=4,
    task=config['task'],
    resize=config.get('resize', 224)  # Use the resize value from config, default to 224
)

print(f"Number of classes for task: {len(train_dataset.label_to_idx.keys())}")
print(f"label_to_idx for task: {train_dataset.label_to_idx}")


# Determine the number of classes for each task
num_classes = len(train_dataset.label_to_idx.keys())

# Initialize the model using MultiViewAttentionCNN
model = AttentionMobileNetShallow(
    input_channels=3,
     n_classes=num_classes,
     input_size=config.get('resize', 224),  # Use the resize value from config, default to 224
     use_attention=True,
     attention_channels=16
     ).to(device)
print(model)

# Load checkpoint if provided from config
if config.get('checkpoint'):
    print(f"Loading checkpoint from {config['checkpoint']}")
    loaded_config = config.copy()
    model, loaded_config = load_checkpoint(model, config['checkpoint'], device)
    print(f"Loaded checkpoint config: {loaded_config}")
    config = loaded_config

# Define loss, optimizer
criterion = nn.CrossEntropyLoss()
if config['optimizer'].lower() == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
elif config['optimizer'].lower() == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
else:
    raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

# Define learning rate scheduler if specified
scheduler = None
# if config['scheduler']:
#     if config['scheduler'].lower() == 'step':
#         scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
#     elif config['scheduler'].lower() == 'cosine':
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
#     else:
#         raise ValueError(f"Unsupported scheduler: {config['scheduler']}")

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
    
    print(f"Epoch {epoch+1}/{config['epochs']} Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Epoch {epoch+1}/{config['epochs']} Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print(f"Epoch {epoch+1}/{config['epochs']} Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # -----------------------------------------------------------------------------------
    # TODO:
    # Save best model based on test
    if test_acc > best_acc:
        best_acc = test_acc
        
        if not os.path.exists("./checkpoints"):
            os.makedirs("./checkpoints")
        
        # Save the model checkpoint under the "checkpoints" directory
        filepath = f"./checkpoints/{config['model']}_{config['dataset']}_{config['task']}_{'pretrained'if config['pretrained'] else 'noPretrained'}_{'transferLearning' if config['transfer_learning'] else 'noTransferLearning'}_best.pth"
        save_checkpoint(model, config, filepath)
        best_acc_epoch = epoch + 1
        
        # Save epoch information to a JSON file
        info_filepath = f"./checkpoints/{config['model']}_{config['dataset']}_{config['task']}_{'pretrained'if config['pretrained'] else 'noPretrained'}_{'transferLearning' if config['transfer_learning'] else 'noTransferLearning'}_best.json"
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
        filepath = f"./checkpoints/{config['model']}_{config['dataset']}_{config['task']}_{'pretrained'if config['pretrained'] else 'noPretrained'}_{'transferLearning' if config['transfer_learning'] else 'noTransferLearning'}_val_best.pth"
        save_checkpoint(model, config, filepath)
        best_acc_val_epoch = epoch + 1

        # Save epoch information to a JSON file
        info_filepath = f"./checkpoints/{config['model']}_{config['dataset']}_{config['task']}_{'pretrained'if config['pretrained'] else 'noPretrained'}_{'transferLearning' if config['transfer_learning'] else 'noTransferLearning'}_val_best.json"
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
# Function to visualize attention maps
def visualize_attention_maps(model, dataloader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Get the first batch and select 5 random images
    data_iter = iter(dataloader)
    inputs, labels = next(data_iter)
    inputs, labels = inputs.to(device), labels.to(device)
    indices = torch.randperm(len(inputs))[:5]
    inputs = inputs[indices]
    labels = labels[indices]
    
    # Get attention maps
    with torch.no_grad():
        _, att_map, x_att = model(inputs, return_att_map=True)
        att_map = att_map.cpu().numpy()
        x_att = x_att.cpu().numpy()
        inputs = inputs.cpu().numpy()
        labels = labels.cpu().numpy()

    # Plot and show attention maps
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(5, 4, figsize=(10, 15))
    for i in range(5):
        # Original image
        axes[i, 0].imshow(np.transpose(inputs[i], (1, 2, 0)))
        axes[i, 0].set_title(f"Original Image - Label: {labels[i]}")
        axes[i, 0].axis('off')
        
        # Attention map overlay
        att_map_reshaped = cv2.resize(att_map[i], (32, 32))
        axes[i, 1].imshow(att_map_reshaped, cmap='hot', interpolation='nearest', alpha=0.6)
        axes[i, 1].imshow(np.transpose(inputs[i], (1, 2, 0)), alpha=0.4)
        axes[i, 1].set_title("Attention Map Overlay")
        axes[i, 1].axis('off')
        
        # Attention output mean over channels
        x_att_cur = np.mean(x_att[i, :, :, :], axis=0)
        x_att_cur_meaned_channels = cv2.resize(x_att_cur, (32, 32))
        x_att_cur_meaned_channels = cv2.normalize(x_att_cur_meaned_channels, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        axes[i, 2].imshow(x_att_cur_meaned_channels, cmap='hot', interpolation='nearest', alpha=0.6)
        axes[i, 2].imshow(np.transpose(inputs[i], (1, 2, 0)), alpha=0.4)
        axes[i, 2].set_title("Attention Output Overlay (Mean over channels)")
        axes[i, 2].axis('off')
        
        # Attention output mean visualization
        axes[i, 3].imshow(x_att_cur_meaned_channels, cmap='hot', interpolation='nearest', alpha=1)
        axes[i, 3].set_title("Attention Output (Mean over channels)")
        axes[i, 3].axis('off')
        
    plt.tight_layout()
    plt.show()

# Visualize attention maps for train and test loaders
visualize_attention_maps(model, train_loader)
visualize_attention_maps(model, test_loader)
sys.stdout.close()
