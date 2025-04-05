from tqdm import tqdm
from models.AttentionMobileNetShallow import AttentionMobileNetShallow
from models.MultiViewAttentionMobileNetShallow import MultiViewAttentionMobileNetShallow

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
    "model": "AttentionMobileNetShallow_multiview_face",
    "model_family": "AttentionMobileNetShallow",
    "dataset": "face",
    "task": "disease",
    
    "checkpoint_age": "/content/CIFAR10-benchmark/checkpoints/AttentionMobileNetShallow_s_single_face_face_age_10_noPretrained_noTransferLearning_val_best.pth",
    "checkpoint_gender": "/content/CIFAR10-benchmark/checkpoints/AttentionMobileNetShallow_s_single_face_face_gender_noPretrained_noTransferLearning_val_best.pth",


    "epochs": 100,
    "batch_size": 64,
    "optimizer": "adam",
    "lr": 0.001,
    "scheduler": None,
    "pretrained": False,
    "transfer_learning": False,
    "resize": 32
}

# Load configuration
print(f"Loaded configuration: {config}")

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
from utils.get_actual_size import get_actual_size
import cv2
import numpy as np
print(f"Hash of the config file: {hashlib.md5(open(save_config_path, 'rb').read()).hexdigest()}")


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataloaders for task
train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, df = get_face_dataloaders(
    data_dir='./data/face',
    batch_size=config['batch_size'],
    num_workers=4,
    task=config['task'],
    resize=config.get('resize', 224)  # Use the resize value from config, default to 224
)

print(f"Number of classes for task: {len(train_dataset.label_to_idx.keys())}")
print(f"label_to_idx for task: {train_dataset.label_to_idx}")

# task is either 'disease', 'age_10' or 'gender'
# now load and get the num class for each
train_loader_age, val_loader_age, test_loader_age, train_dataset_age, val_dataset_age, test_dataset_age, df_age = get_face_dataloaders(
    data_dir='./data/face',
    batch_size=config['batch_size'],
    num_workers=4,
    task='age_10',
    resize=config.get('resize', 224)  # Use the resize value from config, default to 224
)
train_loader_gender, val_loader_gender, test_loader_gender, train_dataset_gender, val_dataset_gender, test_dataset_gender, df_gender = get_face_dataloaders(
    data_dir='./data/face',
    batch_size=config['batch_size'],
    num_workers=4,
    task='gender',
    resize=config.get('resize', 224)  # Use the resize value from config, default to 224
)

# Print dataset information
print(f"Number of classes for age task: {len(train_dataset_age.label_to_idx.keys())}")
print(f"label_to_idx for age task: {train_dataset_age.label_to_idx}")
print(f"Number of classes for gender task: {len(train_dataset_gender.label_to_idx.keys())}")
print(f"label_to_idx for gender task: {train_dataset_gender.label_to_idx}")


# Determine the number of classes for each task
num_classes = len(train_dataset.label_to_idx.keys())
num_classes_age = len(train_dataset_age.label_to_idx.keys())
num_classes_gender = len(train_dataset_gender.label_to_idx.keys())


# Initialize the model using MultiViewAttentionCNN
submodel = AttentionMobileNetShallow(
    input_channels=3,
     n_classes=num_classes,
     input_size=config.get('resize', 224),  # Use the resize value from config, default to 224
     use_attention=True,
     attention_channels=16
     ).to(device)
print(submodel)

submodel_age = AttentionMobileNetShallow(
    input_channels=3,
    n_classes=num_classes_age,
    input_size=config.get('resize', 224),  # Use the resize value from config, default to 224
    use_attention=True,
    attention_channels=16
    ).to(device)
print(submodel_age)

submodel_gender = AttentionMobileNetShallow(
    input_channels=3,
    n_classes=num_classes_gender,
    input_size=config.get('resize', 224),  # Use the resize value from config, default to 224
    use_attention=True,
    attention_channels=16
    ).to(device)
print(submodel_gender)

# Load checkpoint
print(f"Loading checkpoint from {config['checkpoint_age']}")
submodel_age, loaded_config_age = load_checkpoint(submodel_age, config['checkpoint_age'], device)
print(f"Loaded checkpoint config: {loaded_config_age}")
print(f"Loading checkpoint from {config['checkpoint_gender']}")
submodel_gender, loaded_config_gender = load_checkpoint(submodel_gender, config['checkpoint_gender'], device)
print(f"Loaded checkpoint config: {loaded_config_gender}")



model = MultiViewAttentionMobileNetShallow(
    pretrained_models=[
        submodel_age,
        submodel_gender
    ],
    not_trained_models=[submodel],
    n_classes=num_classes,
)
print(model)
model.freeze_pretrained_models()
model.unfreeze_not_trained_models()
model.unfreeze_fusion_layer()

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
    
    # train_loss, train_acc, val_loss, val_acc = train_epoch(model, train_loader, val_loader, criterion, optimizer, device)
    # test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs, return_att_map=False)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_loss = val_loss / len(val_loader)
    val_acc = 100. * val_correct / val_total
    

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, return_att_map=False)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    
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

# # Function to visualize attention maps
# def visualize_attention_maps(model, dataloader):
#     model.eval()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
    
#     # Get the first batch and select 5 random images
#     data_iter = iter(dataloader)
#     inputs, labels = next(data_iter)
#     inputs, labels = inputs.to(device), labels.to(device)
#     indices = torch.randperm(len(inputs))[:5]
#     inputs = inputs[indices]
#     labels = labels[indices]
    
#     # Get attention maps
#     with torch.no_grad():
#         _, att_map, x_att = model(inputs, return_att_map=True)
#         att_map = att_map.cpu().numpy()
#         x_att = x_att.cpu().numpy()
#         inputs = inputs.cpu().numpy()
#         labels = labels.cpu().numpy()

#     # Plot and show attention maps
#     import matplotlib.pyplot as plt
    
#     fig, axes = plt.subplots(5, 4, figsize=(10, 15))
#     for i in range(5):
#         # Original image
#         axes[i, 0].imshow(np.transpose(inputs[i], (1, 2, 0)))
#         axes[i, 0].set_title(f"Original Image - Label: {labels[i]}")
#         axes[i, 0].axis('off')
        
#         # Attention map overlay
#         att_map_reshaped = cv2.resize(att_map[i], (32, 32))
#         axes[i, 1].imshow(att_map_reshaped, cmap='hot', interpolation='nearest', alpha=0.6)
#         axes[i, 1].imshow(np.transpose(inputs[i], (1, 2, 0)), alpha=0.4)
#         axes[i, 1].set_title("Attention Map Overlay")
#         axes[i, 1].axis('off')
        
#         # Attention output mean over channels
#         x_att_cur = np.mean(x_att[i, :, :, :], axis=0)
#         x_att_cur_meaned_channels = cv2.resize(x_att_cur, (32, 32))
#         x_att_cur_meaned_channels = cv2.normalize(x_att_cur_meaned_channels, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
#         axes[i, 2].imshow(x_att_cur_meaned_channels, cmap='hot', interpolation='nearest', alpha=0.6)
#         axes[i, 2].imshow(np.transpose(inputs[i], (1, 2, 0)), alpha=0.4)
#         axes[i, 2].set_title("Attention Output Overlay (Mean over channels)")
#         axes[i, 2].axis('off')
        
#         # Attention output mean visualization
#         axes[i, 3].imshow(x_att_cur_meaned_channels, cmap='hot', interpolation='nearest', alpha=1)
#         axes[i, 3].set_title("Attention Output (Mean over channels)")
#         axes[i, 3].axis('off')
    
        
#     plt.tight_layout()
#     save_path = f"./output_image/{config['model']}_{config['dataset']}_{config['task']}_{'pretrained'if config['pretrained'] else 'noPretrained'}_{'transferLearning' if config['transfer_learning'] else 'noTransferLearning'}_attention_map_{i}.png"
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path)
#     print(f"Saved attention map to {save_path}")
#     plt.close(fig)

# # Visualize attention maps for train and test loaders
# visualize_attention_maps(model, train_loader)
# visualize_attention_maps(model, test_loader)
sys.stdout.close()
