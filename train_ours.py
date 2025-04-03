from models.MultiViewAttentionCNN import MultiViewAttentionCNN
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
    "model": "ours",
    "model_family": "MultiViewAttentionCNN",
    "dataset": "face",
    "task1": "gender",
    "task2": "age_10",
    "task3": "disease",
    "epochs": [5, 5, 5, 5],
    "batch_size": 64,
    "optimizer": "adam",
    "lr": 0.001,
    "scheduler": None,
    "pretrained": False,
    "transfer_learning": False,
    "resize": 32
}
config['task'] = "__".join([config['task1'], config['task2'], config['task3']])

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
print(f"Hash of the config file: {hashlib.md5(open(save_config_path, 'rb').read()).hexdigest()}")


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataloaders for task 1
trainloader_task1, valloader_task1, testloader_task1, train_dataset_task1, val_dataset_task1, test_dataset_task1, df = get_face_dataloaders(
    data_dir='./data/face',
    batch_size=config['batch_size'],
    num_workers=4,
    task=config['task1'],
    resize=config.get('resize', 224)  # Use the resize value from config, default to 224
)

print(f"Number of classes for task 1: {len(train_dataset_task1.label_to_idx.keys())}")
print(f"label_to_idx for task 1: {train_dataset_task1.label_to_idx}")

# Load dataloaders for task 2
trainloader_task2, valloader_task2, testloader_task2, train_dataset_task2, val_dataset_task2, test_dataset_task2, df = get_face_dataloaders(
    data_dir='./data/face',
    batch_size=config['batch_size'],
    num_workers=4,
    task=config['task2'],
    resize=config.get('resize', 224)  # Use the resize value from config, default to 224
)

print(f"Number of classes for task 2: {len(train_dataset_task2.label_to_idx.keys())}")
print(f"label_to_idx for task 2: {train_dataset_task2.label_to_idx}")

# Load dataloaders for task 3
trainloader_task3, valloader_task3, testloader_task3, train_dataset_task3, val_dataset_task3, test_dataset_task3, df = get_face_dataloaders(
    data_dir='./data/face',
    batch_size=config['batch_size'],
    num_workers=4,
    task=config['task3'],
    resize=config.get('resize', 224)  # Use the resize value from config, default to 224
    )

print(f"Number of classes for task 3: {len(train_dataset_task3.label_to_idx.keys())}")
print(f"label_to_idx for task 3: {train_dataset_task3.label_to_idx}")

# now load the model
# Determine the number of classes for each task
num_classes_task1 = len(train_dataset_task1.label_to_idx.keys())
num_classes_task2 = len(train_dataset_task2.label_to_idx.keys())
num_classes_task3 = len(train_dataset_task3.label_to_idx.keys())

# Initialize the model using MultiViewAttentionCNN
model = MultiViewAttentionCNN(
    image_size=config['resize'],
    image_depth=3,  # Assuming RGB images
    num_classes_list=[num_classes_task1, num_classes_task2, num_classes_task3],
    num_classes_final=num_classes_task3,
    drop_prob=0.5,  # Example dropout probability
    device=device,
)
print(model)


# Load checkpoint if provided from config
if config.get('checkpoint'):
    print(f"Loading checkpoint from {config['checkpoint']}")
    loaded_config = config.copy()
    model, loaded_config = load_checkpoint(model, config['checkpoint'], device)
    print(f"Loaded checkpoint config: {loaded_config}")
    config = loaded_config

# Define loss, optimizer
criterion_a = nn.CrossEntropyLoss()
criterion_b = nn.CrossEntropyLoss()
criterion_c = nn.CrossEntropyLoss()
criterion_c_fusion = nn.CrossEntropyLoss()

if config['optimizer'].lower() == 'adam':
    
    optimizer_a = optim.Adam(model.cnn_view_a.parameters(), lr=config['lr'])
    optimizer_b = optim.Adam(model.cnn_view_b.parameters(), lr=config['lr'])
    optimizer_c = optim.Adam(model.cnn_view_c.parameters(), lr=config['lr'])
    
    # last step will train c and fusion together with ab frozen
    optimizer_c_fusion = optim.Adam(
        list(model.cnn_view_c.parameters()) + list(model.fusion_layers.parameters()),
        lr=config['lr']
    )
    
elif config['optimizer'].lower() == 'sgd':
    # optimizer_a = optim.SGD(model.cnn_view_a.parameters(), lr=config['lr'], momentum=0.9)

    optimizer_a = optim.SGD(model.cnn_view_a.parameters(), lr=config['lr'], momentum=0.9)
    optimizer_b = optim.SGD(model.cnn_view_b.parameters(), lr=config['lr'], momentum=0.9)
    optimizer_c = optim.SGD(model.cnn_view_c.parameters(), lr=config['lr'], momentum=0.9)

    # last step will train c and fusion together with ab frozen
    optimizer_c_fusion = optim.SGD(
        list(model.cnn_view_c.parameters()) + list(model.fusion_layers.parameters()),
        lr=config['lr'],
        momentum=0.9
    )
else:
    raise ValueError(f"Unsupported optimizer: {config['optimizer']}")


    # def forward(self, view_a, view_b, view_c, return_individual_outputs=False, return_attention_features=False):
    #     features_a_reshaped_filters, features_a_x, features_a_output = self.cnn_view_a(view_a)
    #     features_b_reshaped_filters, features_b_x, features_b_output = self.cnn_view_b(view_b)
    #     features_c_reshaped_filters, features_c_x, features_c_output = self.cnn_view_c(view_c)

    #     if return_individual_outputs:
    #         if return_attention_features:
    #             return features_a_output, features_b_output, features_c_output, features_a_reshaped_filters, features_b_reshaped_filters, features_c_reshaped_filters
    #         else:
    #             return features_a_output, features_b_output, features_c_output
    #     else:
    #         # print(f"Shape of combined_features: {features_a_reshaped_filters.shape}, {features_b_reshaped_filters.shape}, {features_c_reshaped_filters.shape}")
    #         combined_features = torch.cat((features_a_reshaped_filters, features_b_reshaped_filters, features_c_reshaped_filters), dim=1)
    #         # print(f"Shape of combined_features after cat: {features_a_reshaped_filters.shape}, {features_b_reshaped_filters.shape}, {features_c_reshaped_filters.shape}")
    #         combined_features = combined_features.reshape(combined_features.size(0), -1)
    #         # print(f"Shape of combined_features after reshape: {features_a_reshaped_filters.shape}, {features_b_reshaped_filters.shape}, {features_c_reshaped_filters.shape}")
    #         fused_output = self.fusion_layers(combined_features)
    #         if return_attention_features:
    #             return fused_output, features_a_reshaped_filters, features_b_reshaped_filters, features_c_reshaped_filters
    #         else:
    #             return fused_output

# first train task 1
print("Training task 1...")
# currently all 3 views are the same
# so just loop through the first one
train_loader = trainloader_task1
val_loader = valloader_task1
train_dataset = train_dataset_task1

# Set model to training mode
model.train()
# Loop through epochs
best_acc = 0.0
best_acc_epoch = -1
best_acc_val = 0.0
best_acc_val_epoch = -1
for epoch in range(config['epochs'][0]):
    print(f"\nEpoch {epoch+1}/{config['epochs']}")


# loop through the first dataloader
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_a.zero_grad()

        # Forward pass
        features_a_output, features_b_output, features_c_output, features_a_reshaped_filters, features_b_reshaped_filters, features_c_reshaped_filters = model(inputs, inputs, inputs, return_individual_outputs=True, return_attention_features=False)
        print(f"Shape of features_a_output: {features_a_output.shape}")
        print(f"Shape of features_b_output: {features_b_output.shape}")
        print(f"Shape of features_c_output: {features_c_output.shape}")
        print(f"Shape of features_a_reshaped_filters: {features_a_reshaped_filters.shape}")
        print(f"Shape of features_b_reshaped_filters: {features_b_reshaped_filters.shape}")
        print(f"Shape of features_c_reshaped_filters: {features_c_reshaped_filters.shape}")
        break
