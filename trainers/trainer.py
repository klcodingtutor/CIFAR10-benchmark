import os
from utils.metrics import compute_accuracy
import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, logger, writer, device='cuda', save_suffix=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        self.writer = writer
        self.device = device
        self.save_suffix = save_suffix
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.best_acc = 0.0

    def _get_optimizer(self):
        if self.config['optimizer'] == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.config['lr'], momentum=self.config.get('momentum', 0.9))
        elif self.config['optimizer'] == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.config['lr'])
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer']}")

    def _get_scheduler(self):
        if self.config['scheduler'] == 'steplr':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config['step_size'], gamma=self.config['gamma'])
        return None

    def train(self):
        for epoch in range(self.config['epochs']):
            self.model.train()
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                if i % 10 == 9:
                    self.logger.info(f'Epoch [{epoch+1}/{self.config["epochs"]}], Step [{i+1}], Loss: {running_loss/10:.4f}')
                    self.writer.add_scalar('train/loss', running_loss/10, epoch * len(self.train_loader) + i)
                    running_loss = 0.0
            
            if self.scheduler:
                self.scheduler.step()
            
            # Validation
            val_acc = self.validate(epoch)
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                if not os.path.exists('checkpoints'):
                    os.makedirs('checkpoints')
                if self.save_suffix is not None:
                    torch.save(self.model.state_dict(), f'checkpoints/{self.config["model"]}_{self.save_suffix}_best.pth')
                else:
                    torch.save(self.model.state_dict(), f'checkpoints/{self.config["model"]}_best.pth')
    
    def validate(self, epoch):
        self.model.eval()
        total_acc = 0.0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                acc = compute_accuracy(outputs, targets)[0].item()
                total_acc += acc
        avg_acc = total_acc / len(self.val_loader)
        self.logger.info(f'Epoch [{epoch+1}/{self.config["epochs"]}], Val Accuracy: {avg_acc:.2f}%')
        self.writer.add_scalar('val/accuracy', avg_acc, epoch)
        return avg_acc