import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import Config
from models import VAE  


class VAETrainer:

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.DEVICE

        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ExponentialDecay(
            self.optimizer,
            gamma=config.LR_DECAY_RATE,
            step_size=config.LR_DECAY_STEPS
        )

        self.train_losses = []
        self.recon_losses = []
        self.kld_losses = []

        self.model.to(self.device)

    def calculate_beta(self, epoch, total_epochs):
        return self.config.BETA + (self.config.FINAL_BETA - self.config.BETA) * (epoch / total_epochs)

    def train_epoch(self, data_loader, epoch, total_epochs):
        self.model.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kld_loss = 0

        current_beta = self.calculate_beta(epoch, total_epochs)

        for batch_data, batch_labels in data_loader: 
            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            recon_batch, mu, log_var = self.model(batch_data, batch_labels)
            
            loss, recon_loss, kld_loss = self.model.loss_function(
                recon_batch, batch_data, mu, log_var, beta=current_beta
            )

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kld_loss += kld_loss.item()

        avg_loss = epoch_loss / len(data_loader.dataset)
        avg_recon_loss = epoch_recon_loss / len(data_loader.dataset)
        avg_kld_loss = epoch_kld_loss / len(data_loader.dataset)

        return avg_loss, avg_recon_loss, avg_kld_loss

    def train(self, data_loader, epochs=None):
        if epochs is None:
            epochs = self.config.EPOCHS

        total_epochs = epochs
        for epoch in range(epochs):
            avg_loss, avg_recon_loss, avg_kld_loss = self.train_epoch(data_loader, epoch, total_epochs)
            
            self.scheduler.step()
            
            self.train_losses.append(avg_loss)
            self.recon_losses.append(avg_recon_loss)
            self.kld_losses.append(avg_kld_loss)

            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, '
                      f'Recon: {avg_recon_loss:.4f}, KLD: {avg_kld_loss:.4f}, Beta: {self.calculate_beta(epoch, total_epochs):.2f}')

        if self.model.category:
            self.model.save_weights()

        return self.train_losses, self.recon_losses, self.kld_losses


class ModelManager:
    def __init__(self, config):
        self.config = config
        

        self.cvae = VAE(config)  
        self.trainer = VAETrainer(self.cvae, config)
        

        self.losses = None

    def train_model(self, data_loader, epochs=None):
        print("Training Conditional VAE...")
        self.losses = self.trainer.train(data_loader, epochs=epochs)
        return self.cvae


class LRScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, gamma, step_size, last_epoch=-1):
        self.gamma = gamma
        self.step_size = step_size
        super(LRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        decay_factor = self.gamma ** (self.last_epoch // self.step_size)
        return [base_lr * decay_factor for base_lr in self.base_lrs]

setattr(optim.lr_scheduler, 'ExponentialDecay', LRScheduler)