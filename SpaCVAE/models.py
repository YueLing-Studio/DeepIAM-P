import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from config import Config  



import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class ConvEncoder(nn.Module):
    def __init__(self, time_steps, num_features, latent_dim, num_labels=3, hidden_dims=None):  
        super().__init__()
        self.time_steps = time_steps
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.num_labels = num_labels

        self.label_embed = nn.Embedding(num_labels, time_steps) 

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims

        in_channels = num_features + 1  

        modules = []
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, h_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.flatten = nn.Flatten()

        dummy_input = torch.zeros(1, num_features + 1, time_steps)  
        dummy_output = self.encoder(dummy_input)
        self.flattened_size = self.flatten(dummy_output).shape[1]

        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_var = nn.Linear(self.flattened_size, latent_dim)

    def forward(self, x, labels): 
        batch_size = x.shape[0]
        embedded_labels = self.label_embed(labels) 
        label_channel = embedded_labels.view(batch_size, 1, self.time_steps)  
        x = x.permute(0, 2, 1)  
        x = torch.cat([x, label_channel], dim=1) 
        
        x = self.encoder(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def save(self, path):  
        torch.save(self.state_dict(), path)  
    
    def load(self, path, device=None):  
        if device:  
            self.load_state_dict(torch.load(path, map_location=device))  
        else:  
            self.load_state_dict(torch.load(path)) 


class ConvDecoder(nn.Module):
    def __init__(self, time_steps, num_features, latent_dim, num_labels=3, hidden_dims=None): 
        super().__init__()
        self.time_steps = time_steps
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.num_labels = num_labels


        self.label_embed = nn.Embedding(num_labels, 10)  
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]
        self.hidden_dims = hidden_dims[::-1]


        adjusted_latent_dim = latent_dim + 10  

        self.decoder_input = nn.Linear(adjusted_latent_dim, self.hidden_dims[0] * time_steps)

        modules = []
        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(self.hidden_dims[i], self.hidden_dims[i+1], kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(self.hidden_dims[i+1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(self.hidden_dims[-1], num_features, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z, labels):  

        embedded_labels = self.label_embed(labels)  
        z = torch.cat([z, embedded_labels], dim=1)  

        x = self.decoder_input(z)
        x = x.view(-1, self.hidden_dims[0], self.time_steps)
        x = self.decoder(x)
        x = self.final_layer(x)
        x = x.permute(0, 2, 1)  
        return x


    def save(self, path):  

        torch.save(self.state_dict(), path)  
    
    def load(self, path, device=None):  

        if device:  
            self.load_state_dict(torch.load(path, map_location=device))  
        else:  
            self.load_state_dict(torch.load(path))  


class VAE(nn.Module):
    def __init__(self, config, category=None):
        super().__init__()
        self.config = config
        self.time_steps = config.TIME_STEPS
        self.num_features = config.NUM_FEATURES
        self.latent_dim = config.LATENT_DIM
        self.initial_beta = config.BETA
        self.final_beta = config.FINAL_BETA
        self.category = category

        self.encoder = ConvEncoder(
            self.time_steps,
            self.num_features,
            self.latent_dim,
            num_labels=3, 
            hidden_dims=config.HIDDEN_DIMS
        )
        self.decoder = ConvDecoder(
            self.time_steps,
            self.num_features,
            self.latent_dim,
            num_labels=3,  
            hidden_dims=config.HIDDEN_DIMS
        )
        self.device = config.DEVICE

    def forward(self, x, labels):  
        mu, log_var = self.encoder(x, labels)  
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z, labels)  
        return recon_x, mu, log_var

    def generate(self, z=None, labels=None, num_samples=None): 
        self.eval()
        with torch.no_grad():
            if z is None:
                if num_samples is None:
                    raise ValueError("Please provide either z or num_samples")
                z = torch.randn(num_samples, self.latent_dim, device=self.device)
            if labels is None:
                raise ValueError("Conditional VAE needs target label to generate samples")
            return self.decoder(z, labels) 

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def loss_function(self, recon_x, x, mu, log_var, beta=1.0):

        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + beta * kld_loss
        return loss, recon_loss, kld_loss
    
    
    def to(self, device):
        self.device = device
        return super().to(device)
    
    def save_weights(self, dir_path='.'):  
        category_str = f"_{self.category}" if self.category else ""  
        self.encoder.save(f"{dir_path}/encoder{category_str}.pth")  
        self.decoder.save(f"{dir_path}/decoder{category_str}.pth")  
    
    def load_weights(self, dir_path='.'):  
        category_str = f"_{self.category}" if self.category else ""  
        self.encoder.load(f"{dir_path}/encoder{category_str}.pth", self.device)  
        self.decoder.load(f"{dir_path}/decoder{category_str}.pth", self.device)