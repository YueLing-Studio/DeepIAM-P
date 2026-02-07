import torch
import numpy as np
from config import Config

class Generator:
    def __init__(self, model, scalers, config):
        self.config = config
        self.device = config.DEVICE
        
        self.cvae = model  
        self.cvae.to(self.device)
        
        self.scaler_P1 = scalers[0]
        self.scaler_P2 = scalers[1]
        self.scaler_P3 = scalers[2]
        
        self.gen_data = None
        self.gen_labels = None
        self.gen_P1 = None  
        self.gen_P2 = None
        self.gen_P3 = None

    def generate_samples(self, target_label, num_samples):
        self.cvae.eval()
        
        z = torch.randn(num_samples, self.config.LATENT_DIM, device=self.device)

        labels = torch.full((num_samples,), target_label, dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            samples = self.cvae.generate(z, labels).cpu().numpy()  
            
        if target_label == 0:
            scaled_samples = self.scaler_P1.inverse_transform(samples)
        elif target_label == 1:
            scaled_samples = self.scaler_P2.inverse_transform(samples)
        elif target_label == 2:
            scaled_samples = self.scaler_P3.inverse_transform(samples)
        else:
            raise ValueError("invalid label")
            
        return scaled_samples

    def generate_all(self):

        self.gen_P1 = self.generate_samples(target_label=0, num_samples=self.config.GEN_SAMPLES)
        self.gen_P2 = self.generate_samples(target_label=1, num_samples=self.config.GEN_SAMPLES)
        self.gen_P3 = self.generate_samples(target_label=2, num_samples=self.config.GEN_SAMPLES)
        
        self.gen_data = np.concatenate((self.gen_P1, self.gen_P2, self.gen_P3), axis=0)
        
        self.gen_labels = np.concatenate([
            np.zeros(self.config.GEN_SAMPLES), 
            np.ones(self.config.GEN_SAMPLES),    
            2 * np.ones(self.config.GEN_SAMPLES) 
        ]).astype(np.int32)
        
        return self.gen_data, self.gen_labels, self.gen_P1, self.gen_P2, self.gen_P3