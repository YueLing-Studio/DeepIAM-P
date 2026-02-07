import torch  
import os  

class Config:  
    # Data parameters  
    DATA_DIR = './data_use/'  
    FEATURE_NAMES = [
        'Emissions_CO2', 'FinalEnergy_Industry_Solids_Coal',  'FinalEnergy_Industry_Solids_Biomass', 
        'FinalEnergy_ResidentialandCommercial_Solids_Coal', 'Emissions_CO2_Energy_Demand_Industry', 
        'SecondaryEnergy_Electricity_Coal', 'Emissions_CO2_Energy_Supply_Electricity', 
        'PrimaryEnergy_Coal','Emissions_CO2_EnergyandIndustrialProcesses', 'Emissions_CO2_AFOLU' 
        ]  
    
    
    TIME_STEPS = 17  
    NUM_FEATURES = 10  
    
    # Model parameters  
    LATENT_DIM = 20  
    HIDDEN_DIMS = [64, 128, 256, 512]  
    
    # Training parameters 
    BATCH_SIZE = {"FULL": 60}
    EPOCHS = 1500  
    LEARNING_RATE = 1e-4  
    LR_DECAY_STEPS = 200  
    LR_DECAY_RATE = 0.90  
    BETA =0.2   # Beta parameter for VAE  
    FINAL_BETA = 0.8
    
    # Random seed  
    SEED = 3  
    
    # Device configuration  
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    
    # Generation parameters  
    GEN_SAMPLES = 1000  # per category  
    
    # Evaluation parameters  
    RF_PARAMS = {  
        'n_estimators': [10, 50, 100, 200, 500],  
        'max_depth': [6, 8, 10, 12, 14, 16],  
        'min_samples_split': [ 3, 4, 5, 6]  
    }  
    
    # Categories  
    CATEGORY_NAMES = ['P1', 'P2', 'P3']