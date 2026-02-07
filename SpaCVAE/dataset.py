import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config

class TimeSeriesScaler:
    def __init__(self, range_values=(0, 1)):
        self.min = None
        self.max = None
        self.range_min, self.range_max = range_values

    def fit(self, data):
        self.min = np.min(data, axis=(0, 1), keepdims=True)
        self.max = np.max(data, axis=(0, 1), keepdims=True)
        return self

    def transform(self, data):
        scaled_data = (data - self.min) / (self.max - self.min + 1e-10)
        scaled_data = scaled_data * (self.range_max - self.range_min) + self.range_min
        return scaled_data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, scaled_data):
        data = (scaled_data - self.range_min) / (self.range_max - self.range_min + 1e-10)
        data = data * (self.max - self.min) + self.min
        return data


class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels): 
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx] 


class DataManager:
    def __init__(self, config):
        self.config = config
        self.P1_dataset = None
        self.P2_dataset = None
        self.P3_dataset = None
        self.all_data = None
        self.all_labels = None
        

        self.scaler_P1 = TimeSeriesScaler((0, 1))
        self.scaler_P2 = TimeSeriesScaler((0, 1))
        self.scaler_P3 = TimeSeriesScaler((0, 1))
        
        self.full_loader = None  

    def load_data(self):  
        AFOLU = pd.read_csv(f'{self.config.DATA_DIR}Emissions_CO2_AFOLU.csv')   
        
        mapping = {
            'P1':0, 'P1a':0, 'P1b':0, 'P1c':0, 'P1d':0,  
            'P2':1, 'P2a':1, 'P2b':1, 'P2c':1,  
            'P3a':2, 'P3b':2, 'P3c':2
            }  
        AFOLU['Policy_category'].replace(mapping, inplace=True)  
        AFOLU.reset_index(drop=True, inplace=True)  
        CO2 = pd.read_csv(f'{self.config.DATA_DIR}Emissions_CO2.csv')  
        Solids_Coal = pd.read_csv(f'{self.config.DATA_DIR}Final Energy_Industry_Solids_Coal.csv')  
        Solids_Biomass = pd.read_csv(f'{self.config.DATA_DIR}Final Energy_Industry_Solids_Biomass.csv')  
        RC_Solids_Coal = pd.read_csv(f'{self.config.DATA_DIR}Final Energy_Residential and Commercial_Solids_Coal.csv')  
        Industry = pd.read_csv(f'{self.config.DATA_DIR}Emissions_CO2_Energy_Demand_Industry.csv')  
        Elec_Coal = pd.read_csv(f'{self.config.DATA_DIR}Secondary Energy_Electricity_Coal.csv')  
        Supply_Elec = pd.read_csv(f'{self.config.DATA_DIR}Emissions_CO2_Energy_Supply_Electricity.csv')  
        Coal = pd.read_csv(f'{self.config.DATA_DIR}Primary Energy_Coal.csv') 
        Processes = pd.read_csv(f'{self.config.DATA_DIR}Emissions_CO2_Energy and Industrial Processes.csv') 
        
        model_scenario = AFOLU[['Model', 'Scenario']]  
        variables = [CO2, Solids_Coal, Solids_Biomass, RC_Solids_Coal, Industry, 
                 Elec_Coal, Supply_Elec, Coal, Processes]  
        
        for variable in variables:  
            model_scenario = pd.merge(model_scenario, variable[['Model', 'Scenario']],   
                                     on=['Model', 'Scenario'], how='inner')  
        
        for i in range(len(variables)):  
            variables[i] = pd.merge(model_scenario, variables[i], on=['Model', 'Scenario'], how='inner')  
        
        AFOLU = pd.merge(AFOLU, model_scenario, on=['Model', 'Scenario'], how='inner')  
        variables.append(AFOLU)  
        
        data_num = variables[0].shape[0]  
        X = np.zeros((data_num, self.config.TIME_STEPS, self.config.NUM_FEATURES))  
        
        for i in range(len(variables)):  
            variables[i] = variables[i].iloc[:, 5:].values  
        
        for i in range(data_num):  
            for j in range(self.config.TIME_STEPS):  
                for k in range(self.config.NUM_FEATURES):  
                    X[i][j][k] = (variables[k])[i, j]  
        

        Y = AFOLU['Policy_category'].values
        self.P1_dataset = X[Y == 0]
        self.P2_dataset = X[Y == 1]
        self.P3_dataset = X[Y == 2]
        
        self.all_data = X  
        self.all_labels = Y  
        return self

    def prepare_dataloaders(self):
        if self.P1_dataset is None:
            self.load_data()
        
        scaled_P1 = self.scaler_P1.fit_transform(self.P1_dataset)
        scaled_P2 = self.scaler_P2.fit_transform(self.P2_dataset)
        scaled_P3 = self.scaler_P3.fit_transform(self.P3_dataset)
        
        combined_data = np.concatenate([scaled_P1, scaled_P2, scaled_P3], axis=0)
        combined_labels = np.concatenate([
            np.zeros(len(scaled_P1)), 
            np.ones(len(scaled_P2)), 
            2 * np.ones(len(scaled_P3))
        ])

        dataset = TimeSeriesDataset(combined_data, combined_labels)
        self.full_loader = DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE["FULL"], 
            shuffle=True,
            drop_last=False
        )
        return self