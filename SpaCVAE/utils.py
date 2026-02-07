import numpy as np  
import torch  
import random  
import matplotlib.pyplot as plt  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import GridSearchCV  
from sklearn.metrics import classification_report  

class Utils:  
    """Utility class with helper methods"""  
    @staticmethod  
    def set_seed(seed):  
        """Set random seed for reproducibility"""  
        torch.manual_seed(seed)  
        np.random.seed(seed)  
        random.seed(seed)  
        if torch.cuda.is_available():  
            torch.cuda.manual_seed(seed)  
            torch.cuda.manual_seed_all(seed)  
            torch.backends.cudnn.deterministic = True  
            torch.backends.cudnn.benchmark = False  
    
    @staticmethod
    def compute_enriched_features(data):
        data_num, time_steps, feature_num = data.shape
        enriched_features = np.zeros((data_num, feature_num * 3))  
        
        for i in range(data_num):
            for j in range(feature_num):
                enriched_features[i, j*3] = np.sum(data[i, :, j])             
                enriched_features[i, j*3+1] = (data[i, 2, j] - data[i, 0, j]) / 10         
                enriched_features[i, j*3+2] = (data[i, 4, j] - data[i, 2, j]) / 10       
                
        return enriched_features
    
    @staticmethod
    def get_feature_names(base_features):
        feature_names = []
        for feature in base_features:
            feature_names.extend([
                f"{feature}_sum_2020_2100",
                f"{feature}_trend_2020_2030",
                f"{feature}_trend_2030_2040",
            ])
        return feature_names
