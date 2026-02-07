import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from config import Config
from utils import Utils

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.category_names = config.CATEGORY_NAMES
        self.feature_names = config.FEATURE_NAMES
        
        self.extended_feature_names = Utils.get_feature_names(self.feature_names)
        
        self.real_features = None
        self.gen_features = None
        self.real_df = None
        self.gen_df = None
        
        self.rf_classifier_real_to_gen = None
        self.rf_classifier_gen_to_real = None

        self.real_to_gen_accuracy = 0.0
        self.gen_to_real_accuracy = 0.0     
        self.real_to_real_train_accuracy = 0.0  
        self.gen_to_gen_train_accuracy = 0.0   
        
    def compute_features(self, real_data, real_labels, gen_data, gen_labels):

        self.real_features = Utils.compute_enriched_features(real_data)
        self.gen_features = Utils.compute_enriched_features(gen_data)

        self.real_df = pd.DataFrame(self.real_features, columns=self.extended_feature_names)
        self.gen_df = pd.DataFrame(self.gen_features, columns=self.extended_feature_names)
        
        self.real_df['Policy_category'] = real_labels
        self.gen_df['Policy_category'] = gen_labels
    
    def train_real_to_gen_classifier(self):
        X_train = self.real_df.drop('Policy_category', axis=1)
        y_train = self.real_df['Policy_category']
        X_test = self.gen_df.drop('Policy_category', axis=1)
        y_test = self.gen_df['Policy_category']

        base_clf = RandomForestClassifier(random_state=42)
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.rf_classifier_real_to_gen = GridSearchCV(
            base_clf, 
            self.config.RF_PARAMS, 
            cv=cv_strategy, 
            n_jobs=-1, 
            verbose=1
        )
        
        self.rf_classifier_real_to_gen.fit(X_train, y_train)
        
        print('Best param (real→synthetic):')
        print(self.rf_classifier_real_to_gen.best_params_)

        y_train_pred = self.rf_classifier_real_to_gen.predict(X_train)
        self.real_to_real_train_accuracy = accuracy_score(y_train, y_train_pred)
        print(f"Self-evaluation accuracy for real data: {self.real_to_real_train_accuracy:.4f}")
        print('\nClassification report for self-evaluation on real data:')
        print(classification_report(y_train, y_train_pred, 
                                target_names=self.category_names))
        
        
        y_pred = self.rf_classifier_real_to_gen.predict(X_test)
        self.real_to_gen_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"real→synthetic test accuracy: {self.real_to_gen_accuracy:.4f}")
        print(f"real→synthetic classification report:")
        print(classification_report(y_test, y_pred, target_names=self.category_names))
        
        return self.real_to_gen_accuracy
    
    def train_gen_to_real_classifier(self):
        X_train = self.gen_df.drop('Policy_category', axis=1)
        y_train = self.gen_df['Policy_category']
        X_test = self.real_df.drop('Policy_category', axis=1)
        y_test = self.real_df['Policy_category']
        
        base_clf = RandomForestClassifier(random_state=42)
        cv_strategy = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
        self.rf_classifier_gen_to_real = GridSearchCV(
            base_clf, 
            self.config.RF_PARAMS, 
            cv=cv_strategy, 
            n_jobs=-1, 
            verbose=1
        )
        
        self.rf_classifier_gen_to_real.fit(X_train, y_train)
        
        print('Best param (synthetic→real):')
        print(self.rf_classifier_gen_to_real.best_params_)
        
        
        y_train_pred = self.rf_classifier_gen_to_real.predict(X_train)
        self.gen_to_gen_train_accuracy = accuracy_score(y_train, y_train_pred)
    
        print(f"Self-evaluation accuracy for synthetic data: {self.gen_to_gen_train_accuracy:.4f}")
        print('\nSelf-evaluation classification report for synthetic data:')
        print(classification_report(y_train, y_train_pred, target_names=self.category_names))
        
        y_pred = self.rf_classifier_gen_to_real.predict(X_test)
        self.gen_to_real_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"synthetic→real test accuracy: {self.gen_to_real_accuracy:.4f}")
        print(f"synthetic→real classification report:")
        print(classification_report(y_test, y_pred, target_names=self.category_names))
        
        return self.gen_to_real_accuracy
    
    def print_feature_importance(self, top_n=10):
        if self.rf_classifier_real_to_gen is None or self.rf_classifier_gen_to_real is None:
            return None, None
        
        real_to_gen_importance = self.rf_classifier_real_to_gen.best_estimator_.feature_importances_
        gen_to_real_importance = self.rf_classifier_gen_to_real.best_estimator_.feature_importances_
        
        real_to_gen_df = pd.DataFrame({
            'Feature': self.extended_feature_names,
            'Importance': real_to_gen_importance
        }).sort_values('Importance', ascending=False).head(top_n)
        
        gen_to_real_df = pd.DataFrame({
            'Feature': self.extended_feature_names,
            'Importance': gen_to_real_importance
        }).sort_values('Importance', ascending=False).head(top_n)

        print(f"Top {top_n} feature (real→synthetic classifier):")
        for i, row in real_to_gen_df.iterrows():
            print(f"{row['Feature']}: {row['Importance']:.4f}")
        
        print(f"\nTop {top_n} feature (synthetic→real classifier):")
        for i, row in gen_to_real_df.iterrows():
            print(f"{row['Feature']}: {row['Importance']:.4f}")
        
        return real_to_gen_df, gen_to_real_df
    
    def evaluate(self, real_data, real_labels, gen_data, gen_labels):
        print("calculate features...")
        self.compute_features(real_data, real_labels, gen_data, gen_labels)
        
        print("train real→synthetic classifier...")
        self.train_real_to_gen_classifier()
        
        print("train synthetic→real classifier...")
        self.train_gen_to_real_classifier()
        
        return (
            self.rf_classifier_real_to_gen, 
            self.rf_classifier_gen_to_real, 
            self.real_df, 
            self.gen_df
        )