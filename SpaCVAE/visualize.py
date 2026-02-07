import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from config import Config
from utils import Utils
import matplotlib.patches as mpatches

class Visualizer:
    def __init__(self, config):
        self.config = config
        self.feature_names = config.FEATURE_NAMES
        self.extended_feature_names = Utils.get_feature_names(self.feature_names)
        self.category_names = config.CATEGORY_NAMES
        self.colors = ['gold', 'royalblue', 'mediumpurple'] 
        self.feature_types = ["cumulative amount", "2020-2030 rate", "2030-2040 rate"]
    
    
    def plot_losses(self, losses, recon_losses, kld_losses):
        plt.figure(figsize=(12, 6))
        
        plt.plot(losses, color='darkred', linewidth=2, label='Total Loss')
        plt.plot(recon_losses, color='navy', linestyle='--', alpha=0.7, label='Recon Loss')
        plt.plot(kld_losses, color='darkgreen', linestyle='-.', alpha=0.7, label='KLD Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Conditional VAE Training Loss Components')  
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    

        
    def visualize_results(self, gen_P1, gen_P2, gen_P3):

        for feature_name in self.feature_names[:3]:  
            for feature_type in self.feature_types: 
                self.plot_feature_boxplot(gen_P1, gen_P2, gen_P3, feature_name, feature_type)
        
        for feature_name in self.feature_names[:3]:  
            self.plot_trend(gen_P1, gen_P2, gen_P3, feature_name)
            
    def visualize_feature_importance(self, importance_df_real_to_gen, importance_df_gen_to_real):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        importance_df_real_to_gen = importance_df_real_to_gen.sort_values('Importance')
        colors1 = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(importance_df_real_to_gen)))
        bars1 = ax1.barh(importance_df_real_to_gen['Feature'], 
                         importance_df_real_to_gen['Importance'], 
                         color=colors1, height=0.7)
        
        ax1.set_title('Feature Importance of Classifier based on Real to Synthetic', fontsize=16)
        ax1.set_xlabel('Importance', fontsize=14)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        
        for bar in bars1:
            width = bar.get_width()
            ax1.text(width * 1.02, bar.get_y() + bar.get_height()/2, 
                     f'{width:.3f}', va='center', fontsize=11)
        
        importance_df_gen_to_real = importance_df_gen_to_real.sort_values('Importance')
        colors2 = plt.cm.GnBu(np.linspace(0.3, 0.9, len(importance_df_gen_to_real)))
        bars2 = ax2.barh(importance_df_gen_to_real['Feature'], 
                         importance_df_gen_to_real['Importance'], 
                         color=colors2, height=0.7)
        
        ax2.set_title('Feature Importance of Classifier based on Synthetic to Real', fontsize=16)
        ax2.set_xlabel('Importance', fontsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width * 1.02, bar.get_y() + bar.get_height()/2, 
                     f'{width:.3f}', va='center', fontsize=11)
        
        plt.tight_layout()
        plt.show()
        
    def list_all_features(self):
        print("All available feature names:")
        for i, name in enumerate(self.feature_names):
            print(f"{i+1}. {name}")
            
        print("\nAvailable feature types:")
        for i, type_name in enumerate(self.feature_types):
            print(f"{i+1}. {type_name}")
               

    def plot_trend_compare_with_shade(self, gen_P1, gen_P2, gen_P3, real_P1, real_P2, real_P3, 
                                    feature_name, gen_shade_colors=None, y_label="Value", 
                                    figsize=(15, 18), sample_line_counts=None, sample_line_colors=None,
                                    legend_positions=None, p_label_positions=None):


        if feature_name not in self.feature_names:
            print(f"Error: feature '{feature_name}' does not exist. Available features: {self.feature_names}")
            return
            
        feature_idx = self.feature_names.index(feature_name)
        
        if gen_shade_colors is None:
            gen_shade_colors = ['gold', 'royalblue', 'mediumpurple']
        if sample_line_counts is None:
            sample_line_counts = [5, 5, 5]
        if sample_line_colors is None:
            sample_line_colors = ['orange', 'blue', 'purple']
        if legend_positions is None:
            legend_positions = [(0.02, 0.98)] * 3
        if p_label_positions is None:
            p_label_positions = [(0.5, 0.9)] * 3
        real_shade_colors = ['#C0C4C3', '#C4CBCF', '#C2CCD0']

        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
        
        time_steps = gen_P1.shape[1]
        time_points = np.arange(2020, 2020 + 5*time_steps, 5)
        selected_years = [2030, 2050, 2100]
        selected_year_indices = {}
        for y in selected_years:
            idx = np.where(time_points == y)[0]
            if len(idx) > 0:
                selected_year_indices[y] = int(idx[0])
        
        category_names = self.category_names
        datasets = [(gen_P1, real_P1), (gen_P2, real_P2), (gen_P3, real_P3)]
        gen_std_list = []
        real_std_list = []
        
        for i, ((gen_data, real_data), category_name) in enumerate(zip(datasets, category_names)):
            ax = axes[i]
            gen_series = gen_data[:, :, feature_idx]
            real_series = real_data[:, :, feature_idx]
            gen_mean = gen_series.mean(axis=0)
            real_mean = real_series.mean(axis=0)
            gen_std  = gen_series.std(axis=0)
            real_std = real_series.std(axis=0)
            gen_std_list.append(gen_std)
            real_std_list.append(real_std)
            
            ax.fill_between(time_points, real_mean - 0.5*real_std, real_mean + 0.5*real_std,
                            color=real_shade_colors[i], alpha=0.4)
            ax.fill_between(time_points, gen_mean - 0.5*gen_std, gen_mean + 0.5*gen_std,
                            color=gen_shade_colors[i], alpha=0.5)

            if sample_line_counts[i] > 0:
                valid = []
                for idx in range(gen_series.shape[0]):
                    s = gen_series[idx]
                    if ((gen_mean - 0.5*gen_std) <= s).all() and (s <= (gen_mean + 0.5*gen_std)).all():
                        valid.append(idx)
                count = min(len(valid), sample_line_counts[i])
                if count > 0:
                    for idx in np.random.choice(valid, count, replace=False):
                        ax.plot(time_points, gen_series[idx], '-', 
                                color=sample_line_colors[i], linewidth=0.5, alpha=0.1)
                if len(valid) < sample_line_counts[i]:
                    print(f"Warning: {category_name} has only {len(valid)} valid samples, less than required {sample_line_counts[i]}")
            
            ax.plot(time_points, gen_mean, '-', color='white', linewidth=3)
            ax.plot(time_points, real_mean, '--', color='#1C2D29', linewidth=1.5, alpha=0.5)
            
            ax.text(*p_label_positions[i], category_name,
                    transform=ax.transAxes, color='white', fontsize=14, fontweight='bold',
                    ha='center', va='center')
            ax.text(*legend_positions[i], f'{category_name}',
                    transform=ax.transAxes, color='black', fontsize=20, fontweight='bold',
                    ha='left', va='top')
            
            ax.grid(False)
            
            ax.set_xticks(time_points)
            ax.set_xticklabels(time_points, rotation=45, fontsize=12)

        fig.text(0.04, 0.5, y_label, va='center', ha='center',
                rotation='vertical', fontsize=12, fontweight='bold')

        fig.text(0.5, 0.04, "Year", ha='center', va='center', fontsize=24, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.09, bottom=0.16)
        plt.savefig(f"./TimeSeries_withShade/Comparison between Synthetic Data and Real Data using {feature_name} with Interval.png", dpi=600, bbox_inches='tight')
        plt.show()

        def build_std_df(std_arrays):
            rows = []
            for arr in std_arrays:
                row = {
                    year: arr[selected_year_indices[year]] if year in selected_year_indices else np.nan
                    for year in selected_years
                }
                row["mean_std_all"] = arr.mean()
                rows.append(row)
            return pd.DataFrame(rows, index=category_names, columns=selected_years + ["mean_std_all"])

        std_df = build_std_df(gen_std_list)
        real_std_df = build_std_df(real_std_list)
        return std_df, real_std_df

    
    
    def plot_feature_boxplot_compare(self, gen_P1, gen_P2, gen_P3, real_P1, real_P2, real_P3, 
                                    feature_name, feature_type, p_colors=None, figsize=(18, 10),
                                    box_width=0.6, legend_position=(0.02, 0.98), 
                                    ar6_label_positions=None, synthetic_label_positions=None):

        if feature_name not in self.feature_names:
            print(f"Error: feature '{feature_name}' does not exist. Available features: {self.feature_names}")
            return
            
        if feature_type not in self.feature_types:
            print(f"Error: feature type '{feature_type}' does not exist. Available types: {self.feature_types}")
            return
            
        base_feature_idx = self.feature_names.index(feature_name)
        feature_type_idx = self.feature_types.index(feature_type)
        
        feature_idx = base_feature_idx * 3 + feature_type_idx
        
        if p_colors is None:
            p_colors = ['gold', 'royalblue', 'mediumpurple']
        
        plt.figure(figsize=figsize)
        
        gen_P1_features = Utils.compute_enriched_features(gen_P1)[:, feature_idx]
        gen_P2_features = Utils.compute_enriched_features(gen_P2)[:, feature_idx]
        gen_P3_features = Utils.compute_enriched_features(gen_P3)[:, feature_idx]
        
        real_P1_features = Utils.compute_enriched_features(real_P1)[:, feature_idx]
        real_P2_features = Utils.compute_enriched_features(real_P2)[:, feature_idx]
        real_P3_features = Utils.compute_enriched_features(real_P3)[:, feature_idx]
        
        gen_data = [gen_P1_features, gen_P2_features, gen_P3_features]
        real_data = [real_P1_features, real_P2_features, real_P3_features]
            
        real_box_positions = [2, 7, 12]         
        real_scatter_positions = [3, 8, 13]       
        gen_box_positions = [4, 9, 14]           
        gen_scatter_positions = [5, 10, 15]  
        
        ar6_x_positions = [(real_box_positions[i] + real_scatter_positions[i]) / 2 
                        for i in range(3)]      
        synthetic_x_positions = [(gen_box_positions[i] + gen_scatter_positions[i]) / 2 
                            for i in range(3)]  
        
        labels = ["P1", "P2", "P3"]
        
        real_boxes = plt.boxplot(
            real_data, 
            positions=real_box_positions, 
            widths=box_width, 
            patch_artist=True,
            showfliers=False,
            medianprops={'linewidth': 1.5},  
            boxprops={'facecolor': 'none', 'linewidth': 1.5}, 
            whiskerprops={'linewidth': 1.0},
            capprops={'linewidth': 1.0}
        )
        
        for i, (patch, color) in enumerate(zip(real_boxes['boxes'], p_colors)):
            patch.set_facecolor('none')      
            patch.set_edgecolor(color)    
            patch.set_linewidth(2.5)       
            patch.set_alpha(1.0)              
            
            real_boxes['medians'][i].set_color(color)
            real_boxes['medians'][i].set_linewidth(1.5)
            
            real_boxes['whiskers'][i*2].set_color(color)
            real_boxes['whiskers'][i*2+1].set_color(color)
            real_boxes['whiskers'][i*2].set_linewidth(1.0)
            real_boxes['whiskers'][i*2+1].set_linewidth(1.0)
            
            real_boxes['caps'][i*2].set_color(color)
            real_boxes['caps'][i*2+1].set_color(color)
            real_boxes['caps'][i*2].set_linewidth(1.0)
            real_boxes['caps'][i*2+1].set_linewidth(1.0)
        
        gen_boxes = plt.boxplot(
            gen_data, 
            positions=gen_box_positions, 
            widths=box_width, 
            patch_artist=True,
            showfliers=False,
            medianprops={'color': 'black', 'linewidth': 1},
            boxprops={'alpha': 1, 'edgecolor': 'black', 'linewidth': 0.5},
            whiskerprops={'linewidth': 0.5, 'color': 'black'},
            capprops={'linewidth': 0.5, 'color': 'black'}
        )
        

        for patch, color in zip(gen_boxes['boxes'], p_colors):
            patch.set_facecolor(color)        
            patch.set_edgecolor('black') 
            patch.set_linewidth(1.0)        
            patch.set_alpha(1)          

        for i, (real_feature, gen_feature) in enumerate(zip(real_data, gen_data)):

            x_real = np.random.normal(real_scatter_positions[i], 0.08, size=len(real_feature))
            plt.scatter(x_real, real_feature, color=p_colors[i], s=1, alpha=0.4, 
                    edgecolor=p_colors[i], linewidth=0)
            
            x_gen = np.random.normal(gen_scatter_positions[i], 0.08, size=len(gen_feature))
            plt.scatter(x_gen, gen_feature, color=p_colors[i], s=1, alpha=0.4, 
                    edgecolor='black', linewidth=0)
        
        if ar6_label_positions is not None:
            for i, y_pos in enumerate(ar6_label_positions):
                plt.text(ar6_x_positions[i], y_pos, 'AR6', rotation=90, 
                        ha='center', va='center', fontsize=10, color='black', fontweight='bold')
        else:
            all_data_combined = np.concatenate(gen_data + real_data)
            default_y = (np.min(all_data_combined) + np.max(all_data_combined)) / 2
            for i in range(3):
                plt.text(ar6_x_positions[i], default_y, 'AR6', rotation=90, 
                        ha='center', va='center', fontsize=10, color='black', fontweight='bold')
        
        if synthetic_label_positions is not None:
            for i, y_pos in enumerate(synthetic_label_positions):
                plt.text(synthetic_x_positions[i], y_pos, 'Synthetic', rotation=90, 
                        ha='center', va='center', fontsize=10, color='black', fontweight='bold')
        else:
            all_data_combined = np.concatenate(gen_data + real_data)
            default_y = (np.min(all_data_combined) + np.max(all_data_combined)) / 2
            for i in range(3):
                plt.text(synthetic_x_positions[i], default_y, 'Synthetic', rotation=90, 
                        ha='center', va='center', fontsize=10, color='black', fontweight='bold')
        
        group_centers = [2.5, 7.5, 12.5]  
        plt.xticks(group_centers, labels, fontsize=14)
        
        plt.text(legend_position[0], legend_position[1], feature_type, 
                transform=plt.gca().transAxes, fontsize=14, fontweight='bold',
                verticalalignment='top', horizontalalignment='left', color='black')
        
        plt.xlabel('Policy Category', fontsize=16)
        
        all_data = np.concatenate(gen_data + real_data)
        y_min = np.min(all_data)
        y_max = np.max(all_data)
        y_range = y_max - y_min
        plt.ylim([y_min - y_range * 0.1, y_max + y_range * 0.1])
        plt.xlim([0.5, 16.5]) 
        plt.tight_layout()
        plt.savefig(f"./Boxplot_betweenP/{feature_name}/Box Plot of Policy Category for {feature_name} {feature_type}.png", dpi=600, bbox_inches='tight')
        plt.show()
    
    
    def plot_boxplot_compare(self, gen_P1, gen_P2, gen_P3, real_P1, real_P2, real_P3, 
                        feature_name, year, p_colors=None, figsize=(18, 10),
                        box_width=0.6, legend_position=(0.02, 0.98), 
                        ar6_label_positions=None, synthetic_label_positions=None):

        if feature_name not in self.feature_names:
            print(f"Error: feature '{feature_name}' doesn't exist. Available features: {self.feature_names}")
            return
        
        feature_idx = self.feature_names.index(feature_name)
        
        time_steps = gen_P1.shape[1]
        time_points = np.arange(2020, 2020 + 5*time_steps, 5)
        
        if year not in time_points:
            print(f"Error: year {year} is not in the valid range. Available years: {list(time_points)}")
            return
        
        year_idx = (year - 2020) // 5
        
        if p_colors is None:
            p_colors = ['gold', 'royalblue', 'mediumpurple']
        
        plt.figure(figsize=figsize)
        
        gen_P1_features = gen_P1[:, year_idx, feature_idx]
        gen_P2_features = gen_P2[:, year_idx, feature_idx]
        gen_P3_features = gen_P3[:, year_idx, feature_idx]
        
        real_P1_features = real_P1[:, year_idx, feature_idx]
        real_P2_features = real_P2[:, year_idx, feature_idx]
        real_P3_features = real_P3[:, year_idx, feature_idx]
        
        gen_data = [gen_P1_features, gen_P2_features, gen_P3_features]
        real_data = [real_P1_features, real_P2_features, real_P3_features]
           
        real_box_positions = [2, 7, 12]         
        real_scatter_positions = [3, 8, 13]     
        gen_box_positions = [4, 9, 14]          
        gen_scatter_positions = [5, 10, 15] 
        
        ar6_x_positions = [(real_box_positions[i] + real_scatter_positions[i]) / 2 
                        for i in range(3)]     
        synthetic_x_positions = [(gen_box_positions[i] + gen_scatter_positions[i]) / 2 
                            for i in range(3)]  
        
        labels = ["P1", "P2", "P3"]
        
        real_boxes = plt.boxplot(
            real_data, 
            positions=real_box_positions, 
            widths=box_width, 
            patch_artist=True,
            showfliers=False,
            medianprops={'linewidth': 1.5}, 
            boxprops={'facecolor': 'none', 'linewidth': 1.5}, 
            whiskerprops={'linewidth': 4.0},
            capprops={'linewidth': 4.0}
        )
        

        for i, (patch, color) in enumerate(zip(real_boxes['boxes'], p_colors)):

            patch.set_facecolor('none')       
            patch.set_edgecolor(color)       
            patch.set_linewidth(6)          
            patch.set_alpha(1.0)       
            
            real_boxes['medians'][i].set_color(color)
            real_boxes['medians'][i].set_linewidth(3)
            
            real_boxes['whiskers'][i*2].set_color(color)
            real_boxes['whiskers'][i*2+1].set_color(color)
            real_boxes['whiskers'][i*2].set_linewidth(4.0)
            real_boxes['whiskers'][i*2+1].set_linewidth(4.0)
            
            real_boxes['caps'][i*2].set_color(color)
            real_boxes['caps'][i*2+1].set_color(color)
            real_boxes['caps'][i*2].set_linewidth(4.0)
            real_boxes['caps'][i*2+1].set_linewidth(4.0)
            
        gen_boxes = plt.boxplot(
            gen_data, 
            positions=gen_box_positions, 
            widths=box_width, 
            patch_artist=True,
            showfliers=False,
            medianprops={'color': 'black', 'linewidth': 3},
            boxprops={'alpha': 1, 'edgecolor': 'black', 'linewidth': 0.5},
            whiskerprops={'linewidth': 3, 'color': 'black'},
            capprops={'linewidth': 3, 'color': 'black'}
        )
        

        for patch, color in zip(gen_boxes['boxes'], p_colors):
            patch.set_facecolor(color)        
            patch.set_edgecolor('black')     
            patch.set_linewidth(1.0)         
            patch.set_alpha(1)           
        

        for i, (real_feature, gen_feature) in enumerate(zip(real_data, gen_data)):
            x_real = np.random.normal(real_scatter_positions[i], 0.08, size=len(real_feature))
            plt.scatter(x_real, real_feature, color=p_colors[i], s=2, alpha=0.8, 
                    edgecolor=p_colors[i], linewidth=0)
            
            x_gen = np.random.normal(gen_scatter_positions[i], 0.08, size=len(gen_feature))
            plt.scatter(x_gen, gen_feature, color=p_colors[i], s=2, alpha=0.8, 
                    edgecolor='black', linewidth=0)
        
        if ar6_label_positions is not None:
            for i, y_pos in enumerate(ar6_label_positions):
                plt.text(ar6_x_positions[i], y_pos, 'AR6', rotation=90, 
                        ha='center', va='center', fontsize=28, color='black')
        else:
            all_data_combined = np.concatenate(gen_data + real_data)
            default_y = (np.min(all_data_combined) + np.max(all_data_combined)) / 2
            for i in range(3):
                plt.text(ar6_x_positions[i], default_y, 'AR6', rotation=90, 
                        ha='center', va='center', fontsize=28, color='black')
        
        if synthetic_label_positions is not None:
            for i, y_pos in enumerate(synthetic_label_positions):
                plt.text(synthetic_x_positions[i], y_pos, 'Synthetic', rotation=90, 
                        ha='center', va='center', fontsize=28, color='black')
        else:
            all_data_combined = np.concatenate(gen_data + real_data)
            default_y = (np.min(all_data_combined) + np.max(all_data_combined)) / 2
            for i in range(3):
                plt.text(synthetic_x_positions[i], default_y, 'Synthetic', rotation=90, 
                        ha='center', va='center', fontsize=28, color='black')
        
        group_centers = [2.5, 7.5, 12.5]  
        plt.xticks(group_centers, labels, fontsize=28)
        
        plt.text(legend_position[0], legend_position[1], f'{year}', 
                transform=plt.gca().transAxes, fontsize=40, fontweight='bold',
                verticalalignment='top', horizontalalignment='left', color='black')
        
        plt.xlabel('Policy Category', fontsize=32)
        
        all_data = np.concatenate(gen_data + real_data)
        y_min = np.min(all_data)
        y_max = np.max(all_data)
        y_range = y_max - y_min
        plt.ylim([y_min - y_range * 0.1, y_max + y_range * 0.1])
        plt.xlim([0.5, 16.5]) 
        plt.yticks(fontsize=18) 

        ax = plt.gca() 
        ax.spines['top'].set_visible(False)    
        ax.spines['right'].set_visible(False)    
        plt.tight_layout()
        plt.savefig(f"./Boxplot_betweenP/{feature_name}/Box Plot of Policy Category for {feature_name} Year {year}.png", dpi=600, bbox_inches='tight')
        plt.show()

