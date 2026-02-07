import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

    # 1*3
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
        

    time_steps = gen_P1.shape[1]
    time_points = np.arange(2020, 2020 + 5*time_steps, 5)
        
    category_names = ['P1', 'P2', 'P3']
    datasets = [(gen_P1, real_P1), (gen_P2, real_P2), (gen_P3, real_P3)]
        
    for i, ((gen_data, real_data), category_name) in enumerate(zip(datasets, category_names)):
        ax = axes[i]
        gen_series = gen_data[:, :, feature_idx]
        real_series = real_data[:, :, feature_idx]
        gen_mean = gen_series.mean(axis=0)
        real_mean = real_series.mean(axis=0)
        gen_std  = gen_series.std(axis=0)
        real_std = real_series.std(axis=0)
            
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
                print(f"Caution: The number of satisfied samples in {category_name} ({len(valid)}) is less than required ({sample_line_counts[i]})")
            
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