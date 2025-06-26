import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os

# Choose name for plots
name = 'MCPG Sharpe Original'

# Choose Policy Filename
policy = 'MCPGPolicy'

# Loss Function (if using MCPG)
loss_function = 'markowitz'

data_file_path = f'results/data/training/MCPG/{policy}_{loss_function}_direct.csv'
image_file_path = f'results/images/training/MCPG/{policy}_{loss_function}_direct.png'

def visualize_training_statistics(train_statistics, name, save_path=None):
    """
    Visualize training statistics with epoch on the x-axis.
    
    Parameters:
    train_statistics (pd.DataFrame): DataFrame containing training metrics
        - epoch: epoch numbers
        - avg_return: average returns
        - entropic: entropic values
        - sharpe: Sharpe ratios
        - markowitz: Markowitz values
    save_path (str): Path to save the image (optional)
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{name} Training Metrics Over Epochs', fontsize=16)
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Define metrics to plot (excluding epoch)
    metrics = ['avg_return', 'entropic', 'sharpe', 'markowitz']
    metric_names = ['Average Return', 'Entropic Risk', 'Sharpe Ratio', 'Markowitz Criterion']
    colors = ['blue', 'green', 'red', 'purple']
    
    # Plot each metric
    for idx, (metric, name, color) in enumerate(zip(metrics, metric_names, colors)):
        ax = axes[idx]
        
        if 'epoch' in train_statistics.columns and metric in train_statistics.columns:
            # Check if data is not empty
            if not train_statistics[metric].isna().all():
                ax.plot(train_statistics['epoch'], 
                       train_statistics[metric], 
                       color=color, 
                       linewidth=2, 
                       marker='o', 
                       markersize=5,
                       alpha=0.7,
                       label=name)
                
                # Add grid
                ax.grid(True, alpha=0.3)
                
                # Set labels
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel(name, fontsize=12)
                ax.set_title(f'{name} vs Epoch', fontsize=14)
                
                # Add trend line if enough data points
                valid_data = train_statistics.dropna(subset=['epoch', metric])
                if len(valid_data) > 3:
                    z = np.polyfit(valid_data['epoch'], valid_data[metric], 1)
                    p = np.poly1d(z)
                    ax.plot(valid_data['epoch'], 
                           p(valid_data['epoch']), 
                           "--", 
                           color=color, 
                           alpha=0.5,
                           label='Trend')
                
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', 
                       transform=ax.transAxes)
                ax.set_title(f'{name} vs Epoch', fontsize=14)
        else:
            ax.text(0.5, 0.5, f'Column "{metric}" not found', 
                   ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(f'{name} vs Epoch', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Read the CSV file
    train_statistics = pd.read_csv(data_file_path)
    
    # Create individual plots with specified save path
    visualize_training_statistics(train_statistics, name, save_path=image_file_path)