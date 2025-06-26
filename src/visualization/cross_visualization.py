import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

# Configuration
title = 'Cross-Model Option Hedging Comparison'

# Array of models to compare
model_configs = [
    {'model_type': 'DDQN', 'policy': 'DDQNPolicy_new', 'label': 'DDQN', 'color': '#8B5CF6'},  # Vivid violet
    {'model_type': 'MCPG', 'policy': 'MCPGPolicy_sharpe_100k', 'label': 'MCPG-Sharpe', 'color': '#EC4899'},  # Hot pink
    {'model_type': 'MCPG', 'policy': 'MCPGPolicy_entropic_100k', 'label': 'MCPG-Entropic', 'color': '#10B981'},  # Emerald green
    {'model_type': 'MCPG', 'policy': 'MCPGPolicy_markowitz_100k', 'label': 'MCPG-Markowitz', 'color': '#F59E0B'},  # Amber orange
]

def load_results(config):
    """Load results for a given model configuration."""
    file_path = f"results/data/testing/{config['model_type']}/{config['policy']}.json"
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found - {file_path}")
        return None

def calculate_metrics(results):
    """Calculate performance metrics for a model."""
    returns = np.array(results['returns'])
    
    # Basic metrics
    metrics = {
        'mean_return': returns.mean() * 100,
        'median_return': np.median(returns) * 100,
        'std_dev': returns.std() * 100,
        'sharpe': returns.mean() / (returns.std() + 1e-8),
        'win_rate': (returns > 0).mean() * 100,
        'best_return': returns.max() * 100,
        'worst_return': returns.min() * 100,
    }
    
    # Sortino ratio
    downside_std = returns[returns < 0].std() if (returns < 0).any() else 1e-8
    metrics['sortino'] = returns.mean() / downside_std
    
    # Max drawdown
    cumsum = pd.Series(returns).cumsum()
    cummax = cumsum.cummax()
    metrics['max_drawdown'] = (cumsum - cummax).min() * 100
    
    # Average win/loss
    metrics['avg_win'] = returns[returns > 0].mean() * 100 if (returns > 0).any() else 0
    metrics['avg_loss'] = returns[returns < 0].mean() * 100 if (returns < 0).any() else 0
    
    # Capture percent (if optimal data available)
    if 'optimal_max_returns' in results:
        optimal_max = np.array(results['optimal_max_returns'])
        metrics['capture_percent'] = returns.mean() / optimal_max.mean() * 100
    else:
        metrics['capture_percent'] = np.nan
    
    return metrics

def plot_comparison(model_configs):
    """Create comparison plots for multiple models."""
    # Load all results
    all_results = []
    for config in model_configs:
        results = load_results(config)
        if results:
            all_results.append({
                'config': config,
                'results': results,
                'returns': np.array(results['returns']),
                'metrics': calculate_metrics(results)
            })
    
    if not all_results:
        print("No valid results found!")
        return
    
    # Create 2x3 figure
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Overlayed Returns Distribution (1, 1)
    ax1 = plt.subplot(2, 3, 1)
    for data in all_results:
        returns_pct = data['returns'] * 100
        plt.hist(returns_pct, bins=50, alpha=0.5, 
                label=data['config']['label'], 
                color=data['config']['color'],
                edgecolor='black', linewidth=0.5)
    
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Break-even')
    plt.xlabel('Return (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Returns')
    plt.xlim(-4000, 6000)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Returns vs Episodes (1, 2)
    ax2 = plt.subplot(2, 3, 2)
    window = 1000
    for data in all_results:
        returns_pct = data['returns'] * 100
        # Plot moving average for clarity
        returns_ma = pd.Series(returns_pct).rolling(window).mean()
        episodes = np.arange(len(returns_pct))
        plt.plot(episodes, returns_ma, 
                label=f"{data['config']['label']}",
                color=data['config']['color'],
                linewidth=0.5)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Return (%)')
    plt.title(f'Returns Over Episodes ({window}-episode window)')
    plt.legend()
    plt.ylim(200, 500)
    plt.grid(True, alpha=0.3)
    
    # 3. Sharpe Ratio vs Episodes (1, 3)
    ax3 = plt.subplot(2, 3, 3)
    window = 500
    for data in all_results:
        returns = data['returns']
        # Calculate rolling Sharpe
        rolling_returns = pd.Series(returns)
        rolling_mean = rolling_returns.rolling(window).mean()
        rolling_std = rolling_returns.rolling(window).std()
        rolling_sharpe = rolling_mean / (rolling_std + 1e-8)
        
        plt.plot(rolling_sharpe.dropna(), 
                label=data['config']['label'],
                color=data['config']['color'],
                linewidth=0.5)
    
    # plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    # plt.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')
    plt.xlabel('Episode')
    plt.ylabel('Rolling Sharpe Ratio')
    plt.title(f'Rolling Sharpe Ratio ({window}-episode window)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # 4. Hedging P&L Comparison (2, 1)
    ax4 = plt.subplot(2, 3, 4)
    hedging_pnl_pcts = []
    labels = []
    
    for data in all_results:
        results = data['results']
        hedging_pnl_pct = np.mean(results['hedging_pnls']) / np.mean(results['premiums_paid']) * 100
        hedging_pnl_pcts.append(hedging_pnl_pct)
        labels.append(data['config']['label'])
    
    x = np.arange(len(labels))
    colors = [data['config']['color'] for data in all_results]
    
    bars = plt.bar(x, hedging_pnl_pcts, color=colors, alpha=0.7, label=labels)
    plt.ylabel('Hedging P&L (% of Premium)')
    plt.title('Hedging P&L Comparison')
    plt.xticks([])  # Remove x-axis ticks
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.ylim(0, 120)
    
    # Add value labels on bars
    for bar, val in zip(bars, hedging_pnl_pcts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + np.sign(val)*1,
                f'{val:.1f}%', ha='center', va='bottom' if val > 0 else 'top')
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, label=label) 
                      for color, label in zip(colors, labels)]
    plt.legend(handles=legend_elements, loc='best')
    
    # 5. Standard Deviation Comparison (2, 2) - NEW PLOT
    ax5 = plt.subplot(2, 3, 5)
    std_devs = []
    for data in all_results:
        std_devs.append(data['metrics']['std_dev'])
    
    bars = plt.bar(x, std_devs, color=colors, alpha=0.7)
    plt.ylabel('Standard Deviation (%)')
    plt.title('Returns Standard Deviation Comparison')
    plt.ylim(0, 1300)
    plt.xticks([])  # Remove x-axis ticks
    
    # Add value labels on bars
    for bar, val in zip(bars, std_devs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom')

    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, label=label) 
                      for color, label in zip(colors, labels)]
    plt.legend(handles=legend_elements, loc='best')
    
    # 6. Performance Metrics Table (2, 3)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Prepare table data
    row_labels = [
        'Mean Return (%)',
        'Median Return (%)',
        'Std Deviation (%)',
        'Sharpe Ratio',
        'Sortino Ratio',
        'Max Drawdown (%)',
        'Win Rate (%)',
        'Avg Win (%)',
        'Avg Loss (%)',
        'Best Return (%)',
        'Worst Return (%)',
        'Capture Percent (%)'
    ]
    
    # Create columns for each model
    columns = ['Metric'] + [data['config']['label'] for data in all_results]
    
    # Build table data
    table_data = []
    for row_label in row_labels:
        row = [row_label]
        for data in all_results:
            metrics = data['metrics']
            # Map row labels to metric keys
            metric_map = {
                'Mean Return (%)': 'mean_return',
                'Median Return (%)': 'median_return',
                'Std Deviation (%)': 'std_dev',
                'Sharpe Ratio': 'sharpe',
                'Sortino Ratio': 'sortino',
                'Max Drawdown (%)': 'max_drawdown',
                'Win Rate (%)': 'win_rate',
                'Avg Win (%)': 'avg_win',
                'Avg Loss (%)': 'avg_loss',
                'Best Return (%)': 'best_return',
                'Worst Return (%)': 'worst_return',
                'Capture Percent (%)': 'capture_percent'
            }
            
            key = metric_map.get(row_label)
            if key:
                value = metrics[key]
                if key in ['sharpe', 'sortino']:
                    row.append(f'{value:.2f}')
                elif np.isnan(value):
                    row.append('N/A')
                else:
                    row.append(f'{value:.1f}')
            else:
                row.append('N/A')
        
        table_data.append(row)
    
    # Create table
    table = ax6.table(cellText=table_data,
                     colLabels=columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.5] + [0.15] * len(all_results))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style the table
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            if j == 0:  # Row labels
                table[(i, j)].set_facecolor('#e6e6e6')
                table[(i, j)].set_text_props(weight='bold')
            else:
                table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    plt.suptitle(title, fontsize=16, y=0.98)
    # plt.tight_layout()

    plt.subplots_adjust(left=0.048, bottom=0.03, right=0.975, top=0.92, wspace=0.2, hspace=0.269)
    
    # Save figure
    save_dir = 'results/images/testing/comparison'
    os.makedirs(save_dir, exist_ok=True)
    
    fig.savefig(f'{save_dir}/cross_model_enhanced_plots.png', dpi=300, bbox_inches='tight')
    print(f"Saved enhanced plots to {save_dir}/")
    
    plt.show()

if __name__ == "__main__":
    plot_comparison(model_configs)