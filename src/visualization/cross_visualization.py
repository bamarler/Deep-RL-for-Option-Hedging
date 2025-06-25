import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

model_configs = {
    'Sharpe': 'MCPGPolicy_EntropicRisk_Correct.json',
    'Entropic': 'MCPGPolicy_Sharpe_Correct.json',
    'Markowitz': 'MCPGPolicy_Markowitz_Correct.json'
}

model_results = {}
for name, file in model_configs.items():
    with open(f'test_results_mcpg/{file}') as f:
        data = json.load(f)
        model_results[name] = {
            'returns': np.array(data['returns']),
            'sharpe': np.mean(data['returns']) / (np.std(data['returns']) + 1e-8),
            'sortino': np.mean(data['returns']) / (np.std([r for r in data['returns'] if r < 0]) + 1e-8),
            'mean_return': np.mean(data['returns']),
            'std': np.std(data['returns']),
            'optimal_max': np.array(data['optimal_max_returns']),
            'optimal_min': np.array(data['optimal_min_returns'])
        }

plt.figure(figsize=(10, 6))
for name, results in model_results.items():
    plt.hist(results['returns']*100, bins=60, alpha=0.4, label=name, edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', label='Break-even')
plt.xlabel('Return (%)')
plt.ylabel('Frequency')
plt.title('Return Distribution Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

metrics = ['mean_return', 'sharpe', 'sortino']
colors = ['skyblue', 'lightgreen', 'violet']

fig, ax = plt.subplots(1, len(metrics), figsize=(14, 5))

for i, metric in enumerate(metrics):
    values = [model_results[m][metric] for m in model_configs]
    ax[i].bar(model_configs.keys(), values, color=colors[i])
    ax[i].set_title(metric.replace('_', ' ').title())
    if 'return' in metric:
        ax[i].set_ylabel('Return (%)')
        ax[i].set_ylim(-0.5, max(values)*1.2)
    else:
        ax[i].set_ylabel('Ratio')
    for j, val in enumerate(values):
        ax[i].text(j, val + 0.01, f'{val:.2f}', ha='center')

plt.suptitle('Performance Metrics Across Models', fontsize=16)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
window = 400
for name, results in model_results.items():
    rolling = pd.Series(results['returns']).rolling(window)
    sharpe = rolling.mean() / (rolling.std() + 1e-8)
    plt.plot(sharpe, label=f'{name}', linewidth=2)
plt.axhline(y=1, color='green', linestyle='--', alpha=0.6, label='Sharpe = 1')
plt.axhline(y=0, color='red', linestyle='--', alpha=0.6)
plt.title(f'Rolling Sharpe Ratio')
plt.xlabel('Episode')
plt.ylabel('Sharpe Ratio')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
window = 2500
for name, results in model_results.items():
    rolling = pd.Series(results['returns']).rolling(window).mean()
    plt.plot(rolling, label=f'{name}', linewidth=2)
plt.title(f'Rolling Returns')
plt.xlabel('Episode')
plt.ylabel('Returns')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
data = [model_results[name]['returns']*100 for name in model_configs]
plt.boxplot(data, labels=model_configs.keys(), patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='black'),
            medianprops=dict(color='red'))
plt.axhline(y=0, color='gray', linestyle='--')
plt.ylabel('Return (%)')
plt.title('Boxplot of Returns for Each Model')
plt.tight_layout()
plt.show()

print("Model Summary:\n")
for name in model_configs.keys():
    r = model_results[name]
    print(f"Model: {name}")
    print(f"  Mean Return: {r['mean_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {r['sharpe']:.2f}")
    print(f"  Sortino Ratio: {r['sortino']:.2f}")
    print(f"  Std Dev: {r['std']*100:.2f}%")
    print(f"  Capture of Max: {r['mean_return'] / r['optimal_max'].mean() * 100:.1f}%")
    print()