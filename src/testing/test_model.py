import numpy as np
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from src.models.MCPG import MCPGAgent
from src.models.DDQN import DDQNAgent
from src.environment.option_gym import OptionEnv

# Select Which Tickers to test on from:
# tickers = ['AAPL', 'MSFT', 'IBM', 'JNJ', 'MCD', 'KO', 'PG', 'WMT', 'XOM', 'GE', 'MMM', 'F', 'T', 'CSCO', 'PFE', 'INTC', 'BA', 'CAT', 'CVX', 'PEP']
tickers = ['AAPL', 'MSFT', 'IBM', 'JNJ', 'MCD', 'KO', 'PG', 'WMT', 'XOM', 'GE', 'MMM', 'F', 'T', 'CSCO', 'PFE', 'INTC', 'BA', 'CAT', 'CVX', 'PEP']

# Select Model: MCPG or DDQN
model_type = 'DDQN'

# Choose Policy Filename
policy = 'DDQNPolicy'

# Customize Save File Name
save_file_name = f'{policy}'

# Number of Episodes
num_episodes = 1000

file_path = f'results/data/testing/{model_type}/{save_file_name}.json'
def save_results(results):
    with open(file_path, 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    if model_type == 'MCPG':
        agent = MCPGAgent()
    elif model_type == 'DDQN':
        agent = DDQNAgent()
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    try:
        agent.load_policy(f'policies/{model_type}/{policy}.pkl')
    except FileNotFoundError:
        raise FileNotFoundError(f"Policy file not found: policies/{model_type}/{policy}.pkl")
    
    env = OptionEnv(tickers=tickers, verbose=False)

    results = {
        'returns': [],
        'final_pnls': [],
        'option_payoffs': [],
        'hedging_pnls': [],
        'premiums_paid': [],
        'tickers': [],
        'initial_expiry_days': [],
        'optimal_max_returns': [],
        'optimal_min_returns': []
    }

    save_results(results)

    print(f"Testing {model_type} model on {num_episodes} episodes...")
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        
        initial_expiry = env.time_to_expiry
        initial_investment = env.premium_per_share * env.number_of_shares * env.risk
        
        max_return, min_return = env.compute_optimal_pnls()
        
        trajectory = {
            'positions': [],
            'stock_prices': []
        }
        
        while not done:
            action, _ = agent.select_action(obs, training=False)
            
            position = env.action_space[action]
            stock_price = obs['normalized_stock_price'] * env.strike_price
            
            trajectory['positions'].append(position)
            trajectory['stock_prices'].append(stock_price)
            
            obs, reward, done, truncated, _ = env.step(action)
            done = done or truncated
        
        final_price = trajectory['stock_prices'][-1]
        option_payoff = max(final_price - env.strike_price, 0) * env.number_of_shares
        
        hedging_pnl = 0
        for t in range(len(trajectory['positions']) - 1):
            position = trajectory['positions'][t]
            price_change = trajectory['stock_prices'][t+1] - trajectory['stock_prices'][t]
            hedging_pnl += position * price_change * env.number_of_shares
        
        final_value = option_payoff + hedging_pnl
        normalized_return = (final_value - initial_investment) / initial_investment
        
        results['returns'].append(normalized_return)
        results['final_pnls'].append(final_value - initial_investment)
        results['option_payoffs'].append(option_payoff)
        results['hedging_pnls'].append(hedging_pnl)
        results['premiums_paid'].append(initial_investment)
        results['tickers'].append(env.ticker)
        results['initial_expiry_days'].append(initial_expiry)
        results['optimal_max_returns'].append(max_return)
        results['optimal_min_returns'].append(min_return)
        
        if (episode + 1) % (num_episodes // 10) == 0:
            print(f"Completed {episode + 1}/{num_episodes} episodes and saved results")
            save_results(results)

    save_results(results)
    print(f"Testing completed! Results saved to {file_path}")
    print(f"Total episodes completed: {num_episodes}")
    if len(results['returns']) > 0:
        print(f"Average return: {np.mean(results['returns'])*100:.2f}%")
        print(f"Sharpe ratio: {np.mean(results['returns'])/np.std(results['returns']):.3f}")