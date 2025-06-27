import os
import sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from src.models.MCPG import MCPGAgent
from src.models.DDQN import DDQNAgent
from src.environment.option_gym import OptionEnv

# Select Which Tickers to train on from:
# tickers = ['AAPL', 'MSFT', 'IBM', 'JNJ', 'MCD', 'KO', 'PG', 'WMT', 'XOM', 'GE', 'MMM', 'F', 'T', 'CSCO', 'PFE', 'INTC', 'BA', 'CAT', 'CVX', 'PEP']
tickers = ['AAPL', 'MSFT', 'IBM', 'JNJ', 'MCD', 'KO', 'PG', 'WMT', 'XOM', 'GE', 'MMM', 'F', 'T', 'CSCO', 'PFE', 'INTC', 'BA', 'CAT', 'CVX', 'PEP']

# Select Model: MCPG or DDQN
model_type = 'DDQN'

# Choose Policy Filename
policy = 'DDQNPolicy'

# Customize Save File Name if needed
save_file_name = f'{policy}'

# Number of Episodes
num_episodes = 75000
batch_size = 256
learning_rate = 0.001

# Loss Function (if using MCPG)
loss_function = 'markowitz'

load_previous_policy = True

if model_type == 'MCPG':
    policy_file_path = f'policies/{model_type}/{save_file_name}_{loss_function}.pkl'
    train_statistics_file_path = f'results/data/training/{model_type}/{save_file_name}_{loss_function}.csv'
else:
    policy_file_path = f'policies/{model_type}/{save_file_name}.pkl'
    train_statistics_file_path = f'results/data/training/{model_type}/{save_file_name}.csv'

if __name__ == "__main__":
    env = OptionEnv(tickers=tickers, verbose=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_type == 'MCPG':
        agent = MCPGAgent(risk_aversion=0.65, loss_function=loss_function, learning_rate=learning_rate, device=device)
    elif model_type == 'DDQN':
        agent = DDQNAgent(learning_rate=learning_rate, device=device)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    if load_previous_policy:
        try:
            agent.load_policy(policy_file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Policy file not found: {policy_file_path}")
    
    agent.train(env, policy_file_path, train_statistics_file_path, batch_size=batch_size, num_episodes=num_episodes)
