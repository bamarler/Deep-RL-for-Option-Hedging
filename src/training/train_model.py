import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from src.models.MCPG import MCPGAgent
from src.models.DDQN import DDQNAgent
from src.environment.option_gym import OptionEnv

# Select Which Tickers to train on from:
# tickers = ['AAPL', 'MSFT', 'IBM', 'JNJ', 'MCD', 'KO', 'PG', 'WMT', 'XOM', 'GE', 'MMM', 'F', 'T', 'CSCO', 'PFE', 'INTC', 'BA', 'CAT', 'CVX', 'PEP']
tickers = ['AAPL', 'MSFT', 'IBM', 'JNJ', 'MCD', 'KO', 'PG', 'WMT', 'XOM', 'GE', 'MMM', 'F', 'T', 'CSCO', 'PFE', 'INTC', 'BA', 'CAT', 'CVX', 'PEP']

# Select Model: MCPG or DDQN
model_type = 'MCPG'

# Choose Policy Filename
policy = 'MCPGPolicy_EntropicRisk'

# Customize Save File Name if needed
save_file_name = f'{policy}'

# Number of Episodes
num_episodes = 10000
batch_size = 250

policy_file_path = f'results/data/{model_type}/{save_file_name}.pkl'
train_statistics_file_path = f'results/data/{model_type}/{save_file_name}.csv'


if __name__ == "__main__":
    env = OptionEnv(tickers=tickers, verbose=False)

    if model_type == 'MCPG':
        agent = MCPGAgent(risk_aversion=0.65)
    elif model_type == 'DDQN':
        agent = DDQNAgent()
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    try:
        agent.load_policy(policy_file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Policy file not found: {policy_file_path}")
    
    agent.train(env, policy_file_path, train_statistics_file_path, batch_size=batch_size, num_episodes=num_episodes)
