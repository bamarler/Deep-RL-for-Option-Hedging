import time
import pickle
import numpy as np
import gymnasium as gym
from option_gym import OptionEnv
import random

env = OptionEnv(tickers=['AAPL', 'MSFT', 'IBM', 'JNJ', 'MCD', 
           'KO', 'PG', 'WMT', 'XOM', 'GE', 
           'MMM', 'F', 'T', 'CSCO', 'PFE',
           'INTC', 'BA', 'CAT', 'CVX', 'PEP'], verbose=False)

# TODO: Determine how to hash obs and do Q-learning

def hash(obs : dict):
	# Discrete
	position = obs['position']
	time_to_expiry = obs['time_to_expiry']
	delta = np.round(obs['delta'], 2)
	volatility = np.round(obs['volatility'], 2)

	hash = position * (10**8) + delta * (10**6) + volatility * (10**4) + time_to_expiry

	# Continuous
	normalized_stock_price = obs['normalized_stock_price']
	normalized_portfolio_value = obs['normalized_portfolio_value']
	gamma = obs['gamma']

	return hash + gamma * (10**17) + normalized_portfolio_value * (10**16) + normalized_stock_price * (10**11)

def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=0.999):
	"""
	Run Q-learning algorithm for a specified number of episodes.

    Parameters:
    - num_episodes (int): Number of episodes to run.
    - gamma (float): Discount factor.
    - epsilon (float): Exploration rate.
    - decay_rate (float): Rate at which epsilon decays. Epsilon is decayed as epsilon = epsilon * decay_rate after each episode.

    Returns:
    - Q_table (dict): Dictionary containing the Q-values for each state-action pair.
    """
	Q_table = {}
	Q_updates = {}

	print("Started Learning")
	for i in range(num_episodes):
		if (i + 1) % 100 == 0:
			print(f"Episode {i + 1}")
		obs, info = env.reset()

		if Q_table.get(hash(obs)) is None:
			Q_table[hash(obs)] = np.zeros(len(env.action_space))
			Q_updates[hash(obs)] = np.zeros(len(env.action_space))

		done = False
		truncated = False
		while not done and not truncated:
			state = hash(obs)
			
			if random.random() > epsilon:
				action = np.argmax(Q_table[state])
			else:
				action = random.randint(0, len(env.action_space) - 1)

			obs, reward, done, truncated, info = env.step(action)

			next_state = hash(obs)
			if Q_table.get(next_state) is None:
				Q_table[next_state] = np.zeros(len(env.action_space))
				Q_updates[next_state] = np.zeros(len(env.action_space))

			Q_updates[state][action] += 1
			eta = 1 / Q_updates[state][action]
			Q_table[state][action] = (1 - eta) * Q_table[state][action] + eta * (reward + gamma * np.max(Q_table.get(next_state)))

		epsilon *= decay_rate

		print(info)
	return Q_table


if __name__ == "__main__":
	Q_table = Q_learning(num_episodes=1000000, gamma=0.9, epsilon=1, decay_rate=0.99999) # Run Q-learning
	# Q_table = Q_learning()

	# Save the Q-table dict to a file
	with open('QLearning/Q_table.pickle', 'wb') as handle:
		pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)