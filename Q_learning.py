import time
import pickle
import numpy as np
import gymnasium as gym
from option_gym import OptionEnv

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

env = OptionEnv()

# TODO: Determine how to hash obs and do Q-learning

def hash(obs):
	x,y = obs['player_position']
	h = obs['player_health']
	g = obs['guard_in_cell']
	if not g:
		g = 0
	else:
		g = int(g[-1])

	return x*(5*3*5) + y*(3*5) + h*5 + g

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

	for i in range(375):
		Q_table[i] = np.zeros(6)
		Q_updates[i] = np.zeros(6)

	for i in range(num_episodes):
		if (i + 1) % 1000 == 0:
			print(f"Episode {i + 1}")
		env.reset()

		while not env.is_terminal():
			obs = env.get_observation()
			state = hash(obs)
			if random.random() > epsilon:
				action = np.argmax(Q_table[state])
			elif obs['guard_in_cell']:
				action = random.randint(4, 5)
			else:
				action = random.randint(0, 3)

			obs, reward, done, info = env.step(action)

			Q_updates[state][action] += 1
			eta = 1 / Q_updates[state][action]
			Q_table[state][action] = (1 - eta) * Q_table[state][action] + eta * (reward + gamma * np.max(Q_table[hash(obs)]))

		epsilon *= decay_rate

	return Q_table

decay_rate = 0.99999

Q_table = Q_learning(num_episodes=1000000, gamma=0.9, epsilon=1, decay_rate=decay_rate) # Run Q-learning

# Save the Q-table dict to a file
with open('Q_table.pickle', 'wb') as handle:
    pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)