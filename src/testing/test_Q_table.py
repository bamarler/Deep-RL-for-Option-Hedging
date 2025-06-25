import pickle
import numpy as np
from Q_learning import hash
from ..option_gym import OptionEnv

with open('QLearning/Q_table.pickle', 'rb') as handle:
    Q_table = pickle.load(handle)

env = OptionEnv(tickers=['AAPL', 'MSFT', 'IBM', 'JNJ', 'MCD', 
           'KO', 'PG', 'WMT', 'XOM', 'GE', 
           'MMM', 'F', 'T', 'CSCO', 'PFE',
           'INTC', 'BA', 'CAT', 'CVX', 'PEP'])

obs, _ = env.reset()

while not env.done:
    state = hash(obs)
    action = np.argmax(Q_table[state])
    obs, reward, done, truncated, _ = env.step(action)