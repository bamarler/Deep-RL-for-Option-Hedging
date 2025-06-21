from MCPG import MCPGAgent
from option_gym import OptionEnv

env = OptionEnv(tickers=['AAPL', 'MSFT', 'IBM', 'JNJ', 'MCD', 
           'KO', 'PG', 'WMT', 'XOM', 'GE', 
           'MMM', 'F', 'T', 'CSCO', 'PFE',
           'INTC', 'BA', 'CAT', 'CVX', 'PEP'], verbose=False)

agent = MCPGAgent()
agent.load_policy('MCPGPolicy.pkl')
agent.train(env, batch_size=250, num_episodes=100000)

agent.save_train_statistics('MCPGTrainStatistics.csv')
agent.save_policy('MCPGPolicy.pkl')

agent.plot_train_statistics('MCPGTrainStatistics.csv')


