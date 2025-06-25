from MCPG import MCPGAgent
from option_gym import OptionEnv

env = OptionEnv(tickers=['AAPL', 'MSFT', 'IBM', 'JNJ', 'MCD', 
           'KO', 'PG', 'WMT', 'XOM', 'GE', 
           'MMM', 'F', 'T', 'CSCO', 'PFE',
           'INTC', 'BA', 'CAT', 'CVX', 'PEP'], verbose=False)

agent = MCPGAgent(risk_aversion=0.65)
agent.load_policy('MCPG_EntropicRisk_Scratch.pkl')
# agent.train(env, batch_size=250, num_episodes=100000) # 400

agent.train(env, batch_size=250, num_episodes=7500) # 30

agent.save_train_statistics('MCPGTrainStatistics_EntropicRisk_Scratch.csv')
agent.save_policy('MCPG_EntropicRisk_Scratch.pkl')

agent.plot_train_statistics('MCPGTrainStatistics_EntropicRisk_Scratch.csv')
