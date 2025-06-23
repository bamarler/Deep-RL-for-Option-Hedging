from MCPG import MCPGAgent
from option_gym import OptionEnv
import json

# Set up environment and agent
env = OptionEnv(tickers=['AAPL', 'MSFT', 'IBM', 'JNJ', 'MCD', 
                        'KO', 'PG', 'WMT', 'XOM', 'GE', 
                        'MMM', 'F', 'T', 'CSCO', 'PFE',
                        'INTC', 'BA', 'CAT', 'CVX', 'PEP'], verbose=False)

policy = 'MCPGPolicy_Sharpe_Correct'
agent = MCPGAgent()
agent.load_policy(f'{policy}.pkl')
# agent.plot_train_statistics('MCPGTrainStatistics_Markowitz.csv')

# Run multiple test episodes
num_episodes = 10000
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

print(f"Running {num_episodes} test episodes...")

for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    
    # Store initial conditions
    initial_expiry = env.time_to_expiry
    initial_investment = env.premium_per_share * env.number_of_shares * env.risk
    
    # Get optimal PNLs for this episode
    max_return, min_return = env.compute_optimal_pnls()
    
    trajectory = {
        'positions': [],
        'stock_prices': []
    }
    
    while not done:
        action, _ = agent.select_action(obs, training=False)
        
        # Store trajectory data
        position = env.action_space[action]
        stock_price = obs['normalized_stock_price'] * env.strike_price
        
        trajectory['positions'].append(position)
        trajectory['stock_prices'].append(stock_price)
        
        obs, reward, done, truncated, _ = env.step(action)
        done = done or truncated
    
    # Calculate final metrics
    final_price = trajectory['stock_prices'][-1]
    option_payoff = max(final_price - env.strike_price, 0) * env.number_of_shares
    
    # Calculate hedging P&L
    hedging_pnl = 0
    for t in range(len(trajectory['positions']) - 1):
        position = trajectory['positions'][t]
        price_change = trajectory['stock_prices'][t+1] - trajectory['stock_prices'][t]
        hedging_pnl += position * price_change * env.number_of_shares
    
    # Calculate normalized return
    final_value = option_payoff + hedging_pnl
    normalized_return = (final_value - initial_investment) / initial_investment
    
    # Store results
    results['returns'].append(normalized_return)
    results['final_pnls'].append(final_value - initial_investment)
    results['option_payoffs'].append(option_payoff)
    results['hedging_pnls'].append(hedging_pnl)
    results['premiums_paid'].append(initial_investment)
    results['tickers'].append(env.ticker)
    results['initial_expiry_days'].append(initial_expiry)
    results['optimal_max_returns'].append(max_return)
    results['optimal_min_returns'].append(min_return)
    
    if (episode + 1) % 100 == 0:
        print(f"Completed {episode + 1}/{num_episodes} episodes")

json.dump(results, open(f"test_results/{policy}.json", 'w'))