from abc import ABC, abstractmethod
import torch

class Agent(ABC):    
    @abstractmethod
    def train(self, env, policy_file_path, train_statistics_file_path, batch_size, num_episodes):
        pass
    
    @abstractmethod
    def save_policy(self, policy_file_path):
        pass
    
    @abstractmethod
    def load_policy(self, policy_file_path):
        pass
    
    @abstractmethod
    def save_train_statistics(self, train_statistics_file_path):
        pass

    def compute_terminal_pnl(self, trajectory, env):
        """Calculate terminal P&L for a complete episode from BUYER's perspective"""
        
        # Final stock price
        final_price = trajectory[-1]['stock_price']
        
        # Option payoff at expiry (for put option) - BUYER receives this
        option_payoff = max(final_price - env.strike_price, 0)
        
        # Trading P&L from hedging
        trading_pnl = 0
        for t in range(len(trajectory) - 1):
            position = trajectory[t]['position']
            price_change = trajectory[t+1]['stock_price'] - trajectory[t]['stock_price']
            trading_pnl += position * price_change * env.number_of_shares
        
        # Initial investment (premium paid by buyer)
        initial_investment = env.premium_per_share * env.number_of_shares * env.risk
        
        # Final portfolio value = option payoff + trading P&L
        final_value = option_payoff * env.number_of_shares + trading_pnl
        
        # Normalized return: (final - initial) / initial
        normalized_return = (final_value - initial_investment) / initial_investment
        
        return torch.tensor(normalized_return, dtype=torch.float32, requires_grad=True)