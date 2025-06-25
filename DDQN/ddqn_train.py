import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque
import json
from option_gym import OptionEnv  

class DDQNNetwork(nn.Module):
    def __init__(self, num_actions=51):
        super().__init__()
        
        input_dim = 7
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        
        
        
    def forward(self, obs):
        if isinstance(obs, dict):
            features = torch.tensor([
                obs['position'],
                obs['normalized_stock_price'],
                obs['time_to_expiry'],
                obs['normalized_portfolio_value'],
                obs['delta'],
                obs['gamma'],
                obs['volatility'],
            ], dtype=torch.float32)
        else:
            features = obs
        
        q_values = self.network(features)


        return q_values

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DDQNAgent:
    def __init__(self, num_actions=51, learning_rate=0.001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 target_update_freq=1000, buffer_size=100000):
        
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # Networks
        self.q_network = DDQNNetwork(num_actions)
        self.target_network = DDQNNetwork(num_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = AdamW(self.q_network.parameters(), lr=learning_rate, weight_decay=0.0001)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training statistics
        self.train_statistics = {
            'episode': [],
            'avg_return': [],
            'sharpe': [],
            'epsilon': [],
            'loss': []
        }
        
    def obs_to_tensor(self, obs):
        """Convert observation dict to tensor"""
        return np.array([
            obs['position'],
            obs['normalized_stock_price'],
            obs['time_to_expiry'],
            obs['normalized_portfolio_value'],
            obs['delta'],
            obs['gamma'],
            obs['volatility'],
        ], dtype=np.float32)
    
    def select_action(self, obs, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Random action
            action = random.randint(0, self.num_actions - 1)
            return action, None
        else:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(self.obs_to_tensor(obs)).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
            return action, None
    
    def train_step(self, batch_size=128):
        """Perform one training step"""
        if len(self.replay_buffer) < batch_size:
            return 0.0
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: Use main network to select actions, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def train(self, env, num_episodes=10000):
        print(f"Starting DDQN training for {num_episodes} episodes...")
        
        episode_returns = []
        episode_losses = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_return = 0
            episode_loss = 0
            step_count = 0
            
            # Store trajectory for P&L calculation
            trajectory = []
            
            while not done and not truncated:
                state = self.obs_to_tensor(obs)
                
                action, _ = self.select_action(obs, training=True)
                
                # Store trajectory info
                trajectory.append({
                    'action': action,
                    'position': env.action_space[action],
                    'stock_price': obs['normalized_stock_price'] * env.strike_price,
                    'portfolio_value': obs['normalized_portfolio_value']
                })
                
                # Take action with error handling
                try:
                    next_obs, reward, done, truncated, _ = env.step(action)
                    next_state = self.obs_to_tensor(next_obs)
                    
                    # Store transition in replay buffer
                    self.replay_buffer.push(state, action, reward, next_state, done or truncated)
                    

                    if len(self.replay_buffer) > 1000:  # Start training after collecting some experience
                        loss = self.train_step()
                        episode_loss += loss
                        step_count += 1
                    
                    episode_return += reward
                    obs = next_obs
                    
                except IndexError:
                    done = True
                    truncated = True
                    # Use current state as next state for final transition
                    self.replay_buffer.push(state, action, 0.0, state, True)
                    break
            
            # Calculate terminal P&L 
            normalized_return = self.compute_terminal_pnl(trajectory, env)
            episode_returns.append(normalized_return)
            
            avg_loss = episode_loss / max(step_count, 1)
            episode_losses.append(avg_loss)
            
            if (episode + 1) % 100 == 0:
                recent_returns = episode_returns[-500:]
                avg_return = np.mean(recent_returns)
                std_return = np.std(recent_returns)
                sharpe = avg_return / (std_return + 1e-8)
                
                self.train_statistics['episode'].append(episode + 1)
                self.train_statistics['avg_return'].append(avg_return)
                self.train_statistics['sharpe'].append(sharpe)
                self.train_statistics['epsilon'].append(self.epsilon)
                self.train_statistics['loss'].append(np.mean(episode_losses[-500:]))
                
                print(f"Episode {episode + 1}: Avg Return = {avg_return:.2%}, "
                      f"Sharpe = {sharpe:.3f}, Epsilon = {self.epsilon:.3f}, "
                      f"Loss = {avg_loss:.4f}")
                
                
            if (episode + 1) % 100 == 0:
                self.save_policy(f'DDQNPolicy.pkl')
    
    def compute_terminal_pnl(self, trajectory, env):
        
        # Final stock price
        final_price = trajectory[-1]['stock_price']
        
        # Option payoff at expiry (for put option) - BUYER receives this
        option_payoff = max(env.strike_price - final_price, 0)
        
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
        
        return normalized_return
    
    def save_policy(self, filepath):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_counter': self.update_counter,
        }, filepath)
        print(f"Policy saved to {filepath}")
    
    def load_policy(self, filepath):
        """Load a trained policy network"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.update_counter = checkpoint.get('update_counter', 0)
        print(f"Policy loaded from {filepath}")
    
    def save_train_statistics(self, filepath):
        df = pd.DataFrame(self.train_statistics)
        df.to_csv(filepath, index=False)
    
    def plot_train_statistics(self, filepath):
        df = pd.read_csv(filepath)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Average Return
        axes[0, 0].plot(df['episode'], df['avg_return'])
        axes[0, 0].set_title('Average Return')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Return')
        
        # Sharpe Ratio
        axes[0, 1].plot(df['episode'], df['sharpe'])
        axes[0, 1].set_title('Sharpe Ratio')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Sharpe')
        
        # Epsilon (exploration rate)
        axes[1, 0].plot(df['episode'], df['epsilon'])
        axes[1, 0].set_title('Epsilon (Exploration Rate)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        
        # Training Loss
        axes[1, 1].plot(df['episode'], df['loss'])
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
        plt.show()

def evaluate_ddqn_agent(agent, env, num_episodes=1000):
    print(f"Evaluating DDQN agent for {num_episodes} episodes")
    
    results = {
        'returns': [],
        'option_payoffs': [],
        'hedging_pnls': [],
        'premiums_paid': [],
        'initial_expiry_days': [],
        'optimal_max_returns': [],
        'optimal_min_returns': []
    }
    
    for episode in range(num_episodes):
        if (episode + 1) % 500 == 0:
            print(f"Evaluation episode {episode + 1}/{num_episodes}")
        
        # Reset environment
        obs, _ = env.reset()
        done = False
        truncated = False
        
        # Store initial conditions
        initial_expiry = env.time_to_expiry
        initial_investment = env.premium_per_share * env.number_of_shares * env.risk
        
        # Store trajectory
        trajectory = []
        
        while not done and not truncated:
            # Select action (no exploration during evaluation)
            action, _ = agent.select_action(obs, training=False)
            
            # Store trajectory info
            trajectory.append({
                'action': action,
                'position': env.action_space[action],
                'stock_price': obs['normalized_stock_price'] * env.strike_price,
                'portfolio_value': obs['normalized_portfolio_value']
            })
            
            # Take action with error handling
            try:
                obs, reward, done, truncated, _ = env.step(action)
            except IndexError:
                # Handle case where we've reached end of data
                done = True
                truncated = True
                break
        
        final_price = trajectory[-1]['stock_price']
        option_payoff = max(env.strike_price - final_price, 0) * env.number_of_shares
        
        trading_pnl = 0
        for t in range(len(trajectory) - 1):
            position = trajectory[t]['position']
            price_change = trajectory[t+1]['stock_price'] - trajectory[t]['stock_price']
            trading_pnl += position * price_change * env.number_of_shares
        
        final_value = option_payoff + trading_pnl
        normalized_return = (final_value - initial_investment) / initial_investment
        
        # Compute optimal returns for comparison
        max_optimal, min_optimal = env.compute_optimal_pnls()
        
        # Store results
        results['returns'].append(normalized_return)
        results['option_payoffs'].append(option_payoff)
        results['hedging_pnls'].append(trading_pnl)
        results['premiums_paid'].append(initial_investment)
        results['initial_expiry_days'].append(initial_expiry)
        results['optimal_max_returns'].append(max_optimal)
        results['optimal_min_returns'].append(min_optimal)
    
    return results

if __name__ == "__main__":
    import yfinance as yf
    from datetime import datetime
    import pandas as pd
    import os
    
    def retrieve_data(ticker : str, start: datetime, end: datetime):
        df = yf.download(ticker, start=start, end=end)
        df.columns = [col[0].lower() for col in df.columns]
        return df
    
    def save_data(df : pd.DataFrame, ticker : str):
        df.to_pickle(f'./data/{ticker}.pkl')
    
    def load_data(ticker : str):
        if os.path.exists(f'./data/{ticker}.pkl'):
            return pd.read_pickle(f'./data/{ticker}.pkl')
        else:
            return None
    
    tickers = ['AAPL', 'MSFT', 'IBM', 'JNJ', 'MCD', 
               'KO', 'PG', 'WMT', 'XOM', 'GE', 
               'MMM', 'F', 'T', 'CSCO', 'PFE',
               'INTC', 'BA', 'CAT', 'CVX', 'PEP']
    
    if not os.path.exists('./data'):
        os.makedirs('./data')
    
    for ticker in tickers:
        if not os.path.exists(f'./data/{ticker}.pkl'):
            print(f"Downloading data for {ticker}...")
            df = retrieve_data(ticker, datetime(1998, 1, 1), datetime(2024, 12, 31))
            save_data(df, ticker)
    
    env = OptionEnv(tickers, verbose=False)
    
    # Initialize DDQN agent
    agent = DDQNAgent(
        num_actions=51,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.999,
        target_update_freq=250
    )
    
    agent.train(env, num_episodes=5000)
    
    agent.save_train_statistics('DDQNResults_training.csv')
    
    results = evaluate_ddqn_agent(agent, env, num_episodes=1000)
    
    # Save evaluation results
    with open('DDQNResults_evaluation.json', 'w') as f:
        json.dump(results, f)
    
    print("Training and evaluation completed!")
    print(f"Average return: {np.mean(results['returns'])*100:.2f}%")
    print(f"Sharpe ratio: {np.mean(results['returns'])/np.std(results['returns']):.3f}")