import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from src.models.agent import Agent

class MCPGPositionNetwork(nn.Module):
    def __init__(self, num_actions=51, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Initialize the MCPG position network
        
        Parameters:
        - num_actions: Number of possible actions
        - device: Device to run the network on
        """
        super().__init__()

        input_dim = 7
        
        self.stack = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        
    def forward(self, obs):
        """
        Forward pass through the network
        
        Parameters:
        - obs: Observation dict
        
        Returns:
        - x: Raw logits (softmax applied during action selection)
        """
        features = torch.tensor([
            obs['position'],
            obs['normalized_stock_price'],
            obs['time_to_expiry'],
            obs['normalized_portfolio_value'],
            obs['delta'],
            obs['gamma'],
            obs['volatility'],
        ], dtype=torch.float32).to(self.device)
        
        x = self.stack(features)
        
        return x

class MCPGAgent(Agent):
    def __init__(self, num_actions=51, risk_aversion=1.0, learning_rate=0.003, loss_function='entropic'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = MCPGPositionNetwork(num_actions, device=self.device).to(self.device)
        self.num_actions = num_actions
        self.risk_aversion = risk_aversion  # λ for entropic risk
        self.loss_function = loss_function
        self.optimizer = AdamW(self.network.parameters(), lr=learning_rate, weight_decay=0.0001, eps=1e-7)

    def select_action(self, obs, training=True):        
        """
        Select action using epsilon-greedy policy
        
        Parameters:
        - obs: Observation dict
        - training: Whether to use training policy
        
        Returns:
        - action: Selected action
        - log_prob: Log probability of selected action (None if not training)
        """
        logits = self.network(obs)
        
        if training:
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob
        else:
            action = torch.argmax(logits)
            return action.item(), None
    
    def train(self, env, policy_file_path, train_statistics_file_path, num_episodes=10000, batch_size=256):
        """
        Train the MCPG agent
        
        Parameters:
        - env: Environment to train on
        - policy_file_path: Path to save the trained policy
        - train_statistics_file_path: Path to save the training statistics
        - num_episodes: Number of episodes to train for
        - batch_size: Batch size for training
        """
        print(f"Training MCPG agent for {num_episodes} episodes in batch size {batch_size} with {self.loss_function} loss function...")
        self.train_statistics = {
            'epoch': [],
            'avg_return': [],
            'entropic': [],
            'sharpe': [],
            'markowitz': []
        }
        
        for epoch in range(num_episodes // batch_size):
            batch_log_probs = []
            batch_normalized_returns = []
            
            for episode in range(batch_size):
                episode_log_probs = []
                trajectory = []
                
                obs, _ = env.reset()
                done = False
                truncated = False
                
                while not done and not truncated:
                    action, log_prob = self.select_action(obs, training=True)
                    
                    episode_log_probs.append(log_prob)
                    
                    next_obs, reward, done, truncated, _ = env.step(action)
                    
                    trajectory.append({
                        'action': action,
                        'position': env.action_space[action],
                        'stock_price': obs['normalized_stock_price'] * env.strike_price,
                        'portfolio_value': obs['normalized_portfolio_value']
                    })
                    
                    obs = next_obs
                
                normalized_return = self.compute_terminal_pnl(trajectory, env)
                
                batch_log_probs.append(torch.stack(episode_log_probs))
                batch_normalized_returns.append(normalized_return)

            batch_returns_tensor = torch.stack(batch_normalized_returns).to(self.device)

            mean_return = batch_returns_tensor.mean()
            std_return = batch_returns_tensor.std() + 1e-8

            entropic_risk = (1/self.risk_aversion) * torch.log(torch.mean(torch.exp(-self.risk_aversion * batch_returns_tensor)))
            sharpe_risk = mean_return / (std_return + 1e-8)
            markowitz_risk = -mean_return + (1 / self.risk_aversion) * batch_returns_tensor.var()

            weights = None
            if self.loss_function == 'entropic':
                exp_weighted_returns = torch.exp(-self.risk_aversion * batch_returns_tensor)
                weights = exp_weighted_returns / exp_weighted_returns.mean()
            elif self.loss_function == 'sharpe':
                weights = (batch_returns_tensor - mean_return) / std_return
            elif self.loss_function == 'markowitz':
                with torch.no_grad():
                    weights = batch_returns_tensor - (0.5 * self.risk_aversion * (batch_returns_tensor ** 2))
            else:
                raise ValueError(f"Invalid loss function: {self.loss_function}")
            
            policy_loss = 0
            for i in range(batch_size):
                trajectory_loss = -batch_log_probs[i].sum() * weights[i].detach()
                policy_loss += trajectory_loss
            
            policy_loss = policy_loss / batch_size
            
            self.optimizer.zero_grad()
            policy_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            self.train_statistics['epoch'].append(epoch + 1)
            self.train_statistics['avg_return'].append(mean_return.item())
            self.train_statistics['entropic'].append(entropic_risk.item())
            self.train_statistics['sharpe'].append(sharpe_risk.item())
            self.train_statistics['markowitz'].append(markowitz_risk.item())
            
            print(f"Epoch {epoch + 1}: Loss = {policy_loss.item():.3f}, Avg Return = {mean_return.item():.2%}, Entropic = {entropic_risk.item():.3f}, Sharpe = {sharpe_risk.item():.3f}, Markowitz = {markowitz_risk.item():.3f}")
            self.save_policy(policy_file_path)
            self.save_train_statistics(train_statistics_file_path)
    
    def save_train_statistics(self, filepath):
        """Save the training statistics to a CSV file"""
        df = pd.DataFrame(self.train_statistics)
        df.to_csv(filepath, index=False)
    
    def save_policy(self, filepath):
        """Save the trained policy network"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load_policy(self, filepath):
        """Load a trained policy network"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Policy loaded from {filepath}")