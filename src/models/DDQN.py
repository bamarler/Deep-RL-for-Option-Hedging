import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import pandas as pd
import numpy as np
import random
from collections import deque
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from src.models.agent import Agent

class DDQNNetwork(nn.Module):
    def __init__(self, num_actions=51):
        """Initialize the DDQN network"""
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
        """Forward pass through the network"""
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
        """Initialize the replay buffer"""
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Push a transition into the replay buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of transitions from the replay buffer"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        """Return the number of transitions in the replay buffer"""
        return len(self.buffer)

class DDQNAgent(Agent):
    def __init__(self, num_actions=51, learning_rate=0.001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 target_update_freq=1000, buffer_size=100000):
        """Initialize the DDQN agent"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        self.q_network = DDQNNetwork(num_actions).to(self.device)
        self.target_network = DDQNNetwork(num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = AdamW(self.q_network.parameters(), lr=learning_rate, weight_decay=0.0001)
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        self.train_statistics = {
            'episode': [],
            'avg_return': [],
            'sharpe': [],
            'epsilon': [],
            'loss': []
        }
        
    def obs_to_tensor(self, obs):
        """Convert observation dict to tensor
        
        Parameters:
        - obs: Observation dict
        
        Returns:
        - state_tensor: State tensor
        """
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
        """Select action using epsilon-greedy policy
        
        Parameters:
        - obs: Observation dict
        - training: Whether to use training policy
        
        Returns:
        - action: Selected action
        - log_prob: Log probability of selected action (None if not training)
        """
        if training and random.random() < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
            return action, None
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(self.obs_to_tensor(obs)).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
            return action, None
    
    def train_step(self, batch_size):
        """
        Perform one training step
        
        Parameters:
        - batch_size: Batch size for training
        
        Returns:
        - loss: Training loss
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        loss = F.mse_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def train(self, env, policy_file_path, train_statistics_file_path, batch_size=128, num_episodes=10000, verbose=True):        
        """
        Train the agent
        
        Parameters:
        - env: Environment to train on
        - policy_file_path: Path to save the policy
        - train_statistics_file_path: Path to save the training statistics
        - batch_size: Batch size for training
        - num_episodes: Number of episodes to train for
        - verbose: Whether to print verbose output
        """
        episode_returns = []
        episode_losses = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_return = 0
            episode_loss = 0
            step_count = 0
            
            trajectory = []
            
            while not done and not truncated:
                state = self.obs_to_tensor(obs)
                
                action, _ = self.select_action(obs, training=True)
                
                trajectory.append({
                    'action': action,
                    'position': env.action_space[action],
                    'stock_price': obs['normalized_stock_price'] * env.strike_price,
                    'portfolio_value': obs['normalized_portfolio_value']
                })
                
                try:
                    next_obs, reward, done, truncated, _ = env.step(action)
                    next_state = self.obs_to_tensor(next_obs)
                    
                    self.replay_buffer.push(state, action, reward, next_state, done or truncated)
                    

                    if len(self.replay_buffer) > 1000: 
                        loss = self.train_step(batch_size)
                        episode_loss += loss
                        step_count += 1
                    
                    episode_return += reward
                    obs = next_obs
                    
                except IndexError:
                    done = True
                    truncated = True
                    self.replay_buffer.push(state, action, 0.0, state, True)
                    break
                    
            normalized_return = self.compute_terminal_pnl(trajectory, env)
            episode_returns.append(normalized_return.detach().numpy())
            
            avg_loss = episode_loss / max(step_count, 1)
            episode_losses.append(avg_loss)
            recent_returns = episode_returns[-500:]
            avg_return = np.mean(recent_returns)
            std_return = np.std(recent_returns)
            sharpe = avg_return / (std_return + 1e-8)
            
            self.train_statistics['episode'].append(episode + 1)
            self.train_statistics['avg_return'].append(avg_return)
            self.train_statistics['sharpe'].append(sharpe)
            self.train_statistics['epsilon'].append(self.epsilon)
            self.train_statistics['loss'].append(np.mean(episode_losses[-500:]))
            
            if (episode + 1) % 100 == 0:
                self.save_policy(policy_file_path)
                self.save_train_statistics(train_statistics_file_path)
                
                if verbose:
                    print(f"Episode {episode + 1}: Avg Return = {avg_return:.2%}, "
                          f"Sharpe = {sharpe:.3f}, Epsilon = {self.epsilon:.3f}, "
                          f"Loss = {avg_loss:.4f}")
        
        self.save_policy(policy_file_path)
        self.save_train_statistics(train_statistics_file_path)
    
    def save_policy(self, filepath):
        """Save the policy to a file"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_counter': self.update_counter,
        }, filepath)
        print(f"Policy saved to {filepath}")
    
    def load_policy(self, filepath):
        """Load a policy from a file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.update_counter = checkpoint.get('update_counter', 0)
        print(f"Policy loaded from {filepath}")
    
    def save_train_statistics(self, filepath):
        """Save the training statistics to a file"""
        df = pd.DataFrame(self.train_statistics)
        df.to_csv(filepath, index=False)