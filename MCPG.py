import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pandas as pd
import matplotlib.pyplot as plt

class MCPGPositionNetwork(nn.Module):
    def __init__(self, num_actions=51):
        super().__init__()
        
        # Input features: 7 from obs + 1 previous position = 8 total
        input_dim = 7
        
        self.stack = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )
        
    def forward(self, obs):
        # Extract features from observation dict
        features = torch.tensor([
            obs['position'],
            obs['normalized_stock_price'],
            obs['time_to_expiry'],
            obs['normalized_portfolio_value'],
            obs['delta'],
            obs['gamma'],
            obs['volatility'],
        ], dtype=torch.float32)
        
        # Forward pass
        x = self.stack(features)
        
        # Return raw logits (softmax applied during action selection)
        return x

class MCPGAgent:
    def __init__(self, num_actions=51, risk_aversion=1.0):
        self.network = MCPGPositionNetwork(num_actions)
        self.num_actions = num_actions
        self.risk_aversion = risk_aversion  # λ for entropic risk
        self.optimizer = Adam(self.network.parameters(), lr=0.001)
        
    def select_action(self, obs, training=True):
        # Get logits from network
        logits = self.network(obs)
        
        if training:
            # Sample from categorical distribution during training
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(), log_prob
        else:
            # Deterministic action during testing
            action = torch.argmax(logits)
            return action.item(), None
    
    def train(self, env, num_episodes=10000, batch_size=256):
        """Train the MCPG agent"""
        self.train_statistics = {
            'epoch': [],
            'entropic_risk': [],
            'avg_return': [],
            'sharpe': []
        }
        
        for epoch in range(num_episodes // batch_size):
            # Storage for batch
            batch_log_probs = []
            batch_normalized_returns = []
            
            # 1. MONTE CARLO: Generate batch of complete trajectories
            for episode in range(batch_size):
                # if (episode + 1) % (batch_size // 4) == 0:
                #     print(f"Episode {episode + 1}")
                # Storage for this episode
                episode_log_probs = []
                trajectory = []
                
                # Reset environment
                obs, _ = env.reset()
                done = False
                truncated = False
                
                while not done and not truncated:
                    # Get action from policy network
                    action, log_prob = self.select_action(obs, training=True)
                    
                    # Store log probability for REINFORCE
                    episode_log_probs.append(log_prob)
                    
                    # Take action in environment
                    next_obs, reward, done, truncated, _ = env.step(action)
                    
                    # Store trajectory info for P&L calculation
                    trajectory.append({
                        'action': action,
                        'position': env.action_space[action],
                        'stock_price': obs['normalized_stock_price'] * env.strike_price,
                        'portfolio_value': obs['normalized_portfolio_value']
                    })
                    
                    # Update for next step
                    obs = next_obs
                
                # Calculate normalized terminal return for this episode
                normalized_return = self.compute_terminal_pnl(trajectory, env)
                
                # Store batch data
                batch_log_probs.append(torch.stack(episode_log_probs))
                batch_normalized_returns.append(normalized_return)
            
            # 2. RISK MEASURE: Compute entropic risk measure on normalized returns
            batch_returns_tensor = torch.stack(batch_normalized_returns)
            
            # Entropic risk on returns: ρ(R) = (1/λ) * log(E[exp(-λR)])
            # Note: We minimize risk of negative returns (maximize risk-adjusted returns)
            entropic_risk = (1/self.risk_aversion) * torch.log(
                torch.mean(torch.exp(-self.risk_aversion * batch_returns_tensor))
            )
            
            # 3. POLICY GRADIENT: Compute gradients
            # For each trajectory, gradient is: ∇log π(a|s) * exp(-λ * R) / E[exp(-λ * R)]
            exp_weighted_returns = torch.exp(-self.risk_aversion * batch_returns_tensor)
            weights = exp_weighted_returns / exp_weighted_returns.mean()
            
            # Compute policy gradient loss
            policy_loss = 0
            for i in range(batch_size):
                # Weight each trajectory by its contribution to risk measure
                # Negative because we want to maximize returns (minimize negative returns)
                trajectory_loss = -batch_log_probs[i].sum() * weights[i].detach()
                policy_loss += trajectory_loss
            
            policy_loss = policy_loss / batch_size
            
            # 4. UPDATE: Backpropagate and update network
            self.optimizer.zero_grad()
            policy_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Logging
            avg_return = batch_returns_tensor.mean().item()
            std_return = batch_returns_tensor.std().item()
            sharpe = avg_return / (std_return + 1e-8)
            
            self.train_statistics['epoch'].append(epoch + 1)
            self.train_statistics['entropic_risk'].append(entropic_risk.item())
            self.train_statistics['avg_return'].append(avg_return)
            self.train_statistics['sharpe'].append(sharpe)
            
            print(f"Epoch {epoch + 1}: Entropic Risk = {entropic_risk.item():.4f}, "
                f"Avg Return = {avg_return:.2%}, Sharpe = {sharpe:.3f}")
        
            if (epoch + 1) % 10 == 0:
                self.save_policy('MCPGPolicy.pkl')
    
    def save_train_statistics(self, filepath):
        """Save the training statistics to a CSV file"""
        df = pd.DataFrame(self.train_statistics)
        df.to_csv(filepath, index=False)
    
    def plot_train_statistics(self, filepath):
        """Plot the training statistics"""
        df = pd.read_csv(filepath)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Entropic Risk
        axes[0, 0].plot(df['epoch'], df['entropic_risk'])
        axes[0, 0].set_title('Entropic Risk')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Risk')
        
        # Average Return
        axes[0, 1].plot(df['epoch'], df['avg_return'])
        axes[0, 1].set_title('Average Return')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Return')
        
        # Sharpe Ratio
        axes[1, 0].plot(df['epoch'], df['sharpe'])
        axes[1, 0].set_title('Sharpe Ratio')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Sharpe')
        
        plt.tight_layout()
        plt.show()
    
    def compute_terminal_pnl(self, trajectory, env):
        """Calculate terminal P&L for a complete episode from BUYER's perspective"""
        
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
        
        return torch.tensor(normalized_return, dtype=torch.float32, requires_grad=True)
    
    def save_policy(self, filepath):
        """Save the trained policy network"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Policy saved to {filepath}")
    
    def load_policy(self, filepath):
        """Load a trained policy network"""
        checkpoint = torch.load(filepath)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Policy loaded from {filepath}")