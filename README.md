# Dynamic Hedging Strategies for Option Buyers Using Deep Reinforcement Learning

## Abstract

While extensive research exists on option hedging strategies from the seller's perspective, there remains a significant gap in developing effective strategies for option buyers. Given a fixed strike price, premium, and expiry date, we explore how buyers should optimally hold the underlying stock over time. We compare Deep Double Q-Network (DDQN) against three Monte Carlo Policy Gradient (MCPG) variants using different risk-based loss functions: entropic, Sharpe ratio, and Markowitz. Our custom trading environment features a 7-dimensional state space including position size, normalized stock price, time to expiry, portfolio value, option Greeks ($\delta$ and $\gamma$), and implied volatility with daily data from multiple tickers. DDQN achieves superior mean returns of 352.4% with a 19.9% capture percentage, compared to MCPG variants' 220-264% returns, though at the cost of significantly higher volatility (1159.3% vs. 697-807%). Despite these differences, all models maintain comparable Sharpe ratios (0.30-0.40). Notably, DDQN exhibits weaker downside protection with average losses of -589.2%, compared to MCPG's -261.9% to -344.9%.

**→ [Full Technical Report (PDF)](./report.pdf)**

## Key Features

- Custom gym environment simulating realistic option trading scenarios
- Multiple ticker support with random episode generation
- Black-Scholes pricing with dynamic volatility calculation
- Comprehensive state space including normalized prices and option Greeks
- Flexible action space for position sizing (0-100% of portfolio)

## Results
![image](https://github.com/user-attachments/assets/4e35eb49-35fc-40d5-81d1-f7ac32ffa703)


## How to Use

- create a virtual environment
- run the following command in the root directory to install dependencies
```
pip install -r requirements.txt
```
- optionally, if you have a compatible GPU and want to train with CUDA, follow these instructions: [Successfully using your local NVIDIA GPU with PyTorch or TensorFlow](https://medium.com/@nimritakoul01/successfully-using-your-local-nvidia-gpu-with-pytorch-or-tensorflow-756f3518e88f)
- Now you are ready to train and test models! The relevant project structure is below:
```
├── policies
├── results
│   ├── data
│   │   ├── testing
│   │   │   ├── DDQN
│   │   │   └── MCPG
│   │   └── training
│   │       ├── DDQN
│   │       ├── MCPG
│   │       └── Q-Learning
│   └── images
│       ├── testing
│       │   ├── comparison
│       │   ├── DDQN
│       │   └── MCPG
│       └── training
│           ├── DDQN
│           ├── MCPG
│           └── Q-Learning
├── src
│   ├── environment
│   ├── models
│   ├── testing
│   ├── training
│   ├── util
│   └── visualization
├── README.md
└── report.pdf
```
