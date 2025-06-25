import gymnasium as gym
import gym_trading_env
import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import date, timedelta
import os
import random
import pandas as pd

class OptionEnv():
    def __init__(self, tickers: list[str], risk=0.3, risk_free_rate = 0.05, moneyness=1.0, lookback_days=252, verbose=True):
        self.risk = risk
        self.moneyness = moneyness
        self.risk_free_rate = risk_free_rate
        self.lookback_days = lookback_days

        self.tickers = tickers
        self.data = {
            ticker: self._retrieve_data(ticker) for ticker in tickers
        }

        self.expiry_days = [7, 14, 30, 45, 60, 90]

        self.action_space = [0.02 * i for i in range(0, 51)]
        
        #Episode Specific Variables
        self.env = None
        self.ticker = None
        self.strike_price = None
        self.time_to_expiry = None
        self.premium_per_share = None
        self.number_of_shares = None
        self.initial_portfolio_value = None
        self.current_day = None
        self.done = None

        self.verbose = verbose

        self.reset()

    def reset(self):
        self.ticker = random.choice(self.tickers)
        self.time_to_expiry = random.choice(self.expiry_days)
        end_date = date(2024, 12, 31) - timedelta(days=random.randint(1, 25 * 365 - self.time_to_expiry))

        self.current_day = end_date - timedelta(days=self.time_to_expiry)
        if self.verbose:
            print(f"Ticker: {self.ticker} Current day: {self.current_day} End date: {end_date}")

        stock_data = self.data[self.ticker][self.current_day:end_date]

        self.current_day = stock_data.index[0].date()

        self.strike_price = stock_data['open'].loc[str(self.current_day)] * self.moneyness

        self.number_of_shares = random.randint(1, 10) * 100

        self.premium_per_share = self._black_scholes_call(self.current_day)

        self.initial_portfolio_value = self.number_of_shares * self.premium_per_share * self.risk

        def reward_function(history):
            if history["data_close", -1] > self.strike_price:
                factor = 1
            else:
                factor = 1 + 10 * (1 - (history["data_close", -1] / self.strike_price))
            
            return factor *np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])
        
        self.env = gym.make("TradingEnv",
                name=f"{self.ticker} FROM {self.current_day} TO {end_date}",
                df=self._preprocess(stock_data),
                positions = self.action_space,
                portfolio_initial_value = self.initial_portfolio_value,
                dynamic_feature_functions = [
                    lambda history: history['position', -1],
                    lambda history: history['portfolio_valuation', -1],
                    lambda history: history['date', -1],
                ],
                max_episode_duration='max',
                verbose=self.verbose,
                reward_function=reward_function,
            )
        env_obs, info = self.env.reset()
        self.done = False

        return self._get_obs(env_obs), info

    def step(self, action):
        if self.done:
            return self._get_obs(self.env.get_observation()), 0, True, False, {}

        env_obs, reward, done, truncated, info = self.env.step(action)
        self.done = done
        new_day = pd.to_datetime(env_obs[3]).date()
        self.time_to_expiry -= (new_day - self.current_day).days
        self.current_day = new_day

        if done or self.time_to_expiry == 0:
            self.done = True
        return self._get_obs(env_obs), reward, self.done, truncated, info

    def _retrieve_data(self, ticker: str):
        if os.path.exists(f'./data/{ticker}.pkl'):
            return pd.read_pickle(f'./data/{ticker}.pkl')
        else:
            raise FileNotFoundError(f"Data for {ticker} not found, please download it first.")

    def _preprocess(self, df : pd.DataFrame):
        df = df.copy()
        df['feature_stock_price'] = df['open']
        return df

    def _get_obs(self, env_obs):
        '''
        Returns the observation space for the environment

        Parameters:
        - env_obs: Observation from the environment

        Returns:
        - dict: Observation space for the environment where
            - position: Discrete(0, 1, 0.02)
            - normalized_stock_price: Continuous(positive)
            - time_to_expiry: Discrete(0, 90)
            - normalized_portfolio_value: Continuous(positive)
            - delta: Continuous(0, 1)
            - gamma: Continuous(positive)
            - volatility: Continuous(0, 1)
        '''
        sigma = self._calculate_volatility()
        greeks = self._calculate_greeks()

        return {
            'position': env_obs[1],
            'normalized_stock_price': env_obs[0] / self.strike_price,
            'time_to_expiry': self.time_to_expiry, 
            'normalized_portfolio_value': (env_obs[2] - self.initial_portfolio_value) / self.initial_portfolio_value,
            'delta': greeks['delta'],
            'gamma': greeks['gamma'],
            'volatility': sigma,
        }
    
    def get_observation(self):
        return self._get_obs(self.env.get_observation())

    def _calculate_volatility(self):
        """Calculate annualized historical volatility using EWMA (more emphasis on recent days)"""
        buffer_days = int(self.lookback_days * 1.5)
        start_date = self.current_day - timedelta(days=buffer_days)
        end_date = self.current_day

        stock_data = self.data[self.ticker][start_date:end_date]
        
        ewma_span = max(self.lookback_days // 4, 10)

        prices = stock_data['close']
        log_returns = np.log(prices / prices.shift(1)).dropna()
    
        # Calculate EWMA of squared returns
        squared_returns = log_returns ** 2
        ewma_variance = squared_returns.ewm(span=ewma_span, adjust=False).mean()

        # Take square root to get volatility (use last value)
        return np.sqrt(ewma_variance.iloc[-1]) * np.sqrt(252)

    def _black_scholes_call(self, date: date):
        """
        Calculate Black-Scholes call option price
        
        Parameters:
        - date: Date to calculate premium for
        """
        tau = self.time_to_expiry / 365.0 if self.time_to_expiry > 0 else 1
        S = self.data[self.ticker][date:date + timedelta(days=1)]['open'].values[0]
        K = self.strike_price
        sigma = self._calculate_volatility()

        if self.time_to_expiry <= 0:
            return max(0, S - K)
        
        d1 = self._d1(tau, S, K, sigma)
        d2 = self._d2(tau, S, K, sigma)
        
        premium_per_share = S*norm.cdf(d1) - K*np.exp(-self.risk_free_rate*tau)*norm.cdf(d2)
        
        return premium_per_share

    def _calculate_greeks(self):
        """
        Calculate option Greeks (Delta and Gamma for call options)
        
        Returns dict with delta and gamma
        """
        tau = self.time_to_expiry / 365.0 if self.time_to_expiry > 0 else 1
        S = self.data[self.ticker][self.current_day:self.current_day + timedelta(days=1)]['open'].values[0]
        K = self.strike_price
        sigma = self._calculate_volatility()
        
        d1 = self._d1(tau, S, K, sigma)
        
        return {
            'delta': norm.cdf(d1),
            'gamma': norm.pdf(d1) / (S * sigma * np.sqrt(tau))
        }

    def _d1(self, tau, S, K, sigma):
        """Calculate d1 for Black-Scholes
        
        Parameters:
        - tau: Time to expiration in years
        - S: Current stock price
        - K: Strike price
        - sigma: Volatility
        """
        return (np.log(S/K) + (self.risk_free_rate + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    
    def _d2(self, tau, S, K, sigma):
        """Calculate d2 for Black-Scholes
        
        Parameters:
        - tau: Time to expiration in years
        - S: Current stock price
        - K: Strike price
        - sigma: Volatility
        """
        return self._d1(tau, S, K, sigma) - sigma*np.sqrt(tau)
    
    def compute_optimal_pnls(self):
        """
        Compute the maximum and minimum possible PNLs using perfect hindsight.
        Max: Hold full position (1.0) when price will increase, 0 when it will decrease
        Min: Hold full position (1.0) when price will decrease, 0 when it will increase
        """
        # Get the price series from the data
        start_date = self.current_day
        end_date = start_date + timedelta(days=self.time_to_expiry)
        stock_data = self.data[self.ticker][start_date:end_date]
        prices = stock_data['close'].values
        
        # Calculate price changes
        price_changes = np.diff(prices)
        
        # Optimal positions (with perfect foresight)
        # For max PNL: hold when price will increase
        optimal_positions_max = (price_changes > 0).astype(float)
        # For min PNL: hold when price will decrease (worst case)
        optimal_positions_min = (price_changes < 0).astype(float)
        
        # Calculate PNLs
        max_hedging_pnl = np.sum(optimal_positions_max * price_changes) * self.number_of_shares
        min_hedging_pnl = np.sum(optimal_positions_min * price_changes) * self.number_of_shares
        
        # Add option payoff
        final_price = prices[-1]
        option_payoff = max(final_price - self.strike_price, 0) * self.number_of_shares
        
        # Total PNLs (normalized by initial investment)
        initial_investment = self.premium_per_share * self.number_of_shares * self.risk
        
        max_total_pnl = (option_payoff + max_hedging_pnl - initial_investment) / initial_investment
        min_total_pnl = (option_payoff + min_hedging_pnl - initial_investment) / initial_investment
        
        return max_total_pnl, min_total_pnl