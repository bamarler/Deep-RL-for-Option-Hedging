import gymnasium as gym
import gym_trading_env
import yfinance as yf
from datetime import datetime, timedelta
import random

class OptionEnv():
    def __init__(self, risk=0.3, moneyness=1.0, risk_free_rate=0.05, lookback_days=252):
        self.risk = risk
        self.moneyness = moneyness
        self.risk_free_rate = risk_free_rate
        self.lookback_days = lookback_days

        self.tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 
                       'TSLA', 'JPM', 'JNJ', 'V', 'WMT', 'PG', 'MA', 'DIS']
        self.expiry_days = [7, 14, 30, 45, 60, 90]


        self.action_space = [0.02 * i for i in range(0, 51)]
        
        #Episode Specific Variables
        self.historical_data = None
        self.env = None
        self.ticker = None
        self.strike_price = None
        self.time_to_expiry = None
        self.premium_per_share = None
        self.number_of_shares = None
        self.initial_portfolio_value = None
        self.current_day = None

        self.reset()

    def reset(self):
        self.ticker = random.choice(self.tickers)
        end_date = datetime.now() - timedelta(days=random.randint(10, 20 * 365))

        self.time_to_expiry = random.choice(self.expiry_days)
        self.current_day = end_date - timedelta(days=self.time_to_expiry)

        self.historical_data = self._retrieve_data(self.ticker, self.current_day - timedelta(days=self.lookback_days * 1.5), end_date)

        self.strike_price = self.historical_data['open'].loc[self.current_day].values[0] * self.moneyness

        self.number_of_shares = random.randint(1, 10) * 100

        self.premium_per_share = self._black_scholes_call(self.current_day)

        self.initial_portfolio_value = self.number_of_shares * self.premium_per_share * self.risk

        self.historical_data['feature_stock_price'] = self.historical_data['open']
        
        self.env = gym.make("TradingEnv",
                name="Test Environment",
                df=self.historical_data[self.historical_data.index >= self.current_day],
                positions = self.action_space,
                portfolio_initial_value = self.initial_portfolio_value
            )
        env_obs, info = self.env.reset()

        return self._get_obs(env_obs), info

    def step(self, action):
        self.time_to_expiry -= 1
        env_obs, reward, done, info = self.env.step(action)
        return self._get_obs(env_obs), reward, done, info

    def _retrieve_data(self, ticker : str, start: datetime, end: datetime):
        df = yf.download(ticker, start=start, end=end)
        df.columns = [col[0].lower() for col in df.columns]
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
        sigma = self._calculate_sigma()
        greeks = self._calculate_greeks(self.strike_price, self.strike_price, self.time_to_expiry, sigma)
        
        # TODO: Add normalized portfolio value
        return {
            'position': env_obs[0], 
            'normalized_stock_price': self.historical_data['open'].loc[self.current_day].values[0] / self.strike_price,
            'time_to_expiry': self.time_to_expiry, 
            'normalized_portfolio_value': None,
            'delta': greeks['delta'], 
            'gamma': greeks['gamma'], 
            'volatility': sigma, 
        }

    def _calculate_sigma(self):
        """Calculate annualized historical volatility using EWMA (more emphasis on recent days)"""
        buffer_days = int(self.lookback_days * 1.5)
        start_date = self.current_day - timedelta(days=buffer_days)
        end_date = self.current_day

        stock_data = self.historical_data[start_date:end_date]
        
        ewma_span = max(self.lookback_days // 4, 10)

        prices = stock_data['close'].values
        log_returns = np.log(prices / prices.shift(1)).dropna()
    
        # Calculate EWMA of squared returns
        squared_returns = log_returns ** 2
        ewma_variance = squared_returns.ewm(span=ewma_span, adjust=False).mean()
        
        # Take square root to get volatility (use last value)
        return np.sqrt(ewma_variance) * np.sqrt(252)

    def _black_scholes_call(self, date: datetime):
        """
        Calculate Black-Scholes call option price
        
        Parameters:
        - date: Date to calculate premium for
        """
        tau = self.time_to_expiry / 365.0
        S = self.historical_data['open'].loc[date].values[0]
        K = self.strike_price
        sigma = self._calculate_sigma()

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
        tau = self.time_to_expiry / 365.0
        S = self.historical_data['open'].loc[self.current_day].values[0]
        K = self.strike_price
        sigma = self._calculate_sigma()
        
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