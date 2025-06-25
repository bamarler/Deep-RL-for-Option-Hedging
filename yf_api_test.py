# import modules
from datetime import date, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# initialize parameters
ticker = 'AAPL'
length = 30 # 
start_date = date(2025, 4, 29)
end_date = date(2025, 5, 29)

# get the data
data = yf.download(ticker, start = start_date,
                   end = end_date)

sample = data[date(2025, 5, 10):date(2025, 5, 18)]
print(sample)

print(yf.download(ticker, start = start_date,
                   end = start_date + timedelta(days=1)))

prices = data['Close']
log_returns = np.log(prices / prices.shift(1)).dropna()
    
# Calculate EWMA of squared returns
squared_returns = log_returns ** 2
ewma_span = max(length // 4, 10)
ewma_variance = squared_returns.ewm(span=ewma_span, adjust=False).mean()
        
# Take square root to get volatility (use last value)
volatility = np.sqrt(ewma_variance.iloc[-1]) * np.sqrt(252)

print(pd.to_datetime(1.3366944e+18).date())

# plt.figure(figsize = (10,5))
# plt.title(f'Opening Prices for {ticker} from {start_date} to {end_date}')
# plt.plot(data['High'])
# plt.show()