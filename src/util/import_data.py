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

for ticker in tickers:
    df = retrieve_data(ticker, datetime(1998, 1, 1), datetime(2024, 12, 31))
    save_data(df, ticker)