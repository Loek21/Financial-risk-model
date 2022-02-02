import yfinance as yf
import pandas as pd
import numpy as np

def reader(ticker_list):
    df = yf.Ticker(ticker_list[0]).history(period='10y', interval='1mo')['Close']
    df = pd.DataFrame(df)
    df.columns = [f'{ticker_list[0]}']

    for ticker in ticker_list[1:]:
        ticker_object_data = pd.DataFrame(yf.Ticker(ticker).history(period='10y', interval='1mo')['Close'])
        ticker_object_data.columns = [f'{ticker}']

        df = pd.merge(df, ticker_object_data, how='inner', on='Date')

    df = (np.log(df) - np.log(df.shift(1)))
    df = df.dropna()

    # standard normal returns
    df = (df - df.mean())/df.std()
    
    return df
