import yfinance as yf
import pandas as pd

def download_data(ticker, start_date, end_date, output_path):
    data = yf.download(ticker, start=start_date, end=end_date, group_by='ticker')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns.values]
        data.rename(columns={
            f'{ticker}_Open': 'Open',
            f'{ticker}_High': 'High',
            f'{ticker}_Low': 'Low',
            f'{ticker}_Close': 'Close',
            f'{ticker}_Adj Close': 'Adj Close',
            f'{ticker}_Volume': 'Volume'
        }, inplace=True)
    data.to_csv(output_path)
    return data
