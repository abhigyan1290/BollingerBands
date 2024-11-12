import pandas as pd
import ta

def add_technical_indicators(data):
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['20dSTD'] = data['Close'].rolling(window=20).std()
    data['Upper'] = data['MA20'] + (data['20dSTD'] * 2)
    data['Lower'] = data['MA20'] - (data['20dSTD'] * 2)
    data['PercentB'] = (data['Close'] - data['Lower']) / (data['Upper'] - data['Lower'])
    data['Price_Return'] = data['Close'].pct_change()
    data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    macd_indicator = ta.trend.MACD(data['Close'])
    data['MACD'] = macd_indicator.macd_diff()
    data.dropna(inplace=True)
    return data

def generate_signals(data):
    data['Signal'] = 0
    data.loc[data['PercentB'] < 0.2, 'Signal'] = 1   # Buy signal
    data.loc[data['PercentB'] > 0.8, 'Signal'] = -1  # Sell signal
    data['Signal'] = data['Signal'].shift()
    data.dropna(inplace=True)
    return data
