import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# For technical indicators
import ta

# Disable warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Download data
ticker = 'GOOG'
start_date = '2010-01-01'
end_date = '2023-10-01'

data = yf.download(ticker, start=start_date, end=end_date, group_by='ticker')

# Flatten MultiIndex columns if they exist
if isinstance(data.columns, pd.MultiIndex):
    data.columns = ['_'.join(col).strip() for col in data.columns.values]
    # Adjust column names if needed
    data.rename(columns={
        f'{ticker}_Open': 'Open',
        f'{ticker}_High': 'High',
        f'{ticker}_Low': 'Low',
        f'{ticker}_Close': 'Close',
        f'{ticker}_Adj Close': 'Adj Close',
        f'{ticker}_Volume': 'Volume'
    }, inplace=True)

# Ensure 'Close' is a Series of floats
data['Close'] = data['Close'].astype(float)

# Calculate Bollinger Bands
data['MA20'] = data['Close'].rolling(window=20).mean()
data['20dSTD'] = data['Close'].rolling(window=20).std()

data['Upper'] = data['MA20'] + (data['20dSTD'] * 2)
data['Lower'] = data['MA20'] - (data['20dSTD'] * 2)

# Ensure 'Upper' and 'Lower' are Series of floats
data['Upper'] = data['Upper'].astype(float)
data['Lower'] = data['Lower'].astype(float)

# Percent B (%B)
data['PercentB'] = (data['Close'] - data['Lower']) / (data['Upper'] - data['Lower'])

# Price Returns
data['Price_Return'] = data['Close'].pct_change()

# RSI (14-day)
data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()

# MACD
macd_indicator = ta.trend.MACD(data['Close'])
data['MACD'] = macd_indicator.macd_diff()

data.dropna(inplace=True)

# Generate trading signals based on Bollinger Bands
# Buy when PercentB < 0.2 (price is near lower band)
# Sell when PercentB > 0.8 (price is near upper band)
data['Signal'] = 0
data.loc[data['PercentB'] < 0.2, 'Signal'] = 1   # Buy signal
data.loc[data['PercentB'] > 0.8, 'Signal'] = -1  # Sell signal

# Shift signal to align with next day's returns
data['Signal'] = data['Signal'].shift()
data.dropna(inplace=True)

# Target variable: Next day's return
data['Return'] = data['Close'].pct_change().shift(-1)
data.dropna(inplace=True)

# Features and target variable
features = ['PercentB', 'Price_Return', 'RSI', 'MACD']
X = data[features]
y = data['Signal']

# Split data into training and test sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Backtest the strategy
data_test = data.iloc[split_index:].copy()
data_test['Predicted_Signal'] = y_pred
data_test['Strategy_Return'] = data_test['Predicted_Signal'] * data_test['Return']
data_test['Cumulative_Strategy_Return'] = (1 + data_test['Strategy_Return']).cumprod()
data_test['Cumulative_Market_Return'] = (1 + data_test['Return']).cumprod()

# Plot cumulative returns
plt.figure(figsize=(14, 7))
plt.plot(data_test.index, data_test['Cumulative_Strategy_Return'], label='Strategy Return')
plt.plot(data_test.index, data_test['Cumulative_Market_Return'], label='Market Return')
plt.legend()
plt.title('Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.show()

# Calculate Sharpe Ratio
sharpe_ratio = (data_test['Strategy_Return'].mean() / data_test['Strategy_Return'].std()) * np.sqrt(252)
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Feature Importance
importances = model.feature_importances_
feature_names = X.columns
forest_importances = pd.Series(importances, index=feature_names)

# Plot Feature Importances
plt.figure(figsize=(10, 6))
forest_importances.sort_values().plot(kind='barh')
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Calculate the difference in cumulative returns
difference = (data_test['Cumulative_Strategy_Return'].iloc[-1] - data_test['Cumulative_Market_Return'].iloc[-1]) * 100
print(f"Difference in cumulative returns: {difference:.2f}%")
