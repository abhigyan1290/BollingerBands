import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def backtest_strategy(data, split_index, y_pred, features):
    data_test = data.iloc[split_index:].copy()
    data_test['Predicted_Signal'] = y_pred
    data_test['Strategy_Return'] = data_test['Predicted_Signal'] * data_test['Return']
    data_test['Cumulative_Strategy_Return'] = (1 + data_test['Strategy_Return']).cumprod()
    data_test['Cumulative_Market_Return'] = (1 + data_test['Return']).cumprod()
    return data_test

def plot_cumulative_returns(data_test, output_path):
    plt.figure(figsize=(14, 7))
    plt.plot(data_test.index, data_test['Cumulative_Strategy_Return'], label='Strategy Return')
    plt.plot(data_test.index, data_test['Cumulative_Market_Return'], label='Market Return')
    plt.legend()
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.savefig(output_path)
    plt.close()

def calculate_sharpe_ratio(strategy_returns):
    sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
    return sharpe_ratio

def plot_feature_importance(model, feature_names, output_path):
    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names)
    plt.figure(figsize=(10, 6))
    forest_importances.sort_values().plot(kind='barh')
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.savefig(output_path)
    plt.close()

def calculate_cumulative_difference(data_test):
    difference = (data_test['Cumulative_Strategy_Return'].iloc[-1] - data_test['Cumulative_Market_Return'].iloc[-1]) * 100
    return difference
