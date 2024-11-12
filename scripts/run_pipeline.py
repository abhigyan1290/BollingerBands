import sys
import os

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


from src.data_download import download_data
from src.feature_engineering import add_technical_indicators, generate_signals
from src.model_training import prepare_features, split_data, train_model, save_model
from src.evaluation import print_classification_report, plot_confusion_matrix
from src.backtest import backtest_strategy, plot_cumulative_returns, calculate_sharpe_ratio, plot_feature_importance, calculate_cumulative_difference
import pandas as pd

# Parameters
TICKER = 'GOOG'
START_DATE = '2010-01-01'
END_DATE = '2023-10-01'
DATA_RAW_PATH = 'data/raw/goog_raw.csv'
DATA_PROCESSED_PATH = 'data/processed/goog_processed.csv'
MODEL_PATH = 'models/random_forest.joblib'
CONF_MATRIX_PATH = 'plots/confusion_matrix.png'
CUM_RETURNS_PATH = 'plots/cumulative_returns.png'
FEATURE_IMPORTANCE_PATH = 'plots/feature_importance.png'

# Step 1: Download Data
data = download_data(TICKER, START_DATE, END_DATE, DATA_RAW_PATH)

# Step 2: Feature Engineering
data = add_technical_indicators(data)
data = generate_signals(data)
data['Return'] = data['Close'].pct_change().shift(-1)
data.dropna(inplace=True)
data.to_csv(DATA_PROCESSED_PATH)

# Step 3: Prepare Features and Target
features = ['PercentB', 'Price_Return', 'RSI', 'MACD']
X, y = prepare_features(data, features, 'Signal')

# Step 4: Split Data
split_ratio = 0.8
X_train, X_test, y_train, y_test = split_data(X, y, split_ratio)

# Step 5: Train Model
model = train_model(X_train, y_train)
save_model(model, MODEL_PATH)

# Step 6: Predict
y_pred = model.predict(X_test)

# Step 7: Evaluate Model
print_classification_report(y_test, y_pred)
plot_confusion_matrix(y_test, y_pred, CONF_MATRIX_PATH)

# Step 8: Backtest Strategy
split_index = int(len(X) * split_ratio)
data_test = backtest_strategy(data, split_index, y_pred, features)
plot_cumulative_returns(data_test, CUM_RETURNS_PATH)
sharpe_ratio = calculate_sharpe_ratio(data_test['Strategy_Return'])
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
plot_feature_importance(model, features, FEATURE_IMPORTANCE_PATH)
difference = calculate_cumulative_difference(data_test)
print(f"Difference in cumulative returns: {difference:.2f}%")
