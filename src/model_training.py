import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def prepare_features(data, features, target):
    X = data[features]
    y = data[target]
    return X, y

def split_data(X, y, split_ratio=0.8):
    split_index = int(len(X) * split_ratio)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath):
    joblib.dump(model, filepath)
