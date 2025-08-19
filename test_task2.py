#!/usr/bin/env python3
"""
Test script for Task 2: Time Series Forecasting Models
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_data_loading():
    """Test if we can load the processed data"""
    print("Testing data loading...")
    
    try:
        tsla_data = pd.read_csv('data/processed/TSLA_processed.csv', index_col=0, parse_dates=True)
        print(f" Data loaded successfully: {tsla_data.shape}")
        print(f" Date range: {tsla_data.index[0]} to {tsla_data.index[-1]}")
        print(f" Columns: {list(tsla_data.columns)}")
        return tsla_data
    except Exception as e:
        print(f" Error loading data: {e}")
        return None

def test_imports():
    """Test if all required modules can be imported"""
    print("\nTesting imports...")
    
    try:
        from src.models.arima_model import ARIMAModel
        print(" ARIMA model imported successfully")
    except Exception as e:
        print(f" Error importing ARIMA model: {e}")
        return False
    
    try:
        from src.models.lstm_model import LSTMModel
        print(" LSTM model imported successfully")
    except Exception as e:
        print(f" Error importing LSTM model: {e}")
        return False
    
    try:
        from src.models.model_comparison import ModelComparison
        print(" Model comparison imported successfully")
    except Exception as e:
        print(f" Error importing model comparison: {e}")
        return False
    
    return True

def test_arima_model(data):
    """Test ARIMA model functionality"""
    print("\nTesting ARIMA model...")
    
    try:
        from src.models.arima_model import ARIMAModel
        
        # Initialize model
        arima_model = ARIMAModel(data, target_column='Close')
        print(" ARIMA model initialized")
        
        # Split data
        train_data, test_data = arima_model.split_data(test_size=0.1)
        print(" Data split completed")
        
        # Test parameter optimization (with limited search for speed)
        p, d, q, aic = arima_model.find_optimal_parameters(
            train_data['Close'], max_p=2, max_d=1, max_q=2
        )
        print(f" Parameter optimization completed: p={p}, d={d}, q={q}")
        
        return True
    except Exception as e:
        print(f" Error testing ARIMA model: {e}")
        return False

def test_lstm_model(data):
    """Test LSTM model functionality"""
    print("\nTesting LSTM model...")
    
    try:
        from src.models.lstm_model import LSTMModel
        
        # Initialize model
        lstm_model = LSTMModel(data, target_column='Close', sequence_length=30)
        print(" LSTM model initialized")
        
        # Prepare data
        X_train, y_train, X_test, y_test = lstm_model.prepare_data(test_size=0.1)
        print(" Data preparation completed")
        
        # Test model building
        model = lstm_model.build_model(units=20, layers=1, dropout=0.1, learning_rate=0.001)
        print(" LSTM model built successfully")
        
        return True
    except Exception as e:
        print(f" Error testing LSTM model: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("TASK 2 TESTING")
    print("="*60)
    
    # Test 1: Data loading
    data = test_data_loading()
    if data is None:
        print(" Data loading failed. Cannot proceed with tests.")
        return False
    
    # Test 2: Imports
    if not test_imports():
        print(" Import tests failed. Cannot proceed with tests.")
        return False
    
    # Test 3: ARIMA model
    if not test_arima_model(data):
        print(" ARIMA model test failed.")
        return False
    
    # Test 4: LSTM model
    if not test_lstm_model(data):
        print(" LSTM model test failed.")
        return False
    
    print("\n" + "="*60)
    print(" ALL TESTS PASSED!")
    print("Task 2 implementation is ready to run.")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
