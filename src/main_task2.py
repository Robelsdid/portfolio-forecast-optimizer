#!/usr/bin/env python3
"""
Task 2: Time Series Forecasting Models
Portfolio Forecast Optimizer - GMF Investments

This script implements and compares ARIMA and LSTM models for Tesla stock price forecasting.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.model_comparison import ModelComparison
from src.data.preprocess import DataPreprocessor

def main():
    """
    Main function to run Task 2 analysis
    """
    print("="*80)
    print("TASK 2: TIME SERIES FORECASTING MODELS")
    print("PORTFOLIO FORECAST OPTIMIZER - GMF INVESTMENTS")
    print("="*80)
    
    # Step 1: Load and prepare data
    print("\nStep 1: Loading and preparing data...")
    print("-" * 50)
    
    try:
        # Load processed TSLA data
        tsla_data = pd.read_csv('data/processed/TSLA_processed.csv', index_col=0, parse_dates=True)
        print(f"Loaded TSLA data: {tsla_data.shape}")
        print(f"Date range: {tsla_data.index[0]} to {tsla_data.index[-1]}")
        
        # Check if we have the required columns
        if 'Close' not in tsla_data.columns:
            print("Error: 'Close' column not found in data")
            return
        
        # Display basic statistics
        print(f"\nBasic statistics for TSLA Close prices:")
        print(f"Mean: ${tsla_data['Close'].mean():.2f}")
        print(f"Std: ${tsla_data['Close'].std():.2f}")
        print(f"Min: ${tsla_data['Close'].min():.2f}")
        print(f"Max: ${tsla_data['Close'].max():.2f}")
        
    except FileNotFoundError:
        print("Error: TSLA processed data not found. Please run Task 1 first.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Step 2: Initialize model comparison
    print("\nStep 2: Initializing model comparison...")
    print("-" * 50)
    
    try:
        comparison = ModelComparison(tsla_data, target_column='Close')
        print("Model comparison initialized successfully")
    except Exception as e:
        print(f"Error initializing model comparison: {e}")
        return
    
    # Step 3: Run complete analysis
    print("\nStep 3: Running complete model comparison...")
    print("-" * 50)
    
    try:
        # Run complete comparison pipeline
        results = comparison.run_complete_comparison(
            test_size=0.2,  # Use 20% of data for testing
            optimize_lstm=True  # Optimize LSTM hyperparameters
        )
        
        print("\n" + "="*80)
        print("TASK 2 COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Display summary
        print("\nSUMMARY:")
        print(f"- ARIMA Model: {results['arima_results']['parameters']}")
        print(f"- LSTM Model: {results['lstm_results']['hyperparameters']}")
        print(f"- Overall Winner: {results['comparison_results']['overall_winner']}")
        print(f"- Report saved to: results/model_comparison_report.txt")
        
        return results
        
    except Exception as e:
        print(f"Error during model comparison: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_individual_models():
    """
    Run individual models separately for testing
    """
    print("="*80)
    print("RUNNING INDIVIDUAL MODELS")
    print("="*80)
    
    # Load data
    tsla_data = pd.read_csv('data/processed/TSLA_processed.csv', index_col=0, parse_dates=True)
    
    # Initialize comparison
    comparison = ModelComparison(tsla_data, target_column='Close')
    
    # Run ARIMA only
    print("\nRunning ARIMA model only...")
    arima_results = comparison.run_arima_analysis(test_size=0.2)
    
    # Run LSTM only
    print("\nRunning LSTM model only...")
    lstm_results = comparison.run_lstm_analysis(test_size=0.2, optimize_hyperparams=False)
    
    return arima_results, lstm_results

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--individual":
        # Run individual models
        results = run_individual_models()
    else:
        # Run complete comparison
        results = main()
    
    if results:
        print("\nTask 2 completed successfully!")
    else:
        print("\nTask 2 failed. Please check the error messages above.")
