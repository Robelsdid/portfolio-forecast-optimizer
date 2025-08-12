import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data using YFinance
    
    Parameters:
    ticker (str): Stock ticker symbol
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
    pandas.DataFrame: Historical stock data
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            print(f"No data found for {ticker}")
            return None
            
        print(f"Successfully fetched data for {ticker}")
        print(f"Data shape: {data.shape}")
        print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
        
        return data
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None

def save_data(data, ticker, output_dir):
    """
    Save data to CSV file
    
    Parameters:
    data (pandas.DataFrame): Stock data
    ticker (str): Stock ticker symbol
    output_dir (str): Output directory path
    """
    if data is not None:
        filename = f"{ticker}_data.csv"
        filepath = os.path.join(output_dir, filename)
        data.to_csv(filepath)
        print(f"Data saved to {filepath}")
    else:
        print(f"No data to save for {ticker}")

def get_output_directory():
    """
    Get the correct output directory based on current working directory
    
    Returns:
    str: Path to the output directory
    """
    current_dir = os.getcwd()
    
    # If we're in the notebooks directory, go up one level to find the main data directory
    if "notebooks" in current_dir:
        # Go up one directory to the project root
        project_root = os.path.dirname(current_dir)
        output_dir = os.path.join(project_root, "data", "raw")
    else:
        # If we're in the project root, use the data/raw directory
        output_dir = os.path.join("data", "raw")
    
    return output_dir

def main():
    """
    Main function to collect data for all assets
    """
    # Define parameters
    start_date = "2015-07-01"
    end_date = "2025-07-31"
    tickers = ["TSLA", "BND", "SPY"]
    
    # Get the correct output directory
    output_dir = get_output_directory()
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("PORTFOLIO FORECAST OPTIMIZER - DATA COLLECTION")
    print("=" * 60)
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print(f"Fetching data from {start_date} to {end_date}")
    print(f"Assets: {', '.join(tickers)}")
    print("=" * 60)
    
    # Fetch data for each ticker
    all_data = {}
    
    for ticker in tickers:
        print(f"\nFetching data for {ticker}...")
        data = fetch_stock_data(ticker, start_date, end_date)
        
        if data is not None:
            all_data[ticker] = data
            save_data(data, ticker, output_dir)
            
            # Display basic statistics
            print(f"\nBasic statistics for {ticker}:")
            print(f"  - Total trading days: {len(data)}")
            print(f"  - Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
            print(f"  - Average volume: {data['Volume'].mean():,.0f}")
        
        print("-" * 40)
    
    # Create combined dataset
    if all_data:
        print("\nCreating combined dataset...")
        combined_data = {}
        
        for ticker, data in all_data.items():
            # Select relevant columns and rename them
            combined_data[f"{ticker}_Close"] = data['Close']
            combined_data[f"{ticker}_Volume"] = data['Volume']
        
        combined_df = pd.DataFrame(combined_data)
        
        # Save combined dataset to data/raw directory
        combined_filepath = os.path.join(output_dir, "combined_data.csv")
        combined_df.to_csv(combined_filepath)
        print(f"Combined dataset saved to {combined_filepath}")
        print(f"Combined dataset shape: {combined_df.shape}")
    
    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETED")
    print("=" * 60)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  - Data saved to: {os.path.abspath(output_dir)}")
    print(f"  - Files created: {len(all_data)} individual files + 1 combined file")
    print(f"  - Date range: {start_date} to {end_date}")
    print(f"  - Assets: {', '.join(tickers)}")

if __name__ == "__main__":
    main()
