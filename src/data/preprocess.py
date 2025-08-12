import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Class for preprocessing financial time series data
    """
    
    def __init__(self, data_dir="data/raw"):
        # Get the correct data directory path
        import os
        current_dir = os.getcwd()
        
        # If we're in the notebooks directory, go up one level to find the main data directory
        if "notebooks" in current_dir:
            # Go up one directory to the project root
            project_root = os.path.dirname(current_dir)
            self.data_dir = os.path.join(project_root, data_dir)
        else:
            # If we're in the project root, use the data directory there
            self.data_dir = data_dir
            
        self.tickers = ["TSLA", "BND", "SPY"]
        print(f"DataPreprocessor initialized with data directory: {os.path.abspath(self.data_dir)}")
        
        # Verify the directory exists
        if not os.path.exists(self.data_dir):
            print(f"WARNING: Data directory does not exist: {self.data_dir}")
        else:
            print(f"Data directory exists and contains:")
            for file in os.listdir(self.data_dir):
                print(f"  - {file}")
        
    def load_data(self, ticker):
        """
        Load data for a specific ticker
        
        Parameters:
        ticker (str): Stock ticker symbol
        
        Returns:
        pandas.DataFrame: Loaded data
        """
        filepath = os.path.join(self.data_dir, f"{ticker}_data.csv")
        print(f"Looking for file: {filepath}")
        if os.path.exists(filepath):
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            print(f"Loaded data for {ticker}: {data.shape}")
            return data
        else:
            print(f"Data file not found for {ticker}")
            print(f"Available files in {self.data_dir}:")
            if os.path.exists(self.data_dir):
                for file in os.listdir(self.data_dir):
                    print(f"  - {file}")
            return None
    
    def clean_data(self, data, ticker):
        """
        Clean and preprocess the data
        
        Parameters:
        data (pandas.DataFrame): Raw stock data
        ticker (str): Stock ticker symbol
        
        Returns:
        pandas.DataFrame: Cleaned data
        """
        if data is None:
            return None
            
        print(f"\nCleaning data for {ticker}...")
        
        # Check for missing values
        missing_values = data.isnull().sum()
        print(f"Missing values:\n{missing_values}")
        
        # Handle missing values
        if missing_values.sum() > 0:
            # Forward fill for small gaps, then backward fill
            data = data.fillna(method='ffill').fillna(method='bfill')
            print("Missing values filled using forward/backward fill")
        
        # Remove any remaining rows with NaN values
        data = data.dropna()
        
        # Ensure data types are correct
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove outliers (prices that are 0 or negative)
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in data.columns:
                data = data[data[col] > 0]
        
        print(f"Cleaned data shape: {data.shape}")
        return data
    
    def calculate_returns(self, data, ticker):
        """
        Calculate daily returns and other financial metrics
        
        Parameters:
        data (pandas.DataFrame): Clean stock data
        ticker (str): Stock ticker symbol
        
        Returns:
        pandas.DataFrame: Data with additional features
        """
        if data is None:
            return None
            
        print(f"\nCalculating returns for {ticker}...")
        
        # Calculate daily returns
        data[f'{ticker}_Daily_Return'] = data['Close'].pct_change()
        
        # Calculate log returns
        data[f'{ticker}_Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Calculate volatility (rolling standard deviation of returns)
        data[f'{ticker}_Volatility_20d'] = data[f'{ticker}_Daily_Return'].rolling(window=20).std()
        data[f'{ticker}_Volatility_60d'] = data[f'{ticker}_Daily_Return'].rolling(window=60).std()
        
        # Calculate moving averages
        data[f'{ticker}_MA_20d'] = data['Close'].rolling(window=20).mean()
        data[f'{ticker}_MA_60d'] = data['Close'].rolling(window=60).mean()
        data[f'{ticker}_MA_200d'] = data['Close'].rolling(window=200).mean()
        
        # Calculate price momentum
        data[f'{ticker}_Momentum_5d'] = data['Close'] / data['Close'].shift(5) - 1
        data[f'{ticker}_Momentum_20d'] = data['Close'] / data['Close'].shift(20) - 1
        
        # Calculate volume metrics
        data[f'{ticker}_Volume_MA_20d'] = data['Volume'].rolling(window=20).mean()
        data[f'{ticker}_Volume_Ratio'] = data['Volume'] / data[f'{ticker}_Volume_MA_20d']
        
        # Remove NaN values from calculations
        data = data.dropna()
        
        print(f"Features calculated. Final shape: {data.shape}")
        return data
    
    def test_stationarity(self, data, column, ticker):
        """
        Perform Augmented Dickey-Fuller test for stationarity
        
        Parameters:
        data (pandas.DataFrame): Time series data
        column (str): Column name to test
        ticker (str): Stock ticker symbol
        
        Returns:
        dict: Test results
        """
        print(f"\nTesting stationarity for {ticker} {column}...")
        
        # Perform ADF test
        result = adfuller(data[column].dropna())
        
        # Extract results
        adf_statistic = result[0]
        p_value = result[1]
        critical_values = result[4]
        
        # Determine if series is stationary
        is_stationary = p_value < 0.05
        
        results = {
            'ticker': ticker,
            'column': column,
            'adf_statistic': adf_statistic,
            'p_value': p_value,
            'critical_values': critical_values,
            'is_stationary': is_stationary
        }
        
        print(f"ADF Statistic: {adf_statistic:.6f}")
        print(f"p-value: {p_value:.6f}")
        print(f"Critical values: {critical_values}")
        print(f"Stationary: {'Yes' if is_stationary else 'No'}")
        
        return results
    
    def analyze_seasonality(self, data, column, ticker, period=252):
        """
        Analyze seasonality in the time series
        
        Parameters:
        data (pandas.DataFrame): Time series data
        column (str): Column name to analyze
        ticker (str): Stock ticker symbol
        period (int): Period for seasonal decomposition (252 for daily data)
        
        Returns:
        dict: Seasonality analysis results
        """
        print(f"\nAnalyzing seasonality for {ticker} {column}...")
        
        try:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(data[column].dropna(), period=period, extrapolate_trend='freq')
            
            # Calculate strength of trend and seasonality
            trend_strength = np.var(decomposition.trend.dropna()) / np.var(data[column].dropna())
            seasonal_strength = np.var(decomposition.seasonal.dropna()) / np.var(data[column].dropna())
            
            results = {
                'ticker': ticker,
                'column': column,
                'trend_strength': trend_strength,
                'seasonal_strength': seasonal_strength,
                'decomposition': decomposition
            }
            
            print(f"Trend strength: {trend_strength:.4f}")
            print(f"Seasonal strength: {seasonal_strength:.4f}")
            
            return results
            
        except Exception as e:
            print(f"Error in seasonality analysis: {str(e)}")
            return None
    
    def detect_outliers(self, data, column, ticker, method='iqr'):
        """
        Detect outliers in the data
        
        Parameters:
        data (pandas.DataFrame): Time series data
        column (str): Column name to analyze
        ticker (str): Stock ticker symbol
        method (str): Method for outlier detection ('iqr' or 'zscore')
        
        Returns:
        dict: Outlier detection results
        """
        print(f"\nDetecting outliers for {ticker} {column}...")
        
        series = data[column].dropna()
        
        if method == 'iqr':
            # IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
        elif method == 'zscore':
            # Z-score method
            z_scores = np.abs((series - series.mean()) / series.std())
            outliers = series[z_scores > 3]
        
        results = {
            'ticker': ticker,
            'column': column,
            'method': method,
            'outlier_count': len(outliers),
            'outlier_percentage': len(outliers) / len(series) * 100,
            'outliers': outliers
        }
        
        print(f"Outliers detected: {len(outliers)} ({len(outliers)/len(series)*100:.2f}%)")
        
        return results
    
    def process_all_data(self):
        """
        Process all ticker data and return combined results
        
        Returns:
        dict: Processed data for all tickers
        """
        processed_data = {}
        stationarity_results = []
        seasonality_results = []
        outlier_results = []
        
        print("=" * 60)
        print("DATA PREPROCESSING")
        print("=" * 60)
        
        for ticker in self.tickers:
            print(f"\nProcessing {ticker}...")
            
            # Load data
            data = self.load_data(ticker)
            if data is None:
                continue
            
            # Clean data
            data = self.clean_data(data, ticker)
            if data is None:
                continue
            
            # Calculate returns and features
            data = self.calculate_returns(data, ticker)
            if data is None:
                continue
            
            # Test stationarity
            stationarity_result = self.test_stationarity(data, 'Close', ticker)
            stationarity_results.append(stationarity_result)
            
            # Analyze seasonality
            seasonality_result = self.analyze_seasonality(data, 'Close', ticker)
            if seasonality_result:
                seasonality_results.append(seasonality_result)
            
            # Detect outliers
            outlier_result = self.detect_outliers(data, 'Close', ticker)
            outlier_results.append(outlier_result)
            
            # Store processed data
            processed_data[ticker] = data
            
            # Save processed data to the processed directory
            processed_dir = os.path.join(os.path.dirname(self.data_dir), "processed")
            os.makedirs(processed_dir, exist_ok=True)
            output_file = os.path.join(processed_dir, f"{ticker}_processed.csv")
            data.to_csv(output_file)
            print(f"Processed data saved to {output_file}")
        
        # Create summary
        summary = {
            'processed_data': processed_data,
            'stationarity_results': stationarity_results,
            'seasonality_results': seasonality_results,
            'outlier_results': outlier_results
        }
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETED")
        print("=" * 60)
        
        return summary

def main():
    """
    Main function to run preprocessing
    """
    preprocessor = DataPreprocessor()
    summary = preprocessor.process_all_data()
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-" * 40)
    
    for ticker, data in summary['processed_data'].items():
        print(f"\n{ticker}:")
        print(f"  - Data points: {len(data)}")
        print(f"  - Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"  - Features: {len(data.columns)}")
        
        # Basic statistics for returns
        return_col = f'{ticker}_Daily_Return'
        if return_col in data.columns:
            returns = data[return_col].dropna()
            print(f"  - Mean daily return: {returns.mean():.4f}")
            print(f"  - Std daily return: {returns.std():.4f}")
            print(f"  - Min daily return: {returns.min():.4f}")
            print(f"  - Max daily return: {returns.max():.4f}")

if __name__ == "__main__":
    main()
