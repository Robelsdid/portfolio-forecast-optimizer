import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class ARIMAModel:
    """
    ARIMA/SARIMA model for time series forecasting
    """
    
    def __init__(self, data, target_column='Close'):
        """
        Initialize ARIMA model
        
        Parameters:
        data (pd.DataFrame): Input data with datetime index
        target_column (str): Column name for target variable
        """
        self.data = data
        self.target_column = target_column
        self.model = None
        self.fitted_model = None
        self.predictions = None
        self.test_data = None
        self.train_data = None
        
    def check_stationarity(self, series):
        """
        Check if time series is stationary using Augmented Dickey-Fuller test
        
        Parameters:
        series (pd.Series): Time series to test
        
        Returns:
        dict: Test results
        """
        result = adfuller(series.dropna())
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    
    def make_stationary(self, series, max_diff=2):
        """
        Make time series stationary by differencing
        
        Parameters:
        series (pd.Series): Time series to make stationary
        max_diff (int): Maximum number of differences to try
        
        Returns:
        tuple: (stationary_series, d_value)
        """
        d = 0
        current_series = series.copy()
        
        for i in range(max_diff + 1):
            if i == 0:
                test_result = self.check_stationarity(current_series)
            else:
                current_series = current_series.diff().dropna()
                test_result = self.check_stationarity(current_series)
            
            if test_result['is_stationary']:
                break
            d = i
        
        return current_series, d
    
    def find_optimal_parameters(self, train_data, max_p=5, max_d=2, max_q=5):
        """
        Find optimal ARIMA parameters using grid search
        
        Parameters:
        train_data (pd.Series): Training data
        max_p (int): Maximum AR order
        max_d (int): Maximum differencing order
        max_q (int): Maximum MA order
        
        Returns:
        tuple: (best_p, best_d, best_q, best_aic)
        """
        best_aic = float('inf')
        best_params = (0, 0, 0)
        
        # Make series stationary first
        stationary_series, d = self.make_stationary(train_data, max_d)
        
        print(f"Series made stationary with d={d}")
        
        # Grid search for p and q
        for p in range(0, max_p + 1):
            for q in range(0, max_q + 1):
                try:
                    model = ARIMA(stationary_series, order=(p, 0, q))
                    fitted_model = model.fit()
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_params = (p, d, q)
                        
                except:
                    continue
        
        print(f"Best parameters: p={best_params[0]}, d={best_params[1]}, q={best_params[2]}")
        print(f"Best AIC: {best_aic:.2f}")
        
        return best_params[0], best_params[1], best_params[2], best_aic
    
    def split_data(self, test_size=0.2):
        """
        Split data chronologically into train and test sets
        
        Parameters:
        test_size (float): Proportion of data for testing
        """
        total_size = len(self.data)
        train_size = int(total_size * (1 - test_size))
        
        self.train_data = self.data.iloc[:train_size]
        self.test_data = self.data.iloc[train_size:]
        
        print(f"Training set: {len(self.train_data)} samples ({self.train_data.index[0]} to {self.train_data.index[-1]})")
        print(f"Test set: {len(self.test_data)} samples ({self.test_data.index[0]} to {self.test_data.index[-1]})")
        
        return self.train_data, self.test_data
    
    def fit_model(self, p=1, d=1, q=1):
        """
        Fit ARIMA model with specified parameters
        
        Parameters:
        p (int): AR order
        d (int): Differencing order
        q (int): MA order
        """
        train_series = self.train_data[self.target_column]
        
        # Make stationary if needed
        if d > 0:
            stationary_series, _ = self.make_stationary(train_series, d)
        else:
            stationary_series = train_series
        
        # Fit model
        self.model = ARIMA(stationary_series, order=(p, 0, q))
        self.fitted_model = self.model.fit()
        
        print(f"ARIMA({p},{d},{q}) model fitted successfully")
        print(f"AIC: {self.fitted_model.aic:.2f}")
        
        return self.fitted_model
    
    def forecast(self, steps=None):
        """
        Generate forecasts
        
        Parameters:
        steps (int): Number of steps to forecast (default: length of test set)
        """
        if steps is None:
            steps = len(self.test_data)
        
        # Generate forecasts
        forecast = self.fitted_model.forecast(steps=steps)
        
        # Create forecast index
        last_date = self.train_data.index[-1]
        forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                     periods=steps, freq='B')
        
        self.predictions = pd.Series(forecast.values, index=forecast_index)
        
        return self.predictions
    
    def evaluate_model(self):
        """
        Evaluate model performance using multiple metrics
        
        Returns:
        dict: Evaluation metrics
        """
        if self.predictions is None or self.test_data is None:
            raise ValueError("Model must be fitted and forecasted before evaluation")
        
        actual = self.test_data[self.target_column]
        predicted = self.predictions[:len(actual)]
        
        # Calculate metrics
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        }
        
        print("Model Performance Metrics:")
        print(f"MAE: ${mae:.2f}")
        print(f"MSE: ${mse:.2f}")
        print(f"RMSE: ${rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
        
        return metrics
    
    def plot_results(self):
        """
        Plot actual vs predicted values
        """
        if self.predictions is None or self.test_data is None:
            raise ValueError("Model must be fitted and forecasted before plotting")
        
        plt.figure(figsize=(15, 8))
        
        # Plot training data
        plt.plot(self.train_data.index, self.train_data[self.target_column], 
                label='Training Data', color='blue', alpha=0.7)
        
        # Plot test data
        plt.plot(self.test_data.index, self.test_data[self.target_column], 
                label='Actual Test Data', color='green', linewidth=2)
        
        # Plot predictions
        plt.plot(self.predictions.index, self.predictions, 
                label='Predictions', color='red', linewidth=2, linestyle='--')
        
        plt.title('ARIMA Model: Actual vs Predicted Stock Prices')
        plt.xlabel('Date')
        plt.ylabel('Stock Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self):
        """
        Plot model residuals for diagnostic analysis
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before plotting residuals")
        
        residuals = self.fitted_model.resid
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuals over time
        axes[0, 0].plot(residuals.index, residuals)
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Residuals Distribution')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # ACF of residuals
        plot_acf(residuals, ax=axes[1, 0], lags=40)
        axes[1, 0].set_title('ACF of Residuals')
        
        # PACF of residuals
        plot_pacf(residuals, ax=axes[1, 1], lags=40)
        axes[1, 1].set_title('PACF of Residuals')
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self, test_size=0.2):
        """
        Run complete ARIMA analysis pipeline
        
        Parameters:
        test_size (float): Proportion of data for testing
        
        Returns:
        dict: Complete analysis results
        """
        print("=" * 60)
        print("ARIMA MODEL ANALYSIS")
        print("=" * 60)
        
        # Step 1: Split data
        print("\n1. Splitting data...")
        self.split_data(test_size)
        
        # Step 2: Find optimal parameters
        print("\n2. Finding optimal parameters...")
        p, d, q, aic = self.find_optimal_parameters(self.train_data[self.target_column])
        
        # Step 3: Fit model
        print("\n3. Fitting model...")
        self.fit_model(p, d, q)
        
        # Step 4: Generate forecasts
        print("\n4. Generating forecasts...")
        self.forecast()
        
        # Step 5: Evaluate model
        print("\n5. Evaluating model...")
        metrics = self.evaluate_model()
        
        # Step 6: Plot results
        print("\n6. Creating visualizations...")
        self.plot_results()
        self.plot_residuals()
        
        results = {
            'parameters': (p, d, q),
            'aic': aic,
            'metrics': metrics,
            'predictions': self.predictions,
            'model': self.fitted_model
        }
        
        print("\n" + "=" * 60)
        print("ARIMA ANALYSIS COMPLETED")
        print("=" * 60)
        
        return results
