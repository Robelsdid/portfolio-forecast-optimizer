import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from .arima_model import ARIMAModel
from .lstm_model import LSTMModel

class ModelComparison:
    """
    Compare ARIMA and LSTM models for time series forecasting
    """
    
    def __init__(self, data, target_column='Close'):
        """
        Initialize model comparison
        
        Parameters:
        data (pd.DataFrame): Input data with datetime index
        target_column (str): Column name for target variable
        """
        self.data = data
        self.target_column = target_column
        self.arima_results = None
        self.lstm_results = None
        self.comparison_results = None
        
    def run_arima_analysis(self, test_size=0.2):
        """
        Run ARIMA model analysis
        
        Parameters:
        test_size (float): Proportion of data for testing
        
        Returns:
        dict: ARIMA results
        """
        print("\n" + "="*60)
        print("RUNNING ARIMA MODEL ANALYSIS")
        print("="*60)
        
        arima_model = ARIMAModel(self.data, self.target_column)
        arima_model.split_data(test_size)
        self.test_data = arima_model.test_data  # Store test data for consistency
        self.arima_results = arima_model.run_complete_analysis(test_size)
        
        return self.arima_results
    
    def run_lstm_analysis(self, test_size=0.2, optimize_hyperparams=True):
        """
        Run LSTM model analysis
        
        Parameters:
        test_size (float): Proportion of data for testing
        optimize_hyperparams (bool): Whether to optimize hyperparameters
        
        Returns:
        dict: LSTM results
        """
        print("\n" + "="*60)
        print("RUNNING LSTM MODEL ANALYSIS")
        print("="*60)
        
        lstm_model = LSTMModel(self.data, self.target_column)
        lstm_model.split_data(test_size)
        # Use the same test data as ARIMA for consistency
        if hasattr(self, 'test_data'):
            lstm_model.test_data = self.test_data
            lstm_model.train_data = self.data.iloc[:len(self.data) - len(self.test_data)]
        self.lstm_results = lstm_model.run_complete_analysis(test_size, optimize_hyperparams)
        
        return self.lstm_results
    
    def compare_models(self):
        """
        Compare ARIMA and LSTM model performance
        
        Returns:
        dict: Comparison results
        """
        if self.arima_results is None or self.lstm_results is None:
            raise ValueError("Both models must be run before comparison")
        
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        
        # Extract metrics
        arima_metrics = self.arima_results['metrics']
        lstm_metrics = self.lstm_results['metrics']
        
        # Create comparison table
        comparison_data = {
            'Metric': ['MAE', 'MSE', 'RMSE', 'MAPE'],
            'ARIMA': [
                f"${arima_metrics['MAE']:.2f}",
                f"${arima_metrics['MSE']:.2f}",
                f"${arima_metrics['RMSE']:.2f}",
                f"{arima_metrics['MAPE']:.2f}%"
            ],
            'LSTM': [
                f"${lstm_metrics['MAE']:.2f}",
                f"${lstm_metrics['MSE']:.2f}",
                f"${lstm_metrics['RMSE']:.2f}",
                f"{lstm_metrics['MAPE']:.2f}%"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Determine best model for each metric
        best_model = []
        for metric in ['MAE', 'MSE', 'RMSE', 'MAPE']:
            if arima_metrics[metric] < lstm_metrics[metric]:
                best_model.append('ARIMA')
            else:
                best_model.append('LSTM')
        
        comparison_df['Best Model'] = best_model
        
        print("\nPerformance Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Calculate improvement percentages
        improvements = {}
        for metric in ['MAE', 'MSE', 'RMSE', 'MAPE']:
            arima_val = arima_metrics[metric]
            lstm_val = lstm_metrics[metric]
            
            if arima_val < lstm_val:
                improvement = ((arima_val - lstm_val) / arima_val) * 100
                improvements[metric] = f"ARIMA better by {abs(improvement):.2f}%"
            else:
                improvement = ((lstm_val - arima_val) / lstm_val) * 100
                improvements[metric] = f"LSTM better by {abs(improvement):.2f}%"
        
        print("\nImprovement Analysis:")
        for metric, improvement in improvements.items():
            print(f"{metric}: {improvement}")
        
        # Overall winner
        arima_wins = sum(1 for model in best_model if model == 'ARIMA')
        lstm_wins = sum(1 for model in best_model if model == 'LSTM')
        
        if arima_wins > lstm_wins:
            overall_winner = "ARIMA"
        elif lstm_wins > arima_wins:
            overall_winner = "LSTM"
        else:
            overall_winner = "Tie"
        
        print(f"\nOverall Winner: {overall_winner}")
        print(f"ARIMA wins: {arima_wins}/4 metrics")
        print(f"LSTM wins: {lstm_wins}/4 metrics")
        
        self.comparison_results = {
            'comparison_table': comparison_df,
            'improvements': improvements,
            'overall_winner': overall_winner,
            'arima_wins': arima_wins,
            'lstm_wins': lstm_wins
        }
        
        return self.comparison_results
    
    def plot_comparison(self):
        """
        Plot comparison of model predictions
        """
        if self.arima_results is None or self.lstm_results is None:
            raise ValueError("Both models must be run before plotting comparison")
        
        plt.figure(figsize=(15, 10))
        
        # Use the stored test data for consistency
        test_data = self.test_data
        
        plt.subplot(2, 1, 1)
        # Plot training data (first 80% of data)
        train_size = int(len(self.data) * 0.8)
        train_data = self.data.iloc[:train_size]
        plt.plot(train_data.index, train_data[self.target_column], 
                label='Training Data', color='blue', alpha=0.7)
        
        # Plot test data
        plt.plot(test_data.index, test_data[self.target_column], 
                label='Actual Test Data', color='green', linewidth=2)
        
        # Plot predictions - ensure they align with test data
        arima_pred = self.arima_results['predictions']
        lstm_pred = self.lstm_results['predictions']
        
        # Align predictions with test data length
        if len(arima_pred) > len(test_data):
            arima_pred = arima_pred[:len(test_data)]
        if len(lstm_pred) > len(test_data):
            lstm_pred = lstm_pred[:len(test_data)]
        
        plt.plot(test_data.index[:len(arima_pred)], arima_pred, 
                label='ARIMA Predictions', color='red', linewidth=2, linestyle='--')
        plt.plot(test_data.index[:len(lstm_pred)], lstm_pred, 
                label='LSTM Predictions', color='orange', linewidth=2, linestyle=':')
        
        plt.title('Model Comparison: Actual vs Predicted Stock Prices')
        plt.xlabel('Date')
        plt.ylabel('Stock Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Plot residuals comparison
        plt.subplot(2, 1, 2)
        actual = test_data[self.target_column]
        
        # Ensure predictions and actual data have the same length
        min_len = min(len(actual), len(arima_pred), len(lstm_pred))
        actual = actual[:min_len]
        arima_pred = arima_pred[:min_len]
        lstm_pred = lstm_pred[:min_len]
        
        arima_residuals = actual - arima_pred
        lstm_residuals = actual - lstm_pred
        
        plt.plot(test_data.index[:min_len], arima_residuals, 
                label='ARIMA Residuals', color='red', alpha=0.7)
        plt.plot(test_data.index[:min_len], lstm_residuals, 
                label='LSTM Residuals', color='orange', alpha=0.7)
        
        plt.title('Model Residuals Comparison')
        plt.xlabel('Date')
        plt.ylabel('Residuals ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_metrics_comparison(self):
        """
        Plot metrics comparison as bar chart
        """
        if self.comparison_results is None:
            raise ValueError("Models must be compared before plotting metrics")
        
        arima_metrics = self.arima_results['metrics']
        lstm_metrics = self.lstm_results['metrics']
        
        metrics = ['MAE', 'MSE', 'RMSE', 'MAPE']
        arima_values = [arima_metrics[m] for m in metrics]
        lstm_values = [lstm_metrics[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, arima_values, width, label='ARIMA', color='red', alpha=0.7)
        bars2 = ax.bar(x + width/2, lstm_values, width, label='LSTM', color='orange', alpha=0.7)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Error Values')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        """
        Generate comprehensive comparison report
        
        Returns:
        str: Report text
        """
        if self.comparison_results is None:
            raise ValueError("Models must be compared before generating report")
        
        report = f"""
{'='*80}
TIME SERIES FORECASTING MODEL COMPARISON REPORT
{'='*80}

MODEL PERFORMANCE SUMMARY:
{'-'*40}

{self.comparison_results['comparison_table'].to_string(index=False)}

IMPROVEMENT ANALYSIS:
{'-'*20}
"""
        
        for metric, improvement in self.comparison_results['improvements'].items():
            report += f"{metric}: {improvement}\n"
        
        report += f"""
OVERALL RESULTS:
{'-'*15}
Overall Winner: {self.comparison_results['overall_winner']}
ARIMA wins: {self.comparison_results['arima_wins']}/4 metrics
LSTM wins: {self.comparison_results['lstm_wins']}/4 metrics

MODEL CHARACTERISTICS:
{'-'*25}
ARIMA Model:
- Parameters: {self.arima_results['parameters']}
- AIC: {self.arima_results['aic']:.2f}
- Type: Statistical/Classical
- Interpretability: High
- Training Time: Fast

LSTM Model:
- Hyperparameters: {self.lstm_results['hyperparameters']}
- Type: Deep Learning
- Interpretability: Low
- Training Time: Slow
- Complexity: High

RECOMMENDATIONS:
{'-'*15}
"""
        
        if self.comparison_results['overall_winner'] == 'ARIMA':
            report += """
1. ARIMA performs better overall for this dataset
2. Consider using ARIMA for:
   - Quick prototyping and analysis
   - When interpretability is important
   - When computational resources are limited
3. LSTM may be useful for:
   - Capturing complex non-linear patterns
   - When more data becomes available
   - When computational resources are abundant
"""
        elif self.comparison_results['overall_winner'] == 'LSTM':
            report += """
1. LSTM performs better overall for this dataset
2. Consider using LSTM for:
   - Final production models
   - When accuracy is the primary concern
   - When complex patterns need to be captured
3. ARIMA may still be useful for:
   - Baseline comparison
   - When interpretability is required
   - Quick initial analysis
"""
        else:
            report += """
1. Both models perform similarly
2. Consider using both models in ensemble
3. ARIMA for interpretability and quick analysis
4. LSTM for capturing complex patterns
5. Combine predictions for potentially better results
"""
        
        report += f"""
{'='*80}
REPORT GENERATED: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
        
        return report
    
    def run_complete_comparison(self, test_size=0.2, optimize_lstm=True):
        """
        Run complete model comparison pipeline
        
        Parameters:
        test_size (float): Proportion of data for testing
        optimize_lstm (bool): Whether to optimize LSTM hyperparameters
        
        Returns:
        dict: Complete comparison results
        """
        print("="*80)
        print("COMPLETE MODEL COMPARISON PIPELINE")
        print("="*80)
        
        # Step 1: Run ARIMA analysis
        self.run_arima_analysis(test_size)
        
        # Step 2: Run LSTM analysis
        self.run_lstm_analysis(test_size, optimize_lstm)
        
        # Step 3: Compare models
        self.compare_models()
        
        # Step 4: Generate visualizations
        print("\nGenerating comparison visualizations...")
        self.plot_comparison()
        self.plot_metrics_comparison()
        
        # Step 5: Generate report
        print("\nGenerating comprehensive report...")
        report = self.generate_report()
        print(report)
        
        # Save report to file
        with open('results/model_comparison_report.txt', 'w') as f:
            f.write(report)
        
        print(f"\nReport saved to: results/model_comparison_report.txt")
        
        return {
            'arima_results': self.arima_results,
            'lstm_results': self.lstm_results,
            'comparison_results': self.comparison_results,
            'report': report
        }
