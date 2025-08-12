import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class FinancialVisualizer:
    """
    Class for creating financial data visualizations
    """
    
    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_price_series(self, data_dict, save_plot=True):
        """
        Plot closing prices for all assets
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('Historical Closing Prices', fontsize=16, fontweight='bold')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, (ticker, data) in enumerate(data_dict.items()):
            axes[i].plot(data.index, data['Close'], color=colors[i], linewidth=1.5)
            axes[i].set_title(f'{ticker} Closing Price', fontweight='bold')
            axes[i].set_ylabel('Price ($)')
            axes[i].grid(True, alpha=0.3)
            
            # Add moving averages
            if f'{ticker}_MA_20d' in data.columns:
                axes[i].plot(data.index, data[f'{ticker}_MA_20d'], 
                           color='red', alpha=0.7, linewidth=1, label='20-day MA')
            if f'{ticker}_MA_60d' in data.columns:
                axes[i].plot(data.index, data[f'{ticker}_MA_60d'], 
                           color='orange', alpha=0.7, linewidth=1, label='60-day MA')
            
            axes[i].legend()
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(self.results_dir, 'price_series.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_returns_analysis(self, data_dict, save_plot=True):
        """
        Plot daily returns analysis
        """
        fig, axes = plt.subplots(3, 2, figsize=(18, 12))
        fig.suptitle('Daily Returns Analysis', fontsize=16, fontweight='bold')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, (ticker, data) in enumerate(data_dict.items()):
            return_col = f'{ticker}_Daily_Return'
            if return_col not in data.columns:
                continue
                
            returns = data[return_col].dropna()
            
            # Time series of returns
            axes[i, 0].plot(returns.index, returns, color=colors[i], alpha=0.7, linewidth=0.8)
            axes[i, 0].set_title(f'{ticker} Daily Returns')
            axes[i, 0].set_ylabel('Returns')
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Histogram of returns
            axes[i, 1].hist(returns, bins=50, color=colors[i], alpha=0.7, edgecolor='black')
            axes[i, 1].set_title(f'{ticker} Returns Distribution')
            axes[i, 1].set_xlabel('Returns')
            axes[i, 1].set_ylabel('Frequency')
            axes[i, 1].grid(True, alpha=0.3)
            
            # Add normal distribution overlay
            mu, sigma = returns.mean(), returns.std()
            x = np.linspace(returns.min(), returns.max(), 100)
            y = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            axes[i, 1].plot(x, y * len(returns) * (returns.max() - returns.min()) / 50, 
                           'r-', linewidth=2, label='Normal Distribution')
            axes[i, 1].legend()
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(self.results_dir, 'returns_analysis.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_volatility_analysis(self, data_dict, save_plot=True):
        """
        Plot volatility analysis
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('Volatility Analysis', fontsize=16, fontweight='bold')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, (ticker, data) in enumerate(data_dict.items()):
            vol_20d = f'{ticker}_Volatility_20d'
            vol_60d = f'{ticker}_Volatility_60d'
            
            if vol_20d in data.columns:
                axes[i].plot(data.index, data[vol_20d], color=colors[i], 
                           linewidth=1.5, label='20-day Volatility')
            if vol_60d in data.columns:
                axes[i].plot(data.index, data[vol_60d], color='red', 
                           linewidth=1.5, label='60-day Volatility')
            
            axes[i].set_title(f'{ticker} Rolling Volatility')
            axes[i].set_ylabel('Volatility')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(self.results_dir, 'volatility_analysis.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_matrix(self, data_dict, save_plot=True):
        """
        Plot correlation matrix of returns
        """
        # Create returns dataframe
        returns_data = {}
        for ticker, data in data_dict.items():
            return_col = f'{ticker}_Daily_Return'
            if return_col in data.columns:
                returns_data[ticker] = data[return_col]
        
        if len(returns_data) < 2:
            print("Need at least 2 assets for correlation analysis")
            return
        
        returns_df = pd.DataFrame(returns_data).dropna()
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Daily Returns', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(self.results_dir, 'correlation_matrix.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
        return corr_matrix
    
    def plot_stationarity_test(self, stationarity_results, save_plot=True):
        """
        Plot stationarity test results
        """
        if not stationarity_results:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Stationarity Test Results', fontsize=16, fontweight='bold')
        
        # Extract data
        tickers = [result['ticker'] for result in stationarity_results]
        adf_stats = [result['adf_statistic'] for result in stationarity_results]
        p_values = [result['p_value'] for result in stationarity_results]
        is_stationary = [result['is_stationary'] for result in stationarity_results]
        
        # ADF Statistics
        colors = ['green' if stationary else 'red' for stationary in is_stationary]
        bars1 = axes[0].bar(tickers, adf_stats, color=colors, alpha=0.7)
        axes[0].set_title('ADF Test Statistics')
        axes[0].set_ylabel('ADF Statistic')
        axes[0].grid(True, alpha=0.3)
        
        # Add critical value line
        critical_value = stationarity_results[0]['critical_values']['5%']
        axes[0].axhline(y=critical_value, color='red', linestyle='--', 
                       label=f'5% Critical Value ({critical_value:.3f})')
        axes[0].legend()
        
        # P-values
        bars2 = axes[1].bar(tickers, p_values, color=colors, alpha=0.7)
        axes[1].set_title('P-values')
        axes[1].set_ylabel('P-value')
        axes[1].axhline(y=0.05, color='red', linestyle='--', label='Significance Level (0.05)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Add value labels on bars
        for bar, value in zip(bars1, adf_stats):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        for bar, value in zip(bars2, p_values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(self.results_dir, 'stationarity_test.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_outlier_analysis(self, outlier_results, save_plot=True):
        """
        Plot outlier analysis results
        """
        if not outlier_results:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Outlier Analysis', fontsize=16, fontweight='bold')
        
        # Extract data
        tickers = [result['ticker'] for result in outlier_results]
        outlier_counts = [result['outlier_count'] for result in outlier_results]
        outlier_percentages = [result['outlier_percentage'] for result in outlier_results]
        
        # Outlier counts
        bars1 = axes[0].bar(tickers, outlier_counts, color='skyblue', alpha=0.7)
        axes[0].set_title('Number of Outliers')
        axes[0].set_ylabel('Count')
        axes[0].grid(True, alpha=0.3)
        
        # Outlier percentages
        bars2 = axes[1].bar(tickers, outlier_percentages, color='lightcoral', alpha=0.7)
        axes[1].set_title('Outlier Percentage')
        axes[1].set_ylabel('Percentage (%)')
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, outlier_counts):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value}', ha='center', va='bottom')
        
        for bar, value in zip(bars2, outlier_percentages):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value:.2f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(self.results_dir, 'outlier_analysis.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_risk_metrics_comparison(self, data_dict, save_plot=True):
        """
        Plot risk metrics comparison across assets
        """
        # Calculate risk metrics
        risk_metrics = {}
        
        for ticker, data in data_dict.items():
            return_col = f'{ticker}_Daily_Return'
            if return_col not in data.columns:
                continue
                
            returns = data[return_col].dropna()
            
            risk_metrics[ticker] = {
                'Mean Return': returns.mean() * 252,  # Annualized
                'Volatility': returns.std() * np.sqrt(252),  # Annualized
                'Sharpe Ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
                'Max Drawdown': (returns.cumsum() - returns.cumsum().expanding().max()).min(),
                'VaR_95': returns.quantile(0.05),
                'CVaR_95': returns[returns <= returns.quantile(0.05)].mean()
            }
        
        if not risk_metrics:
            return
        
        # Create comparison plots
        metrics = ['Mean Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Risk Metrics Comparison', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [risk_metrics[ticker][metric] for ticker in risk_metrics.keys()]
            tickers = list(risk_metrics.keys())
            
            bars = axes[i].bar(tickers, values, alpha=0.7)
            axes[i].set_title(metric)
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 
                           (max(values) - min(values)) * 0.01,
                           f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(os.path.join(self.results_dir, 'risk_metrics_comparison.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
        return risk_metrics
    
    def create_interactive_dashboard(self, data_dict, save_plot=True):
        """
        Create an interactive dashboard using Plotly
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Price Comparison', 'Daily Returns', 
                          'Volatility', 'Correlation Heatmap',
                          'Cumulative Returns', 'Risk-Return Scatter'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # Price comparison
        for i, (ticker, data) in enumerate(data_dict.items()):
            fig.add_trace(
                go.Scatter(x=data.index, y=data['Close'], name=f'{ticker} Price',
                          line=dict(color=colors[i])),
                row=1, col=1
            )
        
        # Daily returns
        for i, (ticker, data) in enumerate(data_dict.items()):
            return_col = f'{ticker}_Daily_Return'
            if return_col in data.columns:
                fig.add_trace(
                    go.Scatter(x=data.index, y=data[return_col], name=f'{ticker} Returns',
                              line=dict(color=colors[i])),
                    row=1, col=2
                )
        
        # Volatility
        for i, (ticker, data) in enumerate(data_dict.items()):
            vol_col = f'{ticker}_Volatility_20d'
            if vol_col in data.columns:
                fig.add_trace(
                    go.Scatter(x=data.index, y=data[vol_col], name=f'{ticker} Volatility',
                              line=dict(color=colors[i])),
                    row=2, col=1
                )
        
        # Correlation heatmap
        returns_data = {}
        for ticker, data in data_dict.items():
            return_col = f'{ticker}_Daily_Return'
            if return_col in data.columns:
                returns_data[ticker] = data[return_col]
        
        if len(returns_data) >= 2:
            returns_df = pd.DataFrame(returns_data).dropna()
            corr_matrix = returns_df.corr()
            
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                          colorscale='RdBu', zmid=0),
                row=2, col=2
            )
        
        # Cumulative returns
        for i, (ticker, data) in enumerate(data_dict.items()):
            return_col = f'{ticker}_Daily_Return'
            if return_col in data.columns:
                cumulative_returns = (1 + data[return_col]).cumprod()
                fig.add_trace(
                    go.Scatter(x=data.index, y=cumulative_returns, name=f'{ticker} Cumulative',
                              line=dict(color=colors[i])),
                    row=3, col=1
                )
        
        # Risk-return scatter
        risk_return_data = []
        for ticker, data in data_dict.items():
            return_col = f'{ticker}_Daily_Return'
            if return_col in data.columns:
                returns = data[return_col].dropna()
                annual_return = returns.mean() * 252
                annual_vol = returns.std() * np.sqrt(252)
                risk_return_data.append([annual_vol, annual_return, ticker])
        
        if risk_return_data:
            risk_return_df = pd.DataFrame(risk_return_data, columns=['Risk', 'Return', 'Ticker'])
            fig.add_trace(
                go.Scatter(x=risk_return_df['Risk'], y=risk_return_df['Return'],
                          mode='markers+text', text=risk_return_df['Ticker'],
                          textposition="top center", name='Risk-Return'),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(height=1200, showlegend=True, title_text="Portfolio Analysis Dashboard")
        
        if save_plot:
            fig.write_html(os.path.join(self.results_dir, 'interactive_dashboard.html'))
        
        fig.show()
    
    def generate_all_plots(self, data_dict, stationarity_results=None, 
                          seasonality_results=None, outlier_results=None):
        """
        Generate all visualization plots
        """
        print("Generating visualizations...")
        
        # Generate all plots
        self.plot_price_series(data_dict)
        self.plot_returns_analysis(data_dict)
        self.plot_volatility_analysis(data_dict)
        self.plot_correlation_matrix(data_dict)
        
        if stationarity_results:
            self.plot_stationarity_test(stationarity_results)
        
        if outlier_results:
            self.plot_outlier_analysis(outlier_results)
        
        risk_metrics = self.plot_risk_metrics_comparison(data_dict)
        self.create_interactive_dashboard(data_dict)
        
        print(f"All visualizations saved to {self.results_dir}/")
        return risk_metrics

def main():
    """
    Main function for testing visualizations
    """
    print("Financial Visualizer - Run this module after data preprocessing")

if __name__ == "__main__":
    main()