# Task 2: Time Series Forecasting Models - Documentation

## Overview
This document provides comprehensive documentation for Task 2 of the Portfolio Forecast Optimizer project. Task 2 implements and compares two different time series forecasting models for Tesla stock price prediction: ARIMA (AutoRegressive Integrated Moving Average) and LSTM (Long Short-Term Memory).

## Objectives
- Implement and compare at least two different types of forecasting models
- Use chronological data splitting (train on 2015-2023, test on 2024-2025)
- Optimize model parameters using grid search and auto_arima
- Evaluate models using MAE, RMSE, and MAPE metrics
- Provide analysis of which model performs better and why

## Implementation

### 1. ARIMA/SARIMA Model (`src/models/arima_model.py`)

#### Features:
- **Automatic Parameter Optimization**: Grid search for optimal (p, d, q) parameters
- **Stationarity Testing**: Augmented Dickey-Fuller test with automatic differencing
- **Model Diagnostics**: Residual analysis, ACF/PACF plots
- **Comprehensive Evaluation**: MAE, MSE, RMSE, MAPE metrics

#### Key Methods:
- `find_optimal_parameters()`: Grid search for best ARIMA parameters
- `make_stationary()`: Automatic differencing to achieve stationarity
- `fit_model()`: Fit ARIMA model with specified parameters
- `forecast()`: Generate predictions for test period
- `evaluate_model()`: Calculate performance metrics
- `plot_results()`: Visualize actual vs predicted values
- `plot_residuals()`: Diagnostic plots for model validation

#### Usage:
```python
from src.models.arima_model import ARIMAModel

# Initialize model
arima_model = ARIMAModel(data, target_column='Close')

# Run complete analysis
results = arima_model.run_complete_analysis(test_size=0.2)
```

### 2. LSTM Model (`src/models/lstm_model.py`)

#### Features:
- **Deep Learning Architecture**: Multi-layer LSTM with dropout
- **Hyperparameter Optimization**: Grid search for optimal architecture
- **Data Preprocessing**: MinMaxScaler for normalization
- **Sequence Generation**: Configurable lookback period
- **Early Stopping**: Prevents overfitting during training

#### Key Methods:
- `prepare_data()`: Split data and create sequences for LSTM
- `find_optimal_hyperparameters()`: Grid search for best hyperparameters
- `build_model()`: Construct LSTM architecture
- `fit_model()`: Train LSTM model with early stopping
- `forecast()`: Generate predictions using trained model
- `evaluate_model()`: Calculate performance metrics
- `plot_results()`: Visualize actual vs predicted values
- `plot_training_history()`: Training loss curves

#### Usage:
```python
from src.models.lstm_model import LSTMModel

# Initialize model
lstm_model = LSTMModel(data, target_column='Close', sequence_length=60)

# Run complete analysis
results = lstm_model.run_complete_analysis(test_size=0.2, optimize_hyperparams=True)
```

### 3. Model Comparison (`src/models/model_comparison.py`)

#### Features:
- **Comprehensive Comparison**: Side-by-side evaluation of both models
- **Performance Metrics**: Detailed comparison of MAE, MSE, RMSE, MAPE
- **Visualization**: Combined plots showing both models' predictions
- **Statistical Analysis**: Improvement percentages and overall winner determination
- **Report Generation**: Comprehensive analysis report

#### Key Methods:
- `run_arima_analysis()`: Execute complete ARIMA analysis
- `run_lstm_analysis()`: Execute complete LSTM analysis
- `compare_models()`: Compare performance metrics
- `plot_comparison()`: Visualize both models' predictions
- `plot_metrics_comparison()`: Bar chart comparison of metrics
- `generate_report()`: Create comprehensive analysis report

#### Usage:
```python
from src.models.model_comparison import ModelComparison

# Initialize comparison
comparison = ModelComparison(data, target_column='Close')

# Run complete comparison
results = comparison.run_complete_comparison(test_size=0.2, optimize_lstm=True)
```

## Data Requirements

### Input Data:
- **Format**: CSV file with datetime index
- **Required Column**: 'Close' (stock closing prices)
- **Date Range**: 2015-2025 (as prepared in Task 1)
- **Source**: `data/processed/TSLA_processed.csv`

### Data Splitting:
- **Training Set**: 80% of data (2015-2023)
- **Test Set**: 20% of data (2024-2025)
- **Method**: Chronological splitting (no random shuffling)

## Model Parameters

### ARIMA Model:
- **Parameter Search**: p ∈ [0, 5], d ∈ [0, 2], q ∈ [0, 5]
- **Optimization**: Grid search with AIC minimization
- **Stationarity**: Automatic differencing if needed

### LSTM Model:
- **Architecture**: Configurable layers (1-3) and units (30-100)
- **Hyperparameters**: Dropout (0.1-0.3), Learning rate (0.001-0.01)
- **Sequence Length**: 60 days (configurable)
- **Optimization**: Grid search with validation loss minimization

## Evaluation Metrics

### Primary Metrics:
1. **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
2. **MSE (Mean Squared Error)**: Average squared difference between predicted and actual values
3. **RMSE (Root Mean Squared Error)**: Square root of MSE, in same units as target variable
4. **MAPE (Mean Absolute Percentage Error)**: Average percentage error

### Model Selection Criteria:
- Lower values indicate better performance for all metrics
- Overall winner determined by majority of metrics
- Consideration of model complexity and interpretability

## Usage Instructions

### 1. Quick Start:
```bash
# Run complete Task 2 analysis
python src/main_task2.py
```

### 2. Test Implementation:
```bash
# Test all components
python test_task2.py
```

### 3. Individual Model Testing:
```bash
# Run individual models
python src/main_task2.py --individual
```

### 4. Custom Analysis:
```python
from src.models.model_comparison import ModelComparison
import pandas as pd

# Load data
data = pd.read_csv('data/processed/TSLA_processed.csv', index_col=0, parse_dates=True)

# Run comparison
comparison = ModelComparison(data, target_column='Close')
results = comparison.run_complete_comparison(test_size=0.2)
```

## Output Files

### Generated Files:
1. **`results/model_comparison_report.txt`**: Comprehensive analysis report
2. **Visualizations**: Interactive plots showing model comparisons
3. **Console Output**: Detailed progress and results

### Report Contents:
- Performance comparison table
- Improvement analysis
- Model characteristics
- Recommendations
- Overall winner determination

## Expected Outcomes

### Model Performance:
- **ARIMA**: Expected to perform well for linear trends and seasonality
- **LSTM**: Expected to capture complex non-linear patterns
- **Comparison**: Side-by-side evaluation of strengths and weaknesses

### Key Insights:
- Which model performs better for Tesla stock prediction
- Trade-offs between model complexity and performance
- Recommendations for practical implementation
- Understanding of model limitations and assumptions

## Technical Requirements

### Dependencies:
- **Python**: 3.8+
- **Key Libraries**: 
  - pandas, numpy, matplotlib, seaborn
  - statsmodels (for ARIMA)
  - tensorflow, keras (for LSTM)
  - scikit-learn (for metrics)

### Hardware Requirements:
- **ARIMA**: Minimal computational requirements
- **LSTM**: GPU recommended for faster training
- **Memory**: 8GB+ RAM recommended for large datasets

## Troubleshooting

### Common Issues:
1. **Memory Errors**: Reduce sequence length or batch size for LSTM
2. **Import Errors**: Ensure all dependencies are installed
3. **Data Issues**: Verify data format and column names
4. **Training Time**: LSTM optimization can take several hours

### Performance Optimization:
- Use smaller parameter grids for faster testing
- Disable hyperparameter optimization for quick runs
- Use GPU acceleration for LSTM training
- Reduce sequence length for faster LSTM training

## Future Enhancements

### Potential Improvements:
1. **Ensemble Methods**: Combine ARIMA and LSTM predictions
2. **Additional Models**: Prophet, XGBoost, or other time series models
3. **Feature Engineering**: Include technical indicators
4. **Cross-Validation**: Time series cross-validation
5. **Real-time Updates**: Online learning capabilities

## Conclusion

Task 2 provides a comprehensive framework for time series forecasting with both classical statistical (ARIMA) and modern deep learning (LSTM) approaches. The implementation includes parameter optimization, thorough evaluation, and detailed comparison to determine the most suitable model for Tesla stock price prediction.

The modular design allows for easy extension and modification, while the comprehensive evaluation ensures robust model selection based on multiple performance metrics.
