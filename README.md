# Portfolio Forecast Optimizer

## Project Overview
This project implements Time Series Forecasting for Portfolio Management Optimization for GMF Investments. The goal is to predict market trends, optimize asset allocation, and enhance portfolio performance using advanced time series forecasting models.

## Business Objective
Guide Me in Finance (GMF) Investments leverages cutting-edge technology and data-driven insights to provide clients with tailored investment strategies. By integrating advanced time series forecasting models, GMF aims to predict market trends, optimize asset allocation, and enhance portfolio performance.

## Assets Analyzed
- **TSLA**: High-growth, high-risk stock in the consumer discretionary sector (Automobile Manufacturing)
- **BND**: Vanguard Total Bond Market ETF, providing stability and income
- **SPY**: S&P 500 ETF, offering broad U.S. market exposure

## Project Structure
```
portfolio-forecast-optimizer/
├── data/                   # Data storage
├── notebooks/              # Jupyter notebooks for analysis
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── models/            # Forecasting models
│   ├── portfolio/         # Portfolio optimization
│   └── utils/             # Utility functions
├── results/               # Output files and visualizations
├── requirements.txt       # Dependencies
└── README.md             # Project documentation
```

## Key Features
1. **Data Preprocessing**: Fetch and clean financial data using YFinance
2. **Exploratory Data Analysis**: Comprehensive analysis of trends, volatility, and patterns
3. **Time Series Forecasting**: ARIMA/SARIMA and LSTM models
4. **Portfolio Optimization**: Modern Portfolio Theory implementation
5. **Backtesting**: Historical performance simulation
6. **Risk Analysis**: VaR, Sharpe Ratio, and other risk metrics

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Run data collection: `python src/data/collect_data.py`
2. Execute analysis: `python src/main.py`
3. View results in the `results/` directory

