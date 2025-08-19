import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

class LSTMModel:
    """
    LSTM model for time series forecasting
    """
    
    def __init__(self, data, target_column='Close', sequence_length=60):
        """
        Initialize LSTM model
        
        Parameters:
        data (pd.DataFrame): Input data with datetime index
        target_column (str): Column name for target variable
        sequence_length (int): Number of time steps to look back
        """
        self.data = data
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.predictions = None
        self.test_data = None
        self.train_data = None
        
    def prepare_data(self, test_size=0.2):
        """
        Prepare data for LSTM model
        
        Parameters:
        test_size (float): Proportion of data for testing
        """
        # Split data chronologically
        total_size = len(self.data)
        train_size = int(total_size * (1 - test_size))
        
        self.train_data = self.data.iloc[:train_size]
        self.test_data = self.data.iloc[train_size:]
        
        print(f"Training set: {len(self.train_data)} samples ({self.train_data.index[0]} to {self.train_data.index[-1]})")
        print(f"Test set: {len(self.test_data)} samples ({self.test_data.index[0]} to {self.test_data.index[-1]})")
        
        # Scale the data
        train_scaled = self.scaler.fit_transform(self.train_data[[self.target_column]])
        test_scaled = self.scaler.transform(self.test_data[[self.target_column]])
        
        # Create sequences for training
        self.X_train, self.y_train = self.create_sequences(train_scaled)
        
        # Create sequences for testing
        self.X_test, self.y_test = self.create_sequences(test_scaled)
        
        print(f"Training sequences: {self.X_train.shape}")
        print(f"Testing sequences: {self.X_test.shape}")
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def set_pre_split_data(self, train_data, test_data):
        """
        Set pre-split training and test data
        
        Parameters:
        train_data (pd.DataFrame): Training data
        test_data (pd.DataFrame): Test data
        """
        self.train_data = train_data
        self.test_data = test_data
        
        print(f"Training set: {len(self.train_data)} samples ({self.train_data.index[0]} to {self.train_data.index[-1]})")
        print(f"Test set: {len(self.test_data)} samples ({self.test_data.index[0]} to {self.test_data.index[-1]})")
        
        # Scale the data
        train_scaled = self.scaler.fit_transform(self.train_data[[self.target_column]])
        test_scaled = self.scaler.transform(self.test_data[[self.target_column]])
        
        # Create sequences for training
        self.X_train, self.y_train = self.create_sequences(train_scaled)
        
        # Create sequences for testing
        self.X_test, self.y_test = self.create_sequences(test_scaled)
        
        print(f"Training sequences: {self.X_train.shape}")
        print(f"Testing sequences: {self.X_test.shape}")
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def create_sequences(self, data):
        """
        Create sequences for LSTM model
        
        Parameters:
        data (np.array): Scaled data
        
        Returns:
        tuple: (X, y) sequences
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i, 0])
            y.append(data[i, 0])
        
        return np.array(X), np.array(y)
    
    def build_model(self, units=50, layers=2, dropout=0.2, learning_rate=0.001):
        """
        Build LSTM model architecture
        
        Parameters:
        units (int): Number of LSTM units
        layers (int): Number of LSTM layers
        dropout (float): Dropout rate
        learning_rate (float): Learning rate for optimizer
        
        Returns:
        keras.Model: Compiled LSTM model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units=units, return_sequences=True, 
                      input_shape=(self.sequence_length, 1)))
        model.add(Dropout(dropout))
        
        # Additional LSTM layers
        for i in range(layers - 1):
            model.add(LSTM(units=units, return_sequences=True))
            model.add(Dropout(dropout))
        
        # Final LSTM layer
        model.add(LSTM(units=units, return_sequences=False))
        model.add(Dropout(dropout))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=learning_rate), 
                     loss='mean_squared_error')
        
        return model
    
    def find_optimal_hyperparameters(self, param_grid=None):
        """
        Find optimal hyperparameters using grid search
        
        Parameters:
        param_grid (dict): Dictionary of hyperparameters to test
        
        Returns:
        dict: Best hyperparameters
        """
        if param_grid is None:
            param_grid = {
                'units': [30, 50, 100],
                'layers': [1, 2, 3],
                'dropout': [0.1, 0.2, 0.3],
                'learning_rate': [0.001, 0.01]
            }
        
        best_score = float('inf')
        best_params = {}
        
        print("Grid search for optimal hyperparameters...")
        total_combinations = (len(param_grid['units']) * len(param_grid['layers']) * 
                            len(param_grid['dropout']) * len(param_grid['learning_rate']))
        current_combination = 0
        
        for units in param_grid['units']:
            for layers in param_grid['layers']:
                for dropout in param_grid['dropout']:
                    for lr in param_grid['learning_rate']:
                        current_combination += 1
                        print(f"Testing combination {current_combination}/{total_combinations}: "
                              f"units={units}, layers={layers}, dropout={dropout}, lr={lr}")
                        
                        try:
                            # Build and train model
                            model = self.build_model(units, layers, dropout, lr)
                            
                            # Early stopping callback
                            early_stopping = EarlyStopping(monitor='val_loss', 
                                                          patience=10, 
                                                          restore_best_weights=True)
                            
                            # Train model
                            history = model.fit(self.X_train, self.y_train,
                                              epochs=50,
                                              batch_size=32,
                                              validation_split=0.2,
                                              callbacks=[early_stopping],
                                              verbose=0)
                            
                            # Get best validation loss
                            best_val_loss = min(history.history['val_loss'])
                            
                            if best_val_loss < best_score:
                                best_score = best_val_loss
                                best_params = {
                                    'units': units,
                                    'layers': layers,
                                    'dropout': dropout,
                                    'learning_rate': lr
                                }
                                
                        except Exception as e:
                            print(f"Error with combination: {e}")
                            continue
        
        print(f"\nBest hyperparameters: {best_params}")
        print(f"Best validation loss: {best_score:.6f}")
        
        return best_params
    
    def fit_model(self, units=50, layers=2, dropout=0.2, learning_rate=0.001, 
                  epochs=100, batch_size=32):
        """
        Fit LSTM model with specified parameters
        
        Parameters:
        units (int): Number of LSTM units
        layers (int): Number of LSTM layers
        dropout (float): Dropout rate
        learning_rate (float): Learning rate
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
        Returns:
        keras.Model: Trained LSTM model
        """
        # Build model
        self.model = self.build_model(units, layers, dropout, learning_rate)
        
        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', 
                                      patience=15, 
                                      restore_best_weights=True)
        
        # Train model
        history = self.model.fit(self.X_train, self.y_train,
                               epochs=epochs,
                               batch_size=batch_size,
                               validation_split=0.2,
                               callbacks=[early_stopping],
                               verbose=1)
        
        print(f"Model trained successfully")
        print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
        
        return self.model, history
    
    def forecast(self):
        """
        Generate forecasts using the trained LSTM model
        
        Returns:
        pd.Series: Forecasted values
        """
        if self.model is None:
            raise ValueError("Model must be trained before forecasting")
        
        # Prepare test data for forecasting
        test_scaled = self.scaler.transform(self.test_data[[self.target_column]])
        
        # Generate predictions
        predictions_scaled = []
        
        # Use the last sequence from training data as initial input
        current_sequence = test_scaled[:self.sequence_length]
        
        for i in range(len(test_scaled)):
            # Reshape sequence for prediction
            current_sequence_reshaped = current_sequence.reshape((1, self.sequence_length, 1))
            
            # Predict next value
            next_prediction = self.model.predict(current_sequence_reshaped, verbose=0)
            predictions_scaled.append(next_prediction[0, 0])
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[1:], next_prediction[0, 0])
        
        # Inverse transform predictions
        predictions_reshaped = np.array(predictions_scaled).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions_reshaped)
        
        # Create forecast series
        self.predictions = pd.Series(predictions.flatten(), 
                                   index=self.test_data.index)
        
        return self.predictions
    
    def evaluate_model(self):
        """
        Evaluate model performance using multiple metrics
        
        Returns:
        dict: Evaluation metrics
        """
        if self.predictions is None or self.test_data is None:
            raise ValueError("Model must be trained and forecasted before evaluation")
        
        actual = self.test_data[self.target_column]
        predicted = self.predictions
        
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
            raise ValueError("Model must be trained and forecasted before plotting")
        
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
        
        plt.title('LSTM Model: Actual vs Predicted Stock Prices')
        plt.xlabel('Date')
        plt.ylabel('Stock Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self, history):
        """
        Plot training history (loss curves)
        
        Parameters:
        history: Training history from model.fit()
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy (if available)
        if 'accuracy' in history.history:
            ax2.plot(history.history['accuracy'], label='Training Accuracy')
            ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax2.set_title('Model Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self, test_size=0.2, optimize_hyperparams=True):
        """
        Run complete LSTM analysis pipeline
        
        Parameters:
        test_size (float): Proportion of data for testing
        optimize_hyperparams (bool): Whether to optimize hyperparameters
        
        Returns:
        dict: Complete analysis results
        """
        print("=" * 60)
        print("LSTM MODEL ANALYSIS")
        print("=" * 60)
        
        # Step 1: Prepare data
        print("\n1. Preparing data...")
        if hasattr(self, 'train_data') and hasattr(self, 'test_data') and self.train_data is not None and self.test_data is not None:
            # Use pre-split data
            self.set_pre_split_data(self.train_data, self.test_data)
        else:
            # Split data normally
            self.prepare_data(test_size)
        
        # Step 2: Find optimal hyperparameters (optional)
        if optimize_hyperparams:
            print("\n2. Finding optimal hyperparameters...")
            best_params = self.find_optimal_hyperparameters()
        else:
            best_params = {'units': 50, 'layers': 2, 'dropout': 0.2, 'learning_rate': 0.001}
        
        # Step 3: Fit model
        print("\n3. Fitting model...")
        model, history = self.fit_model(**best_params)
        
        # Step 4: Generate forecasts
        print("\n4. Generating forecasts...")
        self.forecast()
        
        # Step 5: Evaluate model
        print("\n5. Evaluating model...")
        metrics = self.evaluate_model()
        
        # Step 6: Plot results
        print("\n6. Creating visualizations...")
        self.plot_results()
        self.plot_training_history(history)
        
        results = {
            'hyperparameters': best_params,
            'metrics': metrics,
            'predictions': self.predictions,
            'model': self.model,
            'history': history
        }
        
        print("\n" + "=" * 60)
        print("LSTM ANALYSIS COMPLETED")
        print("=" * 60)
        
        return results
