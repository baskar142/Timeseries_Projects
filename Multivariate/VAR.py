import pandas as pd
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class TimeSeriesForecasting:
    def __init__(self, data):
        self.data = data
        self.columns = data.columns
        self.train = None
        self.test = None

    def prepare_data(self, train_size=0.8):
        # Split the data into train and test sets
        split_idx = int(len(self.data) * train_size)
        self.train, self.test = self.data[:split_idx], self.data[split_idx:]
        print(f"Data split into train and test sets: {len(self.train)} training samples, {len(self.test)} test samples.")
        
    def fit_var(self, train):
        # Fit VAR model
        print("Fitting VAR model")
        model = VAR(train)
        model_fit = model.fit(maxlags=15, ic='aic')
        return model_fit

    def forecast_var(self, model_fit, steps):
        # Forecast using VAR model
        print("Forecasting with VAR model")
        lag_order = model_fit.k_ar
        forecast_input = self.data.values[-lag_order:]
        forecast = model_fit.forecast(y=forecast_input, steps=steps)
        forecast_index = self.test.index
        forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=self.columns)
        print(f"VAR forecasted values with length: {len(forecast_df)}")
        return forecast_df

    def evaluate_forecast(self, actual, forecast, columns):
        results = {}
        for column in columns:
            if column not in actual.columns or column not in forecast.columns:
                print(f"Column {column} is missing in actual or forecast data.")
                continue
            
            # Ensure forecast values match the length of the test set
            forecast = forecast.reindex(actual.index)
            
            actual_values = actual[column].dropna().values
            forecast_values = forecast[column].dropna().values
            
            if len(actual_values) != len(forecast_values):
                print(f"Length mismatch between actual and forecast values for {column}.")
                print(f"Actual length: {len(actual_values)}, Forecast length: {len(forecast_values)}")
                continue
            
            if np.any(np.isnan(actual_values)) or np.any(np.isnan(forecast_values)):
                print(f"NaN values present in actual or forecast data for {column}.")
                continue
            
            rmse = mean_squared_error(actual_values, forecast_values, squared=False)
            mape = mean_absolute_percentage_error(actual_values, forecast_values)
            results[column] = (rmse, mape)
        
        return results

    def plot_forecast(self, actual, forecast, columns):
        for column in columns:
            if column not in actual.columns or column not in forecast.columns:
                print(f"Column {column} is missing in actual or forecast data.")
                continue
            
            plt.figure(figsize=(12, 6))
            plt.plot(actual.index, actual[column], label='Actual', color='blue')
            plt.plot(forecast.index, forecast[column], label='Forecast', color='red', linestyle='--')
            plt.title(f'Actual vs Forecast for {column}')
            plt.xlabel('Date')
            plt.ylabel(column)
            plt.legend()
            plt.show()

    def save_forecast(self, forecast, filename):
        # Save forecast data to CSV
        forecast.to_csv(filename)
        print(f"Forecast saved to {filename}")

    def save_evaluation_results(self, results, filename):
        # Save evaluation results to CSV
        results_df = pd.DataFrame(results).T
        results_df.columns = ['RMSE', 'MAPE']
        results_df.to_csv(filename)
        print(f"Evaluation results saved to {filename}")
        
   

