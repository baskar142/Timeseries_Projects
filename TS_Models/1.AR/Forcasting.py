import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def conversion(data, y_train):
    """
    Convert a 1D array-like y_train to a DataFrame with columns matching the data.
    """
    Actual_y_train = pd.DataFrame(y_train, columns=data.columns)
    return Actual_y_train

def conversionSingle(data, y_train):
    """
    Convert a 1D array-like y_train to a DataFrame with a single column.
    """
    Actual_y_train = pd.DataFrame(y_train, columns=[data.columns[0]])
    return Actual_y_train

def graph(Actual, predicted, Actlabel, predlabel, title, Xlabel, ylabel):
    """
    Plot Actual vs Predicted values.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(Actual, color='blue', label=Actlabel)
    plt.plot(predicted, color='green', label=predlabel)
    plt.title(title)
    plt.xlabel(Xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def rmsemape(y_Test, predicted):
    """
    Calculate RMSE and MAPE for the given test and predicted values.
    """
    rmse = mean_squared_error(y_Test, predicted, squared=False)
    mape = mean_absolute_percentage_error(y_Test, predicted)
    
    print("RMSE-Testset:", rmse)
    print("MAPE-Testset:", mape)

def save_forecast(forecast_df, filename):
    """
    Save the forecast DataFrame to a CSV file.
    """
    forecast_df.to_csv(filename, index=False)
    print(f"Forecast data saved to {filename}")

def save_evaluation_results(results, filename):
    """
    Save evaluation results to a CSV file.
    """
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['RMSE', 'MAPE'])
    results_df.to_csv(filename, index=False)
    print(f"Evaluation results saved to {filename}")
