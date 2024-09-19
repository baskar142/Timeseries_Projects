# model_functions.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
from prophet import Prophet
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class TimeSeriesModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.scalers = {}
        self.train = None
        self.test = None
        self.columns = None

    def load_and_preprocess_data(self):
        df = pd.read_csv(self.file_path, parse_dates=['Date'], index_col='Date')
        required_columns = ['Total_Network_Traffic', 'Latency (ms)', 'Packet_Loss (%)']
        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' is missing from the dataset")
        self.df = df
        self.columns = required_columns

    def normalize_data(self):
        for column in self.columns:
            scaler = MinMaxScaler()
            self.scalers[column] = scaler.fit(self.df[[column]])
            self.df[column] = scaler.transform(self.df[[column]])
        return self.df

    def inverse_transform_single_column(self, scaled_data, column):
        scaled_data = scaled_data.reshape(-1, 1)
        unscaled_data = self.scalers[column].inverse_transform(scaled_data)
        return pd.Series(unscaled_data.flatten(), index=self.df.index)

    def fit_ses(self, train, test):
        models = {}
        forecasts = {}
        for column in self.columns:
            model = ExponentialSmoothing(train[column], trend=None, seasonal=None)
            model_fit = model.fit()
            models[column] = model_fit
            forecasts[column] = model_fit.forecast(len(test))
        return models, forecasts

    def fit_holt(self, train, test):
        models = {}
        forecasts = {}
        for column in self.columns:
            model = ExponentialSmoothing(train[column], trend='add', seasonal=None)
            model_fit = model.fit()
            models[column] = model_fit
            forecasts[column] = model_fit.forecast(len(test))
        return models, forecasts

    def fit_hw(self, train, test):
        models = {}
        forecasts = {}
        for column in self.columns:
            model = ExponentialSmoothing(train[column], trend='add', seasonal='add', seasonal_periods=365)
            model_fit = model.fit()
            models[column] = model_fit
            forecasts[column] = model_fit.forecast(len(test))
        return models, forecasts

    def fit_sarima(self, train, test):
        models = {}
        forecasts = {}
        for column in self.columns:
            model = SARIMAX(train[column], order=(1, 1, 1), seasonal_order=(1, 1, 1, 365))
            model_fit = model.fit(disp=False)
            models[column] = model_fit
            forecasts[column] = model_fit.get_forecast(steps=len(test)).predicted_mean
        return models, forecasts

    def fit_var(self, train):
        model = VAR(train)
        model_fit = model.fit(maxlags=15, ic='aic')
        return model_fit

    def forecast_var(self, model_fit, steps, test):
        forecast = model_fit.forecast(model_fit.y, steps=steps)
        return pd.DataFrame(forecast, columns=self.columns, index=test.index)

    def fit_prophet(self, train, column):
        df_prophet = train.reset_index().rename(columns={column: 'y', 'Date': 'ds'})
        model = Prophet()
        model.fit(df_prophet)
        return model

    def forecast_prophet(self, model, steps, future_dates):
        forecast = model.predict(future_dates)
        return forecast[['ds', 'yhat']]

    def evaluate_forecast(self, actual, forecast):
        actual_values = actual.values
        forecast_values = forecast.values
        rmse = mean_squared_error(actual_values, forecast_values, squared=False)
        mape = mean_absolute_percentage_error(actual_values, forecast_values)
        return rmse, mape
