import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Example time series data (dummy data)
dates = pd.date_range('2023-01-01', periods=100, freq='D')
data = pd.Series(np.random.randn(100), index=dates)

# Fit ARIMA model
model = ARIMA(data, order=(1, 1, 1))  # Example order (p, d, q)
fitted_model = model.fit()

# Forecast
forecast = fitted_model.forecast(steps=10)  # Forecasting 10 steps ahead
print(forecast)
