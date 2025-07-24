
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # Load data
    try:
        data = pd.read_csv('sales_data.csv', parse_dates=['Date'])
    except FileNotFoundError:
        print("Error: 'sales_data.csv' file not found in the current directory.")
        return

    # Sort by date
    data = data.sort_values('Date')

    # Data preprocessing
    data['Month'] = data['Date'].dt.to_period('M')
    monthly_sales = data.groupby('Month')['Sales'].sum().reset_index()
    monthly_sales['Month'] = monthly_sales['Month'].dt.to_timestamp()

    # Exploratory Data Analysis (EDA)
    plt.figure(figsize=(12,6))
    sns.lineplot(data=monthly_sales, x='Month', y='Sales')
    plt.title('Monthly Sales Over Time')
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.tight_layout()
    plt.show()

    # Linear Regression Model
    monthly_sales['Month_ordinal'] = monthly_sales['Month'].map(pd.Timestamp.toordinal)
    X = monthly_sales[['Month_ordinal']]
    y = monthly_sales['Sales']

    model_lr = LinearRegression()
    model_lr.fit(X, y)
    monthly_sales['Sales_Pred_LR'] = model_lr.predict(X)

    # Plot actual vs predicted (Linear Regression)
    plt.figure(figsize=(12,6))
    plt.plot(monthly_sales['Month'], y, label='Actual Sales')
    plt.plot(monthly_sales['Month'], monthly_sales['Sales_Pred_LR'], label='Predicted Sales (Linear Regression)')
    plt.legend()
    plt.title('Linear Regression Sales Prediction')
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.tight_layout()
    plt.show()

    r2 = r2_score(y, monthly_sales['Sales_Pred_LR'])
    print(f"Linear Regression R² Score: {r2:.4f}")

    # ARIMA Model for Time Series Forecasting
    monthly_sales.set_index('Month', inplace=True)

    try:
        model_arima = ARIMA(monthly_sales['Sales'], order=(2,1,2))
        model_arima_fit = model_arima.fit()
    except Exception as e:
        print(f"ARIMA model error: {e}")
        return

    forecast_steps = 6
    forecast = model_arima_fit.forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=monthly_sales.index[-1] + pd.offsets.MonthBegin(), periods=forecast_steps, freq='MS')
    forecast = pd.Series(forecast, index=forecast_index)

    print("\nNext 6 months sales forecast (ARIMA):")
    print(forecast)

    # Plot historical sales and forecast
    plt.figure(figsize=(12,6))
    plt.plot(monthly_sales.index, monthly_sales['Sales'], label='Historical Sales')
    plt.plot(forecast.index, forecast, label='Forecasted Sales (ARIMA)', marker='o')
    plt.legend()
    plt.title('Sales Forecast using ARIMA Model')
    plt.xlabel('Month')
    plt.ylabel('Sales')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
