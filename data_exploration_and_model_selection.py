import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import pmdarima as pm
import yfinance as yf


def fetch_prices():
    today = datetime.today().strftime('%Y-%m-%d')
    start_date = '2016-01-01'
    eth_df = yf.download('ETH-USD', start_date, today)
    eth_df.reset_index(inplace=True)

    # taking opening price as the column of interest
    df = eth_df[["Date", "Open"]]
    new_names = {
        "Open": "Value",
    }

    df.rename(columns=new_names, inplace=True)
    return df


def data_split_and_plot(df):
    train_set = df.iloc[:(len(df) - 7), :]  # df.loc[df['Date'] < '2022-05-01']
    test_set = df.iloc[(len(df) - 7):len(df), :]  # df.loc[df.Date >= '2022-05-01']

    # Plotting Price values
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    ax.plot(train_set['Date'], train_set['Value'])

    # train_set copy
    train_set_orig = train_set.copy()

    # Plotting log values of the Price
    train_set['Value'] = np.log(train_set['Value'])
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    ax.plot(train_set['Date'], train_set['Value'])

    return train_set, test_set, train_set_orig


# Augmented Dicky Fuller Test (ADF Test) for checking the series is stationary or not
def adf_test(target_series):
    result = adfuller(target_series)
    adf_statistic = result[0]
    p_value = result[1]
    print('ADF Statistic : % f' % adf_statistic)
    print('p - value : % f' % p_value)
    return adf_statistic, p_value


# The Kwiatkowski–Phillips–Schmidt–Shin (KPSS)
def kpss_test(target_series):
    print("Results of KPSS Test : ")
    kpsstest = kpss(target_series, regression="ct", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0: 3], index=[" Test Statistic ", " p - value ", " Lags Used "])
    for key, value in kpsstest[3].items():
        kpss_output[" Critical Value ( %s )" % key] = value
    print(kpss_output)


# Differentiate the series till we get p value of ADF test less than 0.05
def best_diff_factor(p_value, df):
    d = 0
    while p_value > 0.05:
        df['Value'] = df['Value'].diff()
        df.dropna(inplace=True)
        d += 1
        adf_statistic, p_value = adf_test(df['Value'])
    print("final d value = %f" % d)
    return df


def afc_plot(train_set):
    # ACF plot to check correlation coefficients between the price time series and its lag values
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    plot_acf(train_set['Value'], lags=20, ax=ax)
    plt.ylim = ([-0.05, 0.25])
    plt.yticks(np.arange(-0.10, 1.1, 0.1))
    plt.show()


def pacf_plot(train_set):
    # PACF to get the partial correlation between the series and its lags
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    plot_pacf(train_set['Value'], lags=20, ax=ax)
    plt.ylim = ([-0.05, 0.25])
    plt.yticks(np.arange(-0.10, 1.1, 0.1))
    plt.show()


# Now applying auto arima to find the p, d and q values
def auto_arima(orig_df):
    orig_df = np.log(orig_df['Value'])
    model = pm.auto_arima(orig_df,
                          start_p=10,
                          start_q=10,
                          test='adf',
                          max_p=10,
                          max_q=10,
                          m=1,
                          d=None,
                          seasonal=False,
                          D=0,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)
    # difference df by d found by auto arima
    differenced_by_auto_arima = orig_df.diff(model.order[1])
    return model.order, differenced_by_auto_arima, model.resid()


def fit_and_plot_best_model(train_set_orig, test_set):
    # fit the model with above returned parameters for best model
    model = sm.tsa.arima.ARIMA(np.log(train_set_orig['Value']), order=(2, 1, 2))
    fitted = model.fit()

    # Create forecast for 7 days
    fc = fitted.get_forecast(7)

    # Set confidence to 95 %
    fc = (fc.summary_frame(alpha=0.05))

    # Get mean forecast
    fc_mean = fc['mean']

    # Get lower confidence forecast
    fc_lower = fc['mean_ci_lower']

    # Get upper confidence forecast
    fc_upper = fc['mean_ci_upper']

    # Set figure size
    plt.figure(figsize=(12, 8), dpi=100)

    # Plot last 100 price movements
    plt.plot(train_set_orig['Date'][-50:], train_set_orig['Value'][-50:], label='Ethereum Price ')

    # create date axis for predictions
    future_7_days = test_set['Date']

    # Plot mean forecast
    plt.plot(future_7_days, np.exp(fc_mean), label='mean_forecast', linewidth=1.5)

    # Plot actual test values
    plt.plot(future_7_days, test_set['Value'], label='Actual price', linewidth=1.5)

    # Create confidence interval

    plt.fill_between(future_7_days, np.exp(fc_lower), np.exp(fc_upper), color='b'
                     , alpha=.1, label='95 % Confidence ')

    # Set title
    plt.title('Ethereum 7 Day Forecast ')

    # Set legend
    plt.legend(loc='upper left', fontsize=8)
    plt.show()


## Data exploration and ARIMA Model selection
df = fetch_prices()  # fetch the ethereum prices from yahoo finance
df['Date'] = pd.to_datetime(df['Date'])

train_set, test_set, train_set_orig = data_split_and_plot(df)

adf_statistic, p_value = adf_test(train_set['Value'])  # ADF p value is greater than 0.5, so series is non stationary
kpss_test(train_set['Value'])  # p value < 0.05, so series is non stationary

train_set = best_diff_factor(p_value, train_set)  # find the best difference
kpss_test(train_set['Value'])  # Check the KPSS  p value again

# Plot the train_set after differencing
fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
ax.plot(train_set['Date'], train_set['Value'])

# Plot the ACF and PACF plots and also check the ljungbox p values
afc_plot(train_set)
pacf_plot(train_set)

# Using Ljung–Box test to determine whether any group of autocorrelations in a time series differs from zero.
# H0: The data is distributed independently.
# H1: The data is not distributed randomly; instead, they exhibit serial correlation.
sm.stats.acorr_ljungbox(train_set['Value'], lags=[20], return_df=True)

# Apply the auto_arima find the best order of the arima model
model_order, differenced_data, model_residuals = auto_arima(train_set_orig)

# applying the Ljung–Box test again to the model’s residuals
sm.stats.acorr_ljungbox(model_residuals, lags=[20],
                        return_df=True)  # p_value 0.99 signifies residual are just white noise

fit_and_plot_best_model(train_set_orig, test_set)


##XGboost data preparation and model selection
#testing xgboost models
import xgboost as xgb
from sklearn.metrics import mean_squared_error

def process_xgboost_data(df):
    df['DateStr'] = [str(i) for i in df['Date']]
    df['Year'] = [int(i[0:4]) for i in df['DateStr']]
    df['Month'] = [int(i[5:7]) for i in df['DateStr']]
    df['Day'] = [int(i[8:11]) for i in df['DateStr']]
    df['Price_Shift1'] = df['Price'].shift(periods=7)
    df = df.dropna()
    df['row_num'] = np.arange(len(df))


def model_training(train_set):
    # fit model
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=250)
    xgb_model.fit(train_set[['Day', 'Price_Shift1', 'row_num']],
              train_set['Price'])  # the best model is obtained with these three features

    return xgb_model


df = fetch_prices()
process_xgboost_data(df)

#taking last 15 days as test set for checking model performance
train_set = df.iloc[:(len(df)-15), :]
test_set = df.iloc[(len(df)-15):len(df), :]

model = model_training(train_set)

#prediction
pred = model.predict(test_set[['Day', 'Price_Shift1', 'row_num']])
y_pred = pred.tolist()
y_true = list(test_set['Price'])

#model performance
mse = mean_squared_error(y_pred, y_true)
rmse = np.sqrt(mse)
print("RMSE: %f" % (rmse))