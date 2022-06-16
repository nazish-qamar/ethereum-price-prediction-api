import statsmodels.api as sm
import numpy as np
import streamlit as st

import matplotlib.pyplot as plt
import datetime as DT
import warnings
import datetime

warnings.filterwarnings('ignore')

def arima_prediction(df):
    # fit the model with above returned parameters for best model
    model = sm.tsa.arima.ARIMA(np.log(df['Price']), order=(2, 1, 2)) #order found while exploratory analysis
    fitted = model.fit()
    # Create forecast for next 7 days
    fc = fitted.get_forecast(7)

    # Set confidence to 95 %
    fc = (fc.summary_frame(alpha=0.05))

    # Get mean forecast
    fc_mean = fc['mean']

    # Get lower confidence forecast
    fc_lower = fc['mean_ci_lower']

    # Get upper confidence forecast
    fc_upper = fc['mean_ci_upper']


    today = DT.date.today()
    NextDays = [today]
    for i in range(1,7):
        NextDays.append(today + datetime.timedelta(days=1))

    future_7_days = NextDays

    # Plot mean forecast
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    ax.plot(df['Date'][-30:], df['Price'][-30:], label='Ethereum Price Forecast')

    ax.plot(future_7_days, np.exp(fc_mean), label='mean_forecast', linewidth=1.5)

    # Create confidence interval
    ax.fill_between(future_7_days, np.exp(fc_lower), np.exp(fc_upper), color='b'
                     , alpha=.1, label='95 % Confidence ')

    st.write('\nEthereum 7 Day Forecast')

    # Set legend
    fig.legend(loc='upper left', fontsize=12)
    st.pyplot(fig)
