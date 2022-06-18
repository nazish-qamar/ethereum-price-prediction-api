import numpy as np
import xgboost as xgb
import datetime as dt
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt

def xgboost_prediction(df):
    df['DateStr'] = [str(i) for i in df['Date']]
    df['Year'] = [int(i[0:4]) for i in df['DateStr']]
    df['Month']  = [int(i[5:7]) for i in df['DateStr']]
    df['Day'] = [int(i[8:11]) for i in df['DateStr']]
    df['Price_Shift1'] = df['Price'].shift(periods=7)

    df = df.dropna()
    df['row_num'] = np.arange(len(df))

    day_iter = dt.date.today()
    NextDays = [day_iter]
    for i in range(1, 7):
        day_iter = day_iter + dt.timedelta(days=1)
        NextDays.append(day_iter)


    test_set = pd.DataFrame()
    test_set['Day'] = [int(i.strftime("%d")) for i in NextDays]

    test_set['Price_Shift1'] =  [float(i) for i in df['Price'][-7:]]

    train_last_row = int(df['row_num'][-1:])

    row_idx = [train_last_row]
    for i in range(1,7):
        row_idx.append(i+train_last_row)

    test_set['row_num'] = row_idx

    # fit model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=250)
    model.fit(df[['Day', 'Price_Shift1', 'row_num']],
              df['Price'])  # the best model is obtained with these three features

    pred = model.predict(test_set[['Day', 'Price_Shift1', 'row_num']])
    y_pred = pred.tolist()

    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    ax.plot(df['Date'][-30:], df['Price'][-30:], label='Ethereum Price Forecast')

    ax.plot(NextDays, y_pred , label='mean_forecast', linewidth=1.5)


    st.write('\nEthereum 7 Day Forecast by XGBoost Model')

    # Set legend
    fig.legend(loc='upper left', fontsize=12)
    st.pyplot(fig)
