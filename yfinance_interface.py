import yfinance as yf
import pandas as pd
from datetime import datetime


def collect_data():
    today = datetime.today().strftime('%Y-%m-%d')
    start_date = '2016-01-01'
    eth_df = yf.download('ETH-USD', start_date, today)
    eth_df.reset_index(inplace=True)

    # taking opening price as the column of interest
    df = eth_df[["Date", "Open"]]
    new_names = {
        "Open": "Price",
    }

    df.rename(columns=new_names, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])

    #df = pd.read_csv("to_be_deleted.csv")
    return df
