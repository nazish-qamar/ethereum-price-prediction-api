import yfinance as yf
import pandas as pd
from datetime import datetime


def collect_data():
    today = datetime.today().strftime('%Y-%m-%d') #Fetching today's date
    start_date = '2016-01-01'
    ethereum_data = yf.download('ETH-USD', start_date, today)
    ethereum_data.reset_index(inplace=True)

    # taking opening price as the column of interest
    df = ethereum_data[["Date", "Open"]]
    new_names = {
        "Open": "Price",
    }

    df.rename(columns=new_names, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])

    return df
