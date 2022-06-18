from yfinance_interface import *
from arima_utility import *
from xgboost_utility import *

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    st.title("Ethereum Price Forecast")
    st.write("Ethereum Price Trend")

    choice = st.sidebar.selectbox("Select Model", ("ARIMA", "XGBoost"))
    price_df = collect_data()

    # Plotting Ethereum Prices
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    ax.plot(price_df['Date'], price_df['Price'])

    st.pyplot(fig)

    if "ARIMA" == choice:
        arima_prediction(price_df.copy())

    else:
        xgboost_prediction(price_df.copy())