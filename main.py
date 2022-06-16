from arima_utility import *
from yfinance_interface import *
import warnings
warnings.filterwarnings('ignore')
#streamlit run D:/git_clone/ethereum-price-prediction-api/main.py

if __name__ == '__main__':
    st.title("Ethereum Price Forecast")
    st.write("Ethereum Price Trend")

    choice = st.sidebar.selectbox("Select Model", ("ARIMA", "XGBoost"))
    price_df = collect_data()

    # Plotting Price values
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    ax.plot(price_df['Date'], price_df['Price'])

    st.pyplot(fig)

    if "ARIMA" == choice:
        arima_prediction(price_df)

    else:
        pass