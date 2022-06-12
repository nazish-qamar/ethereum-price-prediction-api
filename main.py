import streamlit as st
import matplotlib.pyplot as plt

from yfinance_interface import *
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    st.title("Ethereum Price Forecast")

    choice = st.sidebar.selectbox("Select Model", ("ARIMA", "XGBoost"))
    if "ARIMA" == choice:
        pass
    else:
        pass