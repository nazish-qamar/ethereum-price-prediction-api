# ethereum-price-prediction-api
A web API for an interactive dashboard for forecasting Ethereum price for the next seven days

## Files Information
### 1.main.py
#### The starting point of the project that provides an interface for different modules.

### 2.yfinance_interface.py
#### Contains the logic for fetching the ethereum prices up to the current date.

### 3.arima_utility.py
#### Contains the logic data pre processing and model selection on seasonal arima model

### 4.xgboost_utility.py
#### Contains the logic for pre processing ethereum price time series data and predicting the price for the next seven days 

### 5.data_exploration_and_model_selection.py
#### The original file used for data exploration and the best model selection for the ARIMA model and the XGBoost models. This is a standalone file that can be run without starting the streamlit server

### 6.requirements.txt
#### The file contains the list of all the required libraries


## Running Project
#### 1. Run the following command in the terminal of the IDE
####      	"streamlit run main.py"
#### 2. Click on the link shown after running the above command
#### 3. Select the required action from the drop-down menu and proceed as directed
