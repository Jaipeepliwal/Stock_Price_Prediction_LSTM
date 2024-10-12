import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import pandas as pd

st.write("""
# Stock Price Predictor Using LSTM:
Stock values is very valuable but extremely hard to predict correctly for any human being on their own. This project seeks to solve the problem of Stock Prices Prediction by utilizes Deep Learning models, Long-Short Term Memory (LSTM) Neural Network algorithm, to predict future stock values.\n

# You can find the ticker symbol of any public company by visiting [Yahoo Finance](https://finance.yahoo.com/)
""")

# Load the pre-trained model
model = load_model('keras_stocks_model.keras')

# Function to make predictions
def predict_stock_prices(data, model):
    # Extract the 'Close' and 'Open' prices from the data
    close_prices = data['Close'].values
    open_prices = data['Open'].values

    # Normalize the prices
    normalizer = MinMaxScaler(feature_range=(0,1))
    close_prices_scaled = normalizer.fit_transform(close_prices.reshape(-1, 1))
    open_prices_scaled = normalizer.transform(open_prices.reshape(-1, 1))

    # Prepare input data for prediction
    time_stamp = 100
    X_predict_close = []
    X_predict_open = []

    for i in range(len(close_prices_scaled) - time_stamp):
        X_predict_close.append(close_prices_scaled[i:i+time_stamp, 0])
        X_predict_open.append(open_prices_scaled[i+time_stamp, 0])

    X_predict_close = np.array(X_predict_close)
    X_predict_open = np.array(X_predict_open)
    X_predict_close = np.reshape(X_predict_close, (X_predict_close.shape[0], X_predict_close.shape[1], 1))

    # Make predictions
    predicted_close_prices_scaled = model.predict(X_predict_close)

    # Inverse transform to get actual prices
    predicted_close_prices = normalizer.inverse_transform(predicted_close_prices_scaled)

    # Prepare input data for predicting next month's open price
    last_close_prices_scaled = close_prices_scaled[-time_stamp:].reshape(1, -1, 1)

    # Make prediction for next month's open price
    predicted_open_price_scaled = model.predict(last_close_prices_scaled)
    predicted_open_price = normalizer.inverse_transform(predicted_open_price_scaled)

    return predicted_close_prices, predicted_open_price

# Streamlit App
def main():
    #st.title('Taken from YFINANCE')

    # Sidebar for user input
    st.sidebar.header('Settings')
    stock_symbol = st.sidebar.text_input('Enter Stock Symbol (e.g., RELIANCE.NS):', 'RELIANCE.NS')

    # Fetch data from Yahoo Finance
    data = yf.download(tickers=stock_symbol, period='5y', interval='1d')

    # Display fetched data
    st.subheader('Stock Data')
    st.write(data)

    # Plot stock price
    st.subheader('Stock Price Chart')
    st.line_chart(data[['Close', 'Open']])

    # Predict stock prices
    if st.button('Predict Next 30 Days'):
        # Make predictions
        predicted_close_prices, predicted_open_price = predict_stock_prices(data, model)

        # Display predicted prices
        st.subheader('Predicted Stock Prices for Next 30 Days')
        st.write('Predicted Close Prices:')
        st.write(predicted_close_prices[-30:])  # Display only the last 30 days of predicted close prices
        st.write('Predicted Next Month Open Price:')
        st.write(predicted_open_price)

        # Plot predicted prices
        fig, ax = plt.subplots()
        ax.plot(predicted_close_prices[-30:], label='Predicted Close Prices')
        ax.axhline(y=predicted_open_price, color='red', linestyle='--', label='Predicted Next Month Open Price: {0}'.format(round(float(*predicted_open_price[len(predicted_open_price)-1]),2)))
        ax.set_xlabel('Days')
        ax.set_ylabel('Price')
        ax.set_title('Predicted Stock Prices')
        ax.legend()
        st.pyplot(fig)

if __name__ == '__main__':
    main()
