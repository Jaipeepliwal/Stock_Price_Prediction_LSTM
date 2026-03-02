# Stock Price Predictor Using LSTM

This project predicts stock prices using **Long Short-Term Memory (LSTM)** neural networks with rigorous backtesting, baseline comparisons, technical feature engineering, and risk analysis. It is built as an interactive **Streamlit** web application powered by historical data from Yahoo Finance.

## Motivation

With the increasing volatility of stock markets, predicting future stock prices has become a valuable tool for both individual and institutional investors. This project is designed to build a reliable, easy-to-use web application that predicts stock prices using advanced deep learning algorithms. By making stock market prediction tools more accessible, the goal is to help users make informed decisions in their investments.

## Summary

The application downloads up to 10 years of daily stock data for any Yahoo Finance ticker, computes a rich set of technical indicators, runs an LSTM model for prediction with a configurable train/test split, and benchmarks the results against naive, SMA, and EMA baselines. It also provides next-day and 15-day future price forecasts, along with comprehensive risk and return statistics.

## Features

- **Data Fetching**: Downloads up to 10 years of daily stock data via the `yfinance` API.
- **Technical Feature Engineering**: Computes SMA (20/50), EMA (12/26), MACD & Signal line, RSI (14), Bollinger Bands (Upper/Lower/Width), Volume SMA & Ratio, Daily Returns, and 20-day Rolling Volatility.
- **LSTM Prediction**: Loads a pre-trained Keras LSTM model (`keras_stocks_model.keras`) with a PyTorch backend to predict stock prices on the test set.
- **Next-Day Forecast**: Predicts the next trading day's closing price.
- **15-Day Multi-Step Forecast**: Iteratively generates price predictions for the next 15 trading days with a summary table.
- **Baseline Model Comparisons**: Benchmarks LSTM against Naive (t-1), Simple Moving Average (20), and Exponential Moving Average (20) baselines.
- **Evaluation Metrics**: Reports RMSE, MAE, MAPE, and Directional Accuracy for all models, with best-in-class highlighting.
- **Risk Metrics**: Annualized Return, Annualized Volatility, Sharpe Ratio, Sortino Ratio, Max Drawdown, VaR (95%), and CVaR (95%) displayed in the sidebar.
- **Descriptive Statistics**: Mean/Std of daily returns, Skewness, Kurtosis, Min/Max daily returns, and observation count.
- **Prediction Error Analysis**: Histogram of LSTM prediction errors with mean, std, skewness, and kurtosis.
- **Interactive Visualizations**: All charts are interactive (zoom, hover, pan) using `Plotly`, including price charts with Bollinger Bands, RSI/MACD sub-plots, train/test split visualization, model comparison, and forecast charts.
- **Configurable Sidebar**: Adjust the stock ticker and train/test split ratio; view return statistics and risk metrics at a glance.

## File Descriptions

| File | Description |
|---|---|
| **stock_prediction.py** | Main Streamlit application – data fetching, feature engineering, LSTM prediction, baseline comparison, risk analysis, and interactive dashboards. |
| **keras_stocks_model.keras** | Pre-trained LSTM model used for inference. |
| **stockprediction_analysis.ipynb** | Jupyter notebook for exploratory analysis and model development. |
| **requirements.txt** | Python dependencies (`pandas`, `numpy`, `streamlit`, `yfinance`, `plotly`, `keras`, `scikit-learn`, `torch`, `scipy`). |
| **README.md** | This file. |

## Setup Instructions

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/stock-price-prediction.git
    ```

2. Navigate to the project directory:

    ```bash
    cd stock-price-prediction
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:

    ```bash
    streamlit run stock_prediction.py
    ```

5. Open your browser and navigate to `http://localhost:8501` to use the application.

6. Alternatively, you can access the deployed web application at the following link:

    [Stock Price Prediction Web App](https://stockpricepredictionlstm.streamlit.app/)

## Technologies Used

| Library | Purpose |
|---|---|
| **pandas** | Data manipulation and analysis |
| **numpy** | Numerical computations |
| **yfinance** | Fetching historical stock data from Yahoo Finance |
| **Plotly** | Interactive charts (price, RSI/MACD, model comparison, forecasts) |
| **Keras** (PyTorch backend) | Loading and running the pre-trained LSTM model |
| **scikit-learn** | `MinMaxScaler` for feature scaling; RMSE/MAE metrics |
| **scipy** | Statistical computations |
| **Streamlit** | Web-based interactive dashboard |

## Future Enhancements

- Expand support for multiple stock indices and portfolio-level analysis.
- Add more sophisticated models, including Transformer-based architectures.
- Incorporate external financial indicators (e.g., sentiment analysis, macroeconomic data) for improved prediction accuracy.
- Enable model retraining from the UI with user-specified hyperparameters.
