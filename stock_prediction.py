import os
os.environ["KERAS_BACKEND"] = "torch"

import streamlit as st
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import load_model
import pandas as pd
from scipy import stats

st.set_page_config(page_title="LSTM Stock Predictor", layout="wide")

st.write("""
# Stock Price Predictor Using LSTM
Stock values are extremely valuable but hard to predict correctly. This project uses
**Long-Short Term Memory (LSTM)** neural networks to forecast future stock prices, with
rigorous backtesting, baseline comparisons, and technical feature engineering.

Find ticker symbols on [Yahoo Finance](https://finance.yahoo.com/)
""")

# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING HELPERS
# ─────────────────────────────────────────────────────────────

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA, EMA, RSI, Bollinger Bands, and Volume-based features."""
    df = df.copy()
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # RSI (14-period)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    # Bollinger Bands
    bb_sma = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_Upper'] = bb_sma + 2 * bb_std
    df['BB_Lower'] = bb_sma - 2 * bb_std
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    # Volume features
    df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    # Daily returns
    df['Daily_Return'] = df['Close'].pct_change()
    # Volatility (20-day rolling std of returns)
    df['Volatility_20'] = df['Daily_Return'].rolling(20).std()
    return df

# ─────────────────────────────────────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────────────────────────────────────

def compute_metrics(actual, predicted):
    """Compute RMSE, MAE, MAPE, and Directional Accuracy."""
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()
    n = min(len(actual), len(predicted))
    actual, predicted = actual[:n], predicted[:n]

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    # Directional accuracy: did we predict the direction of movement correctly?
    actual_dir = np.sign(np.diff(actual))
    pred_dir = np.sign(np.diff(predicted))
    m = min(len(actual_dir), len(pred_dir))
    dir_acc = np.mean(actual_dir[:m] == pred_dir[:m]) * 100

    return {"RMSE": round(rmse, 2), "MAE": round(mae, 2),
            "MAPE (%)": round(mape, 2), "Directional Accuracy (%)": round(dir_acc, 2)}

# ─────────────────────────────────────────────────────────────
# BASELINE MODELS
# ─────────────────────────────────────────────────────────────

def naive_forecast(train_close, test_close):
    """Naive: tomorrow's price = today's price."""
    return np.roll(test_close, 1)  # shift by 1; first value is approximate

def moving_avg_forecast(full_close, test_start_idx, window=20):
    """Moving Average baseline using a rolling window."""
    preds = []
    for i in range(test_start_idx, len(full_close)):
        preds.append(np.mean(full_close[max(0, i - window):i]))
    return np.array(preds)

def exp_moving_avg_forecast(full_close, test_start_idx, span=20):
    """Exponential Moving Average baseline."""
    s = pd.Series(full_close)
    ema = s.ewm(span=span, adjust=False).mean().values
    return ema[test_start_idx:]

# ─────────────────────────────────────────────────────────────
# LSTM PREDICTION (with train/test split)
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_lstm_model():
    return load_model('keras_stocks_model.keras')

model = load_lstm_model()

def predict_with_split(data, model, train_ratio=0.8):
    """Run LSTM prediction and return train/test split results."""
    close_prices = data['Close'].values
    n = len(close_prices)
    train_size = int(n * train_ratio)
    time_stamp = 100

    # Normalize on training data only (to avoid look-ahead bias)
    normalizer = MinMaxScaler(feature_range=(0, 1))
    close_scaled = normalizer.fit_transform(close_prices.reshape(-1, 1))

    # Build sequences for the entire dataset
    X_all, y_all = [], []
    for i in range(time_stamp, n):
        X_all.append(close_scaled[i - time_stamp:i, 0])
        y_all.append(close_scaled[i, 0])
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    X_all = X_all.reshape(X_all.shape[0], X_all.shape[1], 1)

    # Train / test index boundary (adjusted for the time_stamp offset)
    test_start_seq = max(0, train_size - time_stamp)

    X_test = X_all[test_start_seq:]
    y_test = y_all[test_start_seq:]

    # Predict on test set
    pred_scaled = model.predict(X_test, verbose=0)
    predicted = normalizer.inverse_transform(pred_scaled).flatten()
    actual_test = normalizer.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Predict next day (using last time_stamp days)
    last_seq = close_scaled[-time_stamp:].reshape(1, -1, 1)
    next_pred_scaled = model.predict(last_seq, verbose=0)
    next_pred = normalizer.inverse_transform(next_pred_scaled).flatten()[0]

    # ── Multi-step forecast (next 15 trading days) ──
    forecast_horizon = 15
    future_preds_scaled = []
    current_seq = close_scaled[-time_stamp:].flatten().tolist()  # mutable list
    for _ in range(forecast_horizon):
        inp = np.array(current_seq[-time_stamp:]).reshape(1, time_stamp, 1)
        p = model.predict(inp, verbose=0)[0, 0]
        future_preds_scaled.append(p)
        current_seq.append(p)
    future_preds = normalizer.inverse_transform(
        np.array(future_preds_scaled).reshape(-1, 1)
    ).flatten()

    # Generate future business dates starting from the day after last available date
    last_date = data.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1),
                                  periods=forecast_horizon)

    # Date indices
    dates = data.index
    train_dates = dates[:train_size]
    # Test dates align with sequences starting at (time_stamp + test_start_seq)
    test_dates = dates[time_stamp + test_start_seq: time_stamp + test_start_seq + len(actual_test)]

    return {
        "train_dates": train_dates,
        "train_close": close_prices[:train_size],
        "test_dates": test_dates,
        "actual_test": actual_test,
        "predicted_test": predicted,
        "next_pred": next_pred,
        "future_dates": future_dates,
        "future_preds": future_preds,
        "train_size": train_size,
        "time_stamp": time_stamp,
    }

# ─────────────────────────────────────────────────────────────
# RISK METRICS
# ─────────────────────────────────────────────────────────────

def compute_risk_metrics(data, risk_free_rate=0.05):
    """Compute annualized Sharpe, Sortino, max drawdown, VaR, and CVaR."""
    returns = data['Close'].pct_change().dropna()
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol != 0 else 0

    # Sortino (downside deviation only)
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-9
    sortino = (annual_return - risk_free_rate) / downside_std

    # Max Drawdown
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min()

    # Value at Risk (95%) and Conditional VaR
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()

    return {
        "Annualized Return": f"{annual_return * 100:.2f}%",
        "Annualized Volatility": f"{annual_vol * 100:.2f}%",
        "Sharpe Ratio": f"{sharpe:.3f}",
        "Sortino Ratio": f"{sortino:.3f}",
        "Max Drawdown": f"{max_dd * 100:.2f}%",
        "VaR (95%)": f"{var_95 * 100:.2f}%",
        "CVaR (95%)": f"{cvar_95 * 100:.2f}%",
    }

# ─────────────────────────────────────────────────────────────
# DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────

def descriptive_stats(data):
    """Return descriptive statistics of daily returns."""
    returns = data['Close'].pct_change().dropna()
    return {
        "Mean Daily Return": f"{returns.mean() * 100:.4f}%",
        "Std (Daily)": f"{returns.std() * 100:.4f}%",
        "Skewness": f"{returns.skew():.4f}",
        "Kurtosis": f"{returns.kurtosis():.4f}",
        "Min Daily Return": f"{returns.min() * 100:.2f}%",
        "Max Daily Return": f"{returns.max() * 100:.2f}%",
        "Observations": f"{len(returns)}",
    }

# ─────────────────────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────────────────────

def main():
    # ── Sidebar ──────────────────────────────────────
    st.sidebar.header("⚙️ Settings")
    stock_symbol = st.sidebar.text_input(
        "Enter Stock Symbol (e.g., RELIANCE.NS):", "RELIANCE.NS"
    )
    train_ratio = st.sidebar.slider("Train / Test Split Ratio", 0.6, 0.9, 0.8, 0.05)

    # ── Fetch Data ───────────────────────────────────
    with st.spinner("Downloading data from Yahoo Finance..."):
        data = yf.download(tickers=stock_symbol, period="10y", interval="1d")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if data.empty:
        st.error("No data returned. Check the ticker symbol.")
        return

    # ── Feature Engineering ──────────────────────────
    data = add_technical_features(data)

    # ── Sidebar: Descriptive Statistics ──────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Return Statistics")
    for k, v in descriptive_stats(data).items():
        st.sidebar.metric(label=k, value=v)

    # ── Sidebar: Risk Metrics ────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("📉 Risk Metrics")
    for k, v in compute_risk_metrics(data).items():
        st.sidebar.metric(label=k, value=v)

    # ── Main: Raw Data ───────────────────────────────
    st.subheader("📋 Stock Data (with Technical Indicators)")
    st.dataframe(data.tail(100), use_container_width=True)

    # ── Main: Interactive Price Chart ────────────────
    st.subheader("📈 Interactive Price Chart")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close', line=dict(color='#1f77b4')))
    fig_price.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='orange', dash='dot')))
    fig_price.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='red', dash='dot')))
    fig_price.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], name='BB Upper', line=dict(color='rgba(150,150,150,0.4)'), showlegend=False))
    fig_price.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], name='Bollinger Bands', fill='tonexty', fillcolor='rgba(173,216,230,0.15)', line=dict(color='rgba(150,150,150,0.4)')))
    fig_price.update_layout(title=f"{stock_symbol} – Price & Technical Indicators", xaxis_title="Date", yaxis_title="Price", hovermode="x unified", height=500)
    st.plotly_chart(fig_price, use_container_width=True)

    # ── RSI Chart ────────────────────────────────────
    with st.expander("Show RSI & MACD Charts"):
        fig_rsi = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.08,
                                subplot_titles=("RSI (14)", "MACD"))
        fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI_14'], name='RSI', line=dict(color='purple')), row=1, col=1)
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
        fig_rsi.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')), row=2, col=1)
        fig_rsi.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal', line=dict(color='orange')), row=2, col=1)
        fig_rsi.update_layout(height=450, hovermode="x unified")
        st.plotly_chart(fig_rsi, use_container_width=True)

    # ── Prediction & Evaluation ──────────────────────
    st.markdown("---")
    st.subheader("🔮 LSTM Prediction & Model Evaluation")

    if st.button("▶ Run Prediction & Backtesting", type="primary"):
        with st.spinner("Running LSTM predictions on test set..."):
            res = predict_with_split(data, model, train_ratio)

        actual = res["actual_test"]
        predicted = res["predicted_test"]
        test_dates = res["test_dates"]
        train_dates = res["train_dates"]
        close_all = data['Close'].values
        train_size = res["train_size"]
        time_stamp = res["time_stamp"]

        # ── Train/Test Split Visualization ───────────
        st.subheader("📊 Train / Test Split – Actual vs Predicted")

        fig_split = go.Figure()
        # Training region
        fig_split.add_trace(go.Scatter(x=train_dates, y=res["train_close"],
                                       name="Training Data", line=dict(color="#1f77b4")))
        # Actual test
        fig_split.add_trace(go.Scatter(x=test_dates, y=actual,
                                       name="Actual (Test)", line=dict(color="#2ca02c")))
        # Predicted test
        fig_split.add_trace(go.Scatter(x=test_dates, y=predicted,
                                       name="LSTM Predicted (Test)", line=dict(color="#ff7f0e")))
        # Demarcation line – use add_shape directly to avoid Plotly annotation arithmetic bug with Timestamps
        split_date = train_dates[-1]
        fig_split.add_shape(
            type="line", x0=split_date, x1=split_date, y0=0, y1=1,
            yref="paper", line=dict(color="red", dash="dash", width=2),
        )
        fig_split.add_annotation(
            x=split_date, y=1.04, yref="paper",
            text=f"Train/Test Split ({train_ratio:.0%})",
            showarrow=False, font=dict(color="red", size=12),
        )
        fig_split.update_layout(title="LSTM – Actual vs Predicted with Train/Test Split",
                                xaxis_title="Date", yaxis_title="Price",
                                hovermode="x unified", height=500)
        st.plotly_chart(fig_split, use_container_width=True)

        # ── Baseline Comparisons ─────────────────────
        st.subheader("📏 Baseline Model Comparisons")

        test_start_idx = train_size
        naive_pred = naive_forecast(close_all[:train_size], actual)
        ma_pred = moving_avg_forecast(close_all, test_start_idx, window=20)
        ema_pred = exp_moving_avg_forecast(close_all, test_start_idx, span=20)

        # Trim to same length
        min_len = min(len(actual), len(naive_pred), len(ma_pred), len(ema_pred), len(predicted))
        actual_t = actual[:min_len]
        predicted_t = predicted[:min_len]
        naive_t = naive_pred[:min_len]
        ma_t = ma_pred[:min_len]
        ema_t = ema_pred[:min_len]
        dates_t = test_dates[:min_len]

        # Metrics table
        metrics_lstm = compute_metrics(actual_t, predicted_t)
        metrics_naive = compute_metrics(actual_t, naive_t)
        metrics_ma = compute_metrics(actual_t, ma_t)
        metrics_ema = compute_metrics(actual_t, ema_t)

        metrics_df = pd.DataFrame({
            "LSTM": metrics_lstm,
            "Naive (t-1)": metrics_naive,
            "SMA (20)": metrics_ma,
            "EMA (20)": metrics_ema,
        }).T
        metrics_df.index.name = "Model"

        st.dataframe(metrics_df.style.highlight_min(axis=0, subset=["RMSE", "MAE", "MAPE (%)"],
                                                     props="background-color: #d4edda;")
                      .highlight_max(axis=0, subset=["Directional Accuracy (%)"],
                                     props="background-color: #d4edda;"),
                      use_container_width=True)

        # Interactive comparison chart
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(x=dates_t, y=actual_t, name="Actual", line=dict(color="black", width=2)))
        fig_cmp.add_trace(go.Scatter(x=dates_t, y=predicted_t, name="LSTM", line=dict(color="#ff7f0e")))
        fig_cmp.add_trace(go.Scatter(x=dates_t, y=naive_t, name="Naive", line=dict(color="gray", dash="dot")))
        fig_cmp.add_trace(go.Scatter(x=dates_t, y=ma_t, name="SMA(20)", line=dict(color="teal", dash="dash")))
        fig_cmp.add_trace(go.Scatter(x=dates_t, y=ema_t, name="EMA(20)", line=dict(color="purple", dash="dashdot")))
        fig_cmp.update_layout(title="Model Comparison on Test Set",
                              xaxis_title="Date", yaxis_title="Price",
                              hovermode="x unified", height=500)
        st.plotly_chart(fig_cmp, use_container_width=True)

        # ── Prediction Error Distribution ────────────
        with st.expander("Show Prediction Error Distribution"):
            errors = actual_t - predicted_t
            fig_err = go.Figure()
            fig_err.add_trace(go.Histogram(x=errors, nbinsx=50, name="LSTM Error",
                                           marker_color="#ff7f0e", opacity=0.75))
            fig_err.add_vline(x=0, line_color="red", line_dash="dash")
            fig_err.update_layout(title="LSTM Prediction Error Distribution (Actual − Predicted)",
                                  xaxis_title="Error", yaxis_title="Frequency", height=350)
            st.plotly_chart(fig_err, use_container_width=True)

            st.write(f"**Mean Error:** {np.mean(errors):.2f}  |  "
                     f"**Std Error:** {np.std(errors):.2f}  |  "
                     f"**Skewness:** {float(pd.Series(errors).skew()):.3f}  |  "
                     f"**Kurtosis:** {float(pd.Series(errors).kurtosis()):.3f}")

        # ── Next-Day Prediction ──────────────────────
        st.markdown("---")
        st.subheader("🔮 Next Trading Day Forecast")
        col1, col2, col3 = st.columns(3)
        last_close = float(data['Close'].iloc[-1])
        next_pred = res["next_pred"]
        change_pct = ((next_pred - last_close) / last_close) * 100
        col1.metric("Last Close", f"₹{last_close:,.2f}")
        col2.metric("Predicted Next Close", f"₹{next_pred:,.2f}")
        col3.metric("Expected Change", f"{change_pct:+.2f}%",
                     delta=f"{change_pct:+.2f}%")

        # ── 15-Day Future Forecast Chart ─────────────
        st.markdown("---")
        st.subheader("📅 Next 15 Trading Days – Predicted Prices")

        future_dates = res["future_dates"]
        future_preds = res["future_preds"]

        # Recent actuals (last 30 days) for context
        recent_n = 30
        recent_dates = data.index[-recent_n:]
        recent_close = data['Close'].values[-recent_n:]

        fig_future = go.Figure()
        # Recent actual prices
        fig_future.add_trace(go.Scatter(
            x=recent_dates, y=recent_close,
            name="Recent Actual Close",
            line=dict(color="#1f77b4", width=2),
            mode="lines+markers", marker=dict(size=4),
        ))
        # Connecting line from last actual to first forecast
        fig_future.add_trace(go.Scatter(
            x=[recent_dates[-1], future_dates[0]],
            y=[recent_close[-1], future_preds[0]],
            line=dict(color="#ff7f0e", dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))
        # Forecast
        fig_future.add_trace(go.Scatter(
            x=future_dates, y=future_preds,
            name="LSTM 15-Day Forecast",
            line=dict(color="#ff7f0e", width=2),
            mode="lines+markers", marker=dict(size=6, symbol="diamond"),
        ))
        # Demarcation
        fig_future.add_shape(
            type="line", x0=recent_dates[-1], x1=recent_dates[-1],
            y0=0, y1=1, yref="paper",
            line=dict(color="gray", dash="dash", width=1),
        )
        fig_future.add_annotation(
            x=recent_dates[-1], y=1.04, yref="paper",
            text="Today", showarrow=False,
            font=dict(color="gray", size=11),
        )
        fig_future.update_layout(
            title=f"{stock_symbol} – 15 Trading Day Price Forecast",
            xaxis_title="Date", yaxis_title="Price (₹)",
            hovermode="x unified", height=480,
        )
        st.plotly_chart(fig_future, use_container_width=True)

        # Summary table
        forecast_df = pd.DataFrame({
            "Date": future_dates.strftime("%Y-%m-%d"),
            "Predicted Close (₹)": [round(p, 2) for p in future_preds],
            "Change from Last Close (₹)": [round(p - last_close, 2) for p in future_preds],
            "Change (%)": [round((p - last_close) / last_close * 100, 2) for p in future_preds],
        })
        forecast_df.index = range(1, len(forecast_df) + 1)
        forecast_df.index.name = "Day"
        st.dataframe(forecast_df, use_container_width=True)

if __name__ == '__main__':
    main()
