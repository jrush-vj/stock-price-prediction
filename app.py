
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


def get_data(ticker, start, end):
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker/date range.")
    df.to_csv(f"{ticker}_raw.csv")
    return df[["Close"]].dropna()


def add_features(df):
    df = df.copy()
    df["MA7"] = df["Close"].rolling(7).mean()
    df["MA21"] = df["Close"].rolling(21).mean()
    df["Return"] = df["Close"].pct_change()
    return df.dropna()


def make_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(data[i, 0])  # Close price is column 0
    return np.array(X), np.array(y)


def build_lstm(input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def evaluate(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"[{label}] RMSE={rmse:.4f}  MAE={mae:.4f}  MAPE={mape:.2f}%")
    return rmse, mae, mape


def main(ticker, start, end, window, epochs, batch_size):
    raw = get_data(ticker, start, end)
    df = add_features(raw)

    feature_cols = ["Close", "MA7", "MA21", "Return"]
    data = df[feature_cols].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = make_sequences(scaled, window)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]


    lr = LinearRegression()
    lr.fit(X_train.reshape(len(X_train), -1), y_train)
    lr_pred = lr.predict(X_test.reshape(len(X_test), -1))

    model = build_lstm((X_train.shape[1], X_train.shape[2]))
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )
    lstm_pred = model.predict(X_test).flatten()

    def inverse_close(scaled_close_vals):
        dummy = np.zeros((len(scaled_close_vals), scaled.shape[1]))
        dummy[:, 0] = scaled_close_vals
        return scaler.inverse_transform(dummy)[:, 0]

    y_test_actual = inverse_close(y_test)
    lstm_actual = inverse_close(lstm_pred)
    lr_actual = inverse_close(lr_pred)

    evaluate(y_test_actual, lr_actual, "Linear Regression")
    evaluate(y_test_actual, lstm_actual, "LSTM")

    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual, label="Actual", linewidth=2)
    plt.plot(lstm_actual, label="LSTM Prediction")
    plt.plot(lr_actual, label="Linear Regression Prediction", linestyle="--")
    plt.title(f"{ticker} Close Price Prediction")
    plt.xlabel("Time Step (test set)")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{ticker}_prediction.png", dpi=150)
    print(f"Saved plot to {ticker}_prediction.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    main(args.ticker, args.start, args.end, args.window, args.epochs, args.batch_size)