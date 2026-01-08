# Stock Predictor App - Assumptions

This document outlines the core assumptions and design choices for the Stock Predictor Streamlit app.

- **Frequency**: The model operates on daily frequency. All features and targets are based on daily open, high, low, close, and volume data.
- **Target**: The primary prediction target is the **h-day forward return**, defined as `(close[t+h] / close[t]) - 1`. The app then converts this return back to a price level for display.
- **Signal Timing**: The model uses data available up to and including the market close at day `t` to predict the closing price at day `t+h`. There is no lookahead bias.
- **Validation Strategy**:
    - **Time-Series Split**: Data is split into training and testing sets based on time. The first 80% of the data is used for training, and the remaining 20% is used for out-of-sample testing.
    - **No Shuffling**: The temporal order of the data is strictly preserved.
- **Benchmark Baseline**: All models are compared against a "random walk" baseline, which assumes a zero return (`predict 0 return`). This serves as a sanity check to ensure any model is adding value.
- **Data Source**: Uses `yfinance` to source daily stock data, as it requires no API keys and is sufficient for this stage.
- **Error Handling**: The application will display an error and stop execution if the selected date range does not provide enough data to generate the required features and perform a train/test split (e.g., `lookback + horizon + buffer`).
- **Feature Set**: The initial feature set is intentionally minimal and stable, focused on common technical indicators derived from closing price and volume.
- **Model Simplicity**: Simple, interpretable models (Ridge, RandomForest) and avoids complex architectures like LSTMs or Transformers. Hyperparameter tuning is limited to a few key parameters exposed in the UI.
