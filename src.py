import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Phase 1: Data + Caching ---

@st.cache_data
def get_daily_bars(ticker, start, end):
    """
    Fetches daily historical stock data from yfinance.
    - Cleans data: sorts by date, drops duplicates.
    - Ensures a business-day index.
    Returns DataFrame with columns: Open, High, Low, Close, Volume.
    """
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # Handle MultiIndex columns (yfinance sometimes returns this)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    df = df.sort_index()
    df = df.loc[~df.index.duplicated(keep='first')]
    # Reindex to ensure we have all business days, forward-filling missing values
    bday_index = pd.bdate_range(start=df.index.min(), end=df.index.max())
    df = df.reindex(bday_index, method='ffill')
    df.index.name = 'date'

    return df

# --- Phase 2: Feature Pipeline ---
@st.cache_data
def compute_features(df, horizon):
    """
    Computes a set of features from the daily bar data.
    Features: returns (1d, 5d, 20d), rolling vol (20d), momentum (20d), 
    MA ratios (20, 50), volume z-score (20d).
    Target: h-day forward return = close[t+h]/close[t] - 1
    """
    close = df['Close']
    volume = df['Volume']

    # Returns
    features = pd.DataFrame(index=df.index)
    features['ret_1d'] = close.pct_change(1)
    features['ret_5d'] = close.pct_change(5)
    features['ret_20d'] = close.pct_change(20)

    # Rolling Volatility
    features['vol_20d'] = features['ret_1d'].rolling(20).std()

    # Momentum
    features['mom_20d'] = features['ret_20d'] # same as 20d return

    # Moving Average Ratios
    ma_20 = close.rolling(20).mean()
    ma_50 = close.rolling(50).mean()
    features['ma_ratio_20'] = close / ma_20 - 1
    features['ma_ratio_50'] = close / ma_50 - 1

    # Volume Z-Score
    vol_mean_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std()
    features['vol_zscore_20'] = (volume - vol_mean_20) / vol_std_20

    # Target
    y = close.shift(-horizon) / close - 1
    y.name = 'y_h'

    # Align X and y
    X = features
    df_aligned = pd.concat([X, y], axis=1)
    df_aligned = df_aligned.dropna(how='any')

    X = df_aligned.drop(columns=['y_h'])
    y = df_aligned['y_h']

    return X, y

# --- Phase 3: Model Registry ---
from sklearn.preprocessing import StandardScaler

class BaselineModel:
    def fit(self, X_train, y_train):
        pass
    def predict(self, X_test):
        return np.zeros(len(X_test))

class RidgeModel:
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)

    def predict(self, X_test):
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)

class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=10, n_jobs=-1, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

MODELS = {
    "Baseline": BaselineModel,
    "Ridge": RidgeModel,
    "RandomForest": RandomForestModel,
}

# --- Phase 4: Evaluation ---
def compute_evaluation_metrics(y_true, y_pred):
    """Computes MAE, RMSE, and directional accuracy."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    y_true_sign = np.sign(y_true)
    y_pred_sign = np.sign(y_pred)
    y_true_sign[y_true_sign == 0] = 1 
    y_pred_sign[y_pred_sign == 0] = 1
    dir_acc = (y_true_sign == y_pred_sign).mean()
    
    return {'mae': mae, 'rmse': rmse, 'dir_acc': dir_acc}

def get_latest_prediction(model, X, last_close, horizon):
    """Get the prediction for the most recent data point."""
    latest_features = X.iloc[[-1]]
    predicted_return = model.predict(latest_features)[0]
    predicted_price = last_close * (1 + predicted_return)
    
    last_date = latest_features.index[0]
    # Calculate target date using business days (h trading days forward)
    # pd.bdate_range automatically handles weekends/holidays
    bdays = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
    if len(bdays) > 0:
        target_date = bdays[-1]
    else:
        # Fallback if horizon is 0 or calculation fails
        target_date = last_date + pd.Timedelta(days=horizon)
    
    return {
        "predicted_return_pct": predicted_return * 100,
        "predicted_price": predicted_price,
        "target_date": target_date,
        "last_date": last_date
    }

def main():
    st.set_page_config(page_title="Stock Predictor", layout="wide")
    st.title("📈 Stock Predictor MVP")

    with st.expander("Assumptions"):
        st.markdown("""
        - **Target**: Predict h-day forward return: `(close[t+h] / close[t]) - 1`.
        - **Signal Timing**: Use data up to `close[t]` to predict `close[t+h]`. No lookahead.
        - **Validation**: Time-series split (80% train / 20% test). No shuffling.
        - **Baseline**: "Predict 0 return" (random walk) for comparison.
        - **Features**: Minimal set from close and volume (returns, volatility, momentum, etc.).
        """)

    # --- Sidebar ---
    with st.sidebar:
        st.header("Configuration")
        # Use session state to remember last settings
        if 'last_run_config' not in st.session_state:
            st.session_state.last_run_config = {
                "ticker": "AAPL",
                "date_range": [pd.to_datetime("2020-01-01"), pd.to_datetime("2023-12-31")],
                "horizon": 5,
                "model_choice": "RandomForest",
                "model_params": {"alpha": 1.0, "n_estimators": 100, "max_depth": 10}
            }
        cfg = st.session_state.last_run_config

        with st.form(key='config_form'):
            ticker = st.text_input("Ticker", cfg['ticker'])
            date_range = st.date_input("Date Range", cfg['date_range'])
            horizon = st.slider("Prediction Horizon (days)", 1, 30, cfg['horizon'])
            model_choice = st.selectbox("Select Model", list(MODELS.keys()), index=list(MODELS.keys()).index(cfg['model_choice']))

            model_params = {}
            if model_choice == "Ridge":
                model_params['alpha'] = st.slider("Alpha", 0.01, 10.0, cfg['model_params'].get('alpha', 1.0), 0.01)
            elif model_choice == "RandomForest":
                model_params['n_estimators'] = st.slider("Number of Estimators", 50, 500, cfg['model_params'].get('n_estimators', 100), 50)
                model_params['max_depth'] = st.slider("Max Depth", 3, 20, cfg['model_params'].get('max_depth', 10), 1)

            run_button = st.form_submit_button(label='Run')

    if not run_button:
        st.info("Configure parameters on the left and click 'Run'.")
        # Optionally display last run's results if they exist
        if 'last_run_results' in st.session_state:
             display_results(st.session_state.last_run_results)
        return
    
    # --- Save current config ---
    st.session_state.last_run_config = {
        "ticker": ticker,
        "date_range": date_range,
        "horizon": horizon,
        "model_choice": model_choice,
        "model_params": model_params,
    }

    # --- Main Panel Execution ---
    start_date, end_date = date_range
    
    with st.spinner("Loading data → building features → training → predicting..."):
        results = run_pipeline(ticker, start_date, end_date, horizon, model_choice, model_params)
    
    if results:
        st.session_state.last_run_results = results
        display_results(results)

def run_pipeline(ticker, start_date, end_date, horizon, model_choice, model_params):
    # 1. Data
    data = get_daily_bars(ticker, start_date, end_date)
    if data.empty:
        st.error(f"No data found for ticker {ticker}. Please check the ticker symbol and date range.")
        return None

    # 2. Features
    X, y = compute_features(data, horizon)
    
    # Minimum data size: max lookback (50 for MA50) + horizon + buffer for train/test split
    # Buffer: need at least 50 rows for test set (20% of 250 = 50)
    lookback = 50  # Maximum feature lookback period
    buffer = 50    # Minimum rows for meaningful train/test split
    min_data_size = lookback + horizon + buffer
    
    if len(X) < min_data_size:
        st.error(f"Not enough data ({len(X)} rows) for {ticker}. Need at least {min_data_size} rows (lookback={lookback} + horizon={horizon} + buffer={buffer}). Please select a longer date range.")
        return None

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 4. Model Training
    model_class = MODELS[model_choice]
    model = model_class(**model_params)
    model.fit(X_train, y_train)

    # 5. Prediction & Evaluation
    test_predictions = pd.Series(model.predict(X_test), index=X_test.index)
    metrics = compute_evaluation_metrics(y_test, test_predictions)
    
    baseline_preds = BaselineModel().predict(X_test)
    baseline_metrics = compute_evaluation_metrics(y_test, baseline_preds)

    last_close = data['Close'].iloc[-1]
    latest_pred_info = get_latest_prediction(model, X, last_close, horizon)

    return {
        "ticker": ticker,
        "last_close": last_close,
        "latest_pred_info": latest_pred_info,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "test_predictions": test_predictions,
        "y_test": y_test,
        "X": X,
        "y": y,
        "data": data,
        "model_choice": model_choice,
    }

def display_results(results):
    st.header(f"Results for {results['ticker']}")

    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Last Price", f"${results['last_close']:,.2f}")
    col2.metric("Predicted Return", f"{results['latest_pred_info']['predicted_return_pct']:,.2f}%")
    col3.metric("Predicted Price", f"${results['latest_pred_info']['predicted_price']:,.2f}")
    col4.metric("Test RMSE (returns)", f"{results['metrics']['rmse']:.4f}")
    col5.metric("Baseline RMSE", f"{results['baseline_metrics']['rmse']:.4f}")
    
    st.divider()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Price Forecast", "Model Performance", "Data Preview"])

    with tab1:
        st.subheader("Price + Forecast")
        
        price_data = results['data'][['Close']].copy()
        forecast_date = results['latest_pred_info']['target_date']
        forecast_price = results['latest_pred_info']['predicted_price']
        last_date = results['latest_pred_info']['last_date']
        
        # Create chart data with historical prices and forecast point
        chart_data = price_data.copy()
        chart_data.columns = ['Close Price']
        
        # Add forecast point
        forecast_df = pd.DataFrame({'Close Price': [forecast_price]}, index=[forecast_date])
        chart_data = pd.concat([chart_data, forecast_df])
        chart_data = chart_data.sort_index()
        
        # Create a line connecting last price to forecast
        connection_df = pd.DataFrame({
            'Close Price': [price_data.loc[last_date, 'Close'], forecast_price]
        }, index=[last_date, forecast_date])
        
        # Use plotly or altair for better visualization, but for MVP, use streamlit's line_chart
        # Mark the forecast point clearly
        st.line_chart(chart_data)
        
        # Add annotation text
        bdays_count = len(pd.bdate_range(start=last_date + pd.Timedelta(days=1), end=forecast_date))
        st.caption(f"Forecast: ${forecast_price:,.2f} on {forecast_date.strftime('%Y-%m-%d')} ({bdays_count} trading days ahead)")

    with tab2:
        st.subheader("Model Performance")
        
        # Metrics comparison table
        perf_df = pd.DataFrame({
            'MAE (returns)': [results['metrics']['mae'], results['baseline_metrics']['mae']],
            'RMSE (returns)': [results['metrics']['rmse'], results['baseline_metrics']['rmse']],
            'Directional Accuracy': [f"{results['metrics']['dir_acc']:.2%}", f"{results['baseline_metrics']['dir_acc']:.2%}"]
        }, index=[results['model_choice'], "Baseline"])
        st.dataframe(perf_df.style.format({'MAE (returns)': '{:.6f}', 'RMSE (returns)': '{:.6f}'}))
        
        # Predicted vs Actual returns chart
        pred_vs_actual = pd.DataFrame({
            'Actual Return': results['y_test'],
            'Predicted Return': results['test_predictions']
        })
        st.line_chart(pred_vs_actual)
        
        # Scatter plot for better visualization
        scatter_data = pd.DataFrame({
            'Actual': results['y_test'].values,
            'Predicted': results['test_predictions'].values
        })
        st.scatter_chart(scatter_data)
        
        # Download Button
        download_df = pd.DataFrame({
            'date': results['y_test'].index,
            'y_true': results['y_test'].values,
            'y_pred': results['test_predictions'].values
        })
        
        @st.cache_data
        def to_csv(df):
            return df.to_csv(index=False).encode('utf-8')
        
        csv = to_csv(download_df)
        st.download_button(
             label="Download Test Predictions CSV",
             data=csv,
             file_name=f"{results['ticker']}_test_predictions.csv",
             mime='text/csv',
        )

    with tab3:
        st.subheader("Data Preview (Features + Target)")
        st.dataframe(pd.concat([results['X'], results['y']], axis=1).tail(10))

if __name__ == '__main__':
    main()
    