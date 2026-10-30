import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

HAS_STATSMODELS = False
HAS_SKLEARN = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def find_best_arima_order(series, max_p=2, max_d=1, max_q=2):
    """
    Grid searches ARIMA orders (p,d,q) to find the order with lowest AIC.
    """
    if not HAS_STATSMODELS:
        return (1, 1, 1), 0.0

    best_aic = float("inf")
    best_order = (1, 1, 1)
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    res = model.fit()
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
                    
    return best_order, best_aic


def evaluate_forecast_metrics(y_true, y_pred):
    """
    Calculates MAE, RMSE, and MAPE metrics using sklearn or pure numpy.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    if HAS_SKLEARN:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    else:
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
    mask = y_true != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0
        
    accuracy = max(0.0, round(100.0 - mape, 2))
    
    return {
        "mae": float(round(mae, 2)),
        "rmse": float(round(rmse, 2)),
        "mape": float(round(mape, 2)),
        "accuracy": accuracy
    }


def fallback_exponential_smoothing_forecast(series, forecast_horizon=30, alpha=0.3, beta=0.1):
    """
    Fallback Holt's Linear Exponential Smoothing forecasting when statsmodels is not installed.
    """
    y = series.values.astype(float)
    n = len(y)
    
    level = y[0]
    trend = y[1] - y[0] if n > 1 else 0.0
    
    fitted = np.zeros(n)
    fitted[0] = level
    
    for t in range(1, n):
        val = y[t]
        last_level = level
        level = alpha * val + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        fitted[t] = level
        
    # Generate future horizon predictions
    future = [max(0.0, level + (h + 1) * trend) for h in range(forecast_horizon)]
    
    # Calculate residual standard deviation for bounds
    residuals = y - fitted
    std_err = np.std(residuals) if len(residuals) > 1 else np.std(y) * 0.1
    if std_err == 0:
        std_err = np.mean(y) * 0.05
        
    conf_bounds = []
    for h, f_val in enumerate(future):
        margin = 1.96 * std_err * np.sqrt(1 + (h + 1) * 0.05)
        conf_bounds.append((max(0.0, f_val - margin), f_val + margin))
        
    return fitted, future, conf_bounds


def train_and_forecast(series, forecast_horizon=30, order=None, seasonal_order=None, auto_arima=True):
    """
    Main ML workflow:
    1. Splits data into train and test set
    2. Fits ARIMA (or Exponential Smoothing fallback) and evaluates metrics
    3. Forecasts future horizon with 95% confidence intervals.
    """
    n = len(series)
    if n < 5:
        raise ValueError("Time series is too short for forecasting (requires at least 5 records).")
        
    test_size = max(5, int(n * 0.2))
    if test_size >= n - 2:
        test_size = max(2, int(n * 0.15))
        
    train_series = series.iloc[:-test_size]
    test_series = series.iloc[-test_size:]
    
    if HAS_STATSMODELS:
        if auto_arima or order is None:
            selected_order, _ = find_best_arima_order(train_series)
        else:
            selected_order = tuple(order)
            
        try:
            if seasonal_order:
                model_eval = SARIMAX(train_series, order=selected_order, seasonal_order=tuple(seasonal_order))
            else:
                model_eval = ARIMA(train_series, order=selected_order)
                
            fit_eval = model_eval.fit()
            test_pred = fit_eval.forecast(steps=test_size).values
        except Exception:
            selected_order = (1, 1, 1)
            model_eval = ARIMA(train_series, order=selected_order)
            fit_eval = model_eval.fit()
            test_pred = fit_eval.forecast(steps=test_size).values
            
        # Fit on full dataset
        try:
            if seasonal_order:
                model_full = SARIMAX(series, order=selected_order, seasonal_order=tuple(seasonal_order))
            else:
                model_full = ARIMA(series, order=selected_order)
            fit_full = model_full.fit()
            
            forecast_res = fit_full.get_forecast(steps=forecast_horizon)
            future_preds = forecast_res.predicted_mean.values
            conf_int = forecast_res.conf_int(alpha=0.05).values
            future_bounds = [(max(0, lower), max(val, upper)) for val, (lower, upper) in zip(future_preds, conf_int)]
        except Exception:
            _, future_preds, future_bounds = fallback_exponential_smoothing_forecast(series, forecast_horizon)
            
    else:
        # Pure Python / Numpy fallback if statsmodels is not installed
        selected_order = (1, 1, 1)
        _, test_pred, _ = fallback_exponential_smoothing_forecast(train_series, test_size)
        _, future_preds, future_bounds = fallback_exponential_smoothing_forecast(series, forecast_horizon)

    metrics = evaluate_forecast_metrics(test_series.values, test_pred)
    
    test_dates = [d.strftime('%Y-%m-%d') for d in test_series.index]
    test_actual = [float(round(val, 2)) for val in test_series.values]
    test_predicted = [float(round(max(0, val), 2)) for val in test_pred]
    
    last_date = series.index[-1]
    freq = series.index.freq
    if freq is None:
        freq = pd.infer_freq(series.index)
    if freq is None:
        freq = 'D'
        
    future_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq=freq)[1:]
    
    future_forecast_list = []
    total_forecasted_sales = 0.0
    
    for dt, val, (lower, upper) in zip(future_dates, future_preds, future_bounds):
        val_clean = float(round(max(0, val), 2))
        lower_clean = float(round(max(0, lower), 2))
        upper_clean = float(round(max(val_clean, upper), 2))
        
        total_forecasted_sales += val_clean
        future_forecast_list.append({
            "date": dt.strftime('%Y-%m-%d'),
            "forecast": val_clean,
            "lower_bound": lower_clean,
            "upper_bound": upper_clean
        })
        
    historical_dates = [d.strftime('%Y-%m-%d') for d in series.index]
    historical_values = [float(round(v, 2)) for v in series.values]
    
    return {
        "order": selected_order,
        "engine": "ARIMA/Statsmodels" if HAS_STATSMODELS else "Holt's Exponential Smoothing (Fallback)",
        "metrics": metrics,
        "historical": {
            "dates": historical_dates,
            "values": historical_values
        },
        "test_eval": {
            "dates": test_dates,
            "actual": test_actual,
            "predicted": test_predicted
        },
        "future_forecast": future_forecast_list,
        "summary": {
            "forecast_horizon": forecast_horizon,
            "total_forecasted_sales": float(round(total_forecasted_sales, 2)),
            "avg_daily_forecast": float(round(total_forecasted_sales / forecast_horizon, 2)),
            "historical_total": float(round(np.sum(historical_values), 2)),
            "historical_mean": float(round(np.mean(historical_values), 2))
        }
    }
