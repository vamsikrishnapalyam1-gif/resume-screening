import os
import sys

# Ensure local imports work
sys.path.insert(0, os.path.dirname(__file__))

from generate_sample_data import generate_sample_sales_csv
from forecast import detect_columns, prepare_time_series
from model import train_and_forecast

def run_pipeline_test():
    sample_file = os.path.join(os.path.dirname(__file__), "sample_data", "retail_sales_sample.csv")
    print("[1] Generating sample dataset...")
    generate_sample_sales_csv(sample_file)
    assert os.path.exists(sample_file), "Sample dataset file not created!"

    print("[2] Testing column detection...")
    cols_info = detect_columns(sample_file)
    print("Detected columns:", cols_info)
    assert cols_info["default_date"] == "Date"
    assert cols_info["default_value"] == "Sales"

    print("[3] Testing time series preprocessing...")
    ts_series, summary = prepare_time_series(sample_file, "Date", "Sales", resample_freq='D')
    print("Summary statistics:", summary)
    assert summary["total_records"] == 730
    assert summary["total_sales"] > 0

    print("[4] Testing ARIMA training & forecasting...")
    results = train_and_forecast(
        series=ts_series,
        forecast_horizon=30,
        auto_arima=True
    )
    
    print("Fitted ARIMA Order:", results["order"])
    print("Evaluation Metrics (MAE, RMSE, MAPE, Accuracy):", results["metrics"])
    print("Future Forecast Count:", len(results["future_forecast"]))
    
    assert results["metrics"]["mae"] >= 0
    assert results["metrics"]["rmse"] >= 0
    assert len(results["future_forecast"]) == 30
    assert "lower_bound" in results["future_forecast"][0]
    assert "upper_bound" in results["future_forecast"][0]
    
    print("\nSUCCESS: All backend tests passed clean!")

if __name__ == "__main__":
    run_pipeline_test()
