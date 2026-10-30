# Sales Forecasting using Time Series Analysis

A full-stack web application for predicting future sales revenue and volume using statistical **Time Series Analysis (ARIMA / SARIMA)**. The system enables business owners and analysts to upload CSV sales datasets, visualize historical trends, auto-tune ARIMA orders, compute accuracy metrics (MAE, RMSE, MAPE), and export projected forecasts.

---

## Key Features

- **Ingest CSV Sales Data**: Auto-detects Date and Revenue/Sales columns with support for missing data interpolation and daily/weekly/monthly frequency aggregation.
- **Auto-ARIMA Optimization**: Grid-searches optimal `(p, d, q)` ARIMA parameters using Akaike Information Criterion (AIC) or allows custom parameter overrides.
- **Train-Test Model Evaluation**: Evaluates backtest performance against holdout test data using **MAE**, **RMSE**, and **MAPE** metrics.
- **Interactive Visualizations**: Powered by Chart.js with historical trend lines, model test fit curves, projected forecast curves, and 95% confidence intervals.
- **Flexible Forecast Horizons**: Select 7, 30, 90, or 180-day forecast horizons.
- **Forecast CSV Export**: Download generated forecast reports including dates, predictions, lower bounds, and upper bounds with a single click.
- **Instant Demo Dataset**: Includes a built-in 2-year daily retail dataset for immediate out-of-the-box testing.

---

## Project Structure

```
sales_forecasting_app/
├── app.py                  # Flask application entry point & REST API endpoints
├── model.py                # ARIMA/SARIMA fitting, order selection, metric evaluation
├── forecast.py             # Data preprocessing, column detection, resampling & interpolation
├── generate_sample_data.py # Generator script for 2-year retail sales sample dataset
├── templates/
│   └── index.html          # Dashboard HTML template with responsive glassmorphism design
├── static/
│   ├── css/
│   │   └── style.css       # Dark mode theme styles & glassmorphic UI tokens
│   └── js/
│       └── main.js         # Chart.js visualization logic and AJAX controller
├── uploads/                # Directory for user-uploaded CSV files
├── models/                 # Directory for model output artifacts
├── sample_data/
│   └── retail_sales_sample.csv # Built-in sample dataset for 1-click testing
├── requirements.txt        # Python package dependencies
└── README.md               # Setup and usage guide
```

---

## Installation & Setup

### Prerequisites
- Python 3.9 or higher

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
python app.py
```

### 3. Open in Browser
Navigate to `http://127.0.0.1:5000` in your web browser.

---

## How to Use

1. **Try Sample Data**: Click the **"Try Sample Retail Dataset"** button on the top right for instant 1-click analysis.
2. **Upload Custom Dataset**: Drag & drop any CSV sales dataset into the uploader zone.
3. **Configure Parameters**:
   - Select your Date column and Target Sales column.
   - Choose your Forecast Horizon (e.g. 7, 30, or 90 days).
   - Choose Resample Frequency (Daily, Weekly, Monthly).
   - Choose Auto-Fit ARIMA or manually enter `p`, `d`, `q`.
4. **Generate Forecast**: Click **"Generate Sales Forecast"**.
5. **Analyze Results**: Review historical trend, forecast confidence intervals, KPI cards, and evaluation metrics (MAE, RMSE, MAPE).
6. **Download Report**: Click **"Download CSV Report"** to export the predictions.

---

## Technical Details & Metrics

- **MAE (Mean Absolute Error)**: Measures average absolute dollar magnitude of errors.
- **RMSE (Root Mean Squared Error)**: Standard deviation of residuals; penalizes larger errors.
- **MAPE (Mean Absolute Percentage Error)**: Percentage error relative to actual sales values.
- **Accuracy Score**: Calculated as `max(0, 100% - MAPE)`.
