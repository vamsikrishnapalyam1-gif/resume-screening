import pandas as pd
import numpy as np
import os

def generate_sample_sales_csv(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.random.seed(42)
    # Generate 730 daily dates (2 years: Jan 2024 to Dec 2025)
    dates = pd.date_range(start="2024-01-01", periods=730, freq="D")
    
    # Base trend (gradual growth over 2 years)
    trend = np.linspace(1200, 2500, 730)
    
    # Weekly seasonality (higher sales on Fri, Sat, Sun)
    day_of_week_mult = np.array([0.85, 0.90, 0.95, 1.0, 1.15, 1.35, 1.25])
    weekly_pattern = np.array([day_of_week_mult[d.weekday()] for d in dates])
    
    # Monthly / Annual seasonality (summer peak, Nov/Dec holiday spike)
    day_of_year = dates.dayofyear
    annual_pattern = 1.0 + 0.15 * np.sin(2 * np.pi * day_of_year / 365) + 0.25 * np.exp(-((day_of_year - 350)/20)**2)
    
    # Random noise
    noise = np.random.normal(loc=1.0, scale=0.08, size=730)
    
    # Calculate final daily sales
    sales = (trend * weekly_pattern * annual_pattern * noise).round(2)
    
    # Create DataFrame
    df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Sales": sales,
        "Store_ID": np.random.choice(["Store_101", "Store_102"], size=730),
        "Transactions": (sales / np.random.uniform(35, 55, size=730)).astype(int)
    })
    
    df.to_csv(output_path, index=False)
    print(f"Sample dataset saved to: {output_path} ({len(df)} rows)")

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "sample_data", "retail_sales_sample.csv")
    generate_sample_sales_csv(out)
