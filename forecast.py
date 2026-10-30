import pandas as pd
import numpy as np

def detect_columns(filepath):
    """
    Inspects CSV header and first few rows to auto-detect date and numerical sales columns.
    """
    df = pd.read_csv(filepath, nrows=100)
    
    date_candidates = []
    numeric_candidates = []
    
    for col in df.columns:
        col_clean = str(col).strip()
        # Check for numeric type
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_candidates.append(col_clean)
        else:
            # Try parsing as date
            try:
                sample_non_null = df[col].dropna().astype(str).head(10)
                if len(sample_non_null) > 0:
                    pd.to_datetime(sample_non_null, errors='raise')
                    date_candidates.append(col_clean)
            except Exception:
                pass
                
    # If no date candidate found via parsing, check column names
    if not date_candidates:
        for col in df.columns:
            if any(kw in col.lower() for kw in ['date', 'time', 'day', 'month', 'year', 'ds', 'timestamp']):
                date_candidates.append(col)

    # If no numeric candidates, find any column with numbers
    if not numeric_candidates:
        for col in df.columns:
            if col not in date_candidates:
                numeric_candidates.append(col)
                
    # Best default picks
    best_date = date_candidates[0] if date_candidates else df.columns[0]
    
    sales_keywords = ['sales', 'revenue', 'amount', 'total', 'val', 'price', 'quantity', 'units']
    best_value = None
    for nc in numeric_candidates:
        if any(kw in nc.lower() for kw in sales_keywords):
            best_value = nc
            break
    if not best_value and numeric_candidates:
        best_value = numeric_candidates[0]
    elif not best_value:
        best_value = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
    return {
        "columns": list(df.columns),
        "date_candidates": date_candidates,
        "numeric_candidates": numeric_candidates,
        "default_date": best_date,
        "default_value": best_value
    }


def prepare_time_series(filepath, date_col, value_col, resample_freq='D'):
    """
    Loads CSV, parses date & value columns, aggregates/resamples, fills missing values,
    and returns clean time series DataFrame & metadata.
    """
    df = pd.read_csv(filepath)
    
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Specified columns ({date_col}, {value_col}) not found in CSV.")
        
    # Clean up values
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    
    # Convert target value to numeric
    df[value_col] = pd.to_numeric(df[value_col].astype(str).str.replace(',', '').str.replace('$', '').str.replace('₹', ''), errors='coerce')
    df = df.dropna(subset=[value_col])
    
    # Sort chronologically
    df = df.sort_values(by=date_col)
    
    # Group by date if multiple entries exist per timestamp
    ts_df = df.groupby(date_col)[value_col].sum().reset_index()
    ts_df.set_index(date_col, inplace=True)
    
    # Resample to specified frequency ('D'=Daily, 'W'=Weekly, 'M'=Monthly)
    freq_map = {'D': 'D', 'W': 'W-MON', 'M': 'MS'}
    actual_freq = freq_map.get(resample_freq, 'D')
    
    ts_resampled = ts_df[value_col].resample(actual_freq).sum()
    
    # Interpolate missing values (e.g. gaps in days)
    ts_resampled = ts_resampled.interpolate(method='linear').bfill().ffill()
    
    # Ensure positive/non-negative sales values
    ts_resampled = ts_resampled.clip(lower=0)
    
    summary = {
        "total_records": len(ts_resampled),
        "start_date": ts_resampled.index.min().strftime('%Y-%m-%d'),
        "end_date": ts_resampled.index.max().strftime('%Y-%m-%d'),
        "total_sales": float(round(ts_resampled.sum(), 2)),
        "mean_sales": float(round(ts_resampled.mean(), 2)),
        "min_sales": float(round(ts_resampled.min(), 2)),
        "max_sales": float(round(ts_resampled.max(), 2)),
        "std_sales": float(round(ts_resampled.std(), 2))
    }
    
    return ts_resampled, summary
