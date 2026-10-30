import os
import io
import csv
from flask import Flask, render_template, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename
import pandas as pd

from forecast import detect_columns, prepare_time_series
from model import train_and_forecast
from generate_sample_data import generate_sample_sales_csv

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
SAMPLE_DIR = os.path.join(BASE_DIR, 'sample_data')
SAMPLE_FILE = os.path.join(SAMPLE_DIR, 'retail_sales_sample.csv')
ALLOWED_EXTENSIONS = {'csv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            col_info = detect_columns(filepath)
            df = pd.read_csv(filepath, nrows=10)
            preview = df.to_dict(orient='records')
            
            return jsonify({
                "message": "File uploaded successfully.",
                "filename": filename,
                "filepath": filepath,
                "columns": col_info["columns"],
                "date_candidates": col_info["date_candidates"],
                "numeric_candidates": col_info["numeric_candidates"],
                "default_date": col_info["default_date"],
                "default_value": col_info["default_value"],
                "preview": preview,
                "total_rows_preview": len(df)
            })
        except Exception as e:
            return jsonify({"error": f"Error parsing CSV file: {str(e)}"}), 500
    else:
        return jsonify({"error": "Only .csv files are supported."}), 400


@app.route('/api/sample-data', methods=['GET'])
def get_sample_data():
    if not os.path.exists(SAMPLE_FILE):
        generate_sample_sales_csv(SAMPLE_FILE)
        
    try:
        col_info = detect_columns(SAMPLE_FILE)
        df = pd.read_csv(SAMPLE_FILE, nrows=10)
        preview = df.to_dict(orient='records')
        
        return jsonify({
            "message": "Sample dataset loaded successfully.",
            "filename": "retail_sales_sample.csv",
            "is_sample": True,
            "filepath": SAMPLE_FILE,
            "columns": col_info["columns"],
            "date_candidates": col_info["date_candidates"],
            "numeric_candidates": col_info["numeric_candidates"],
            "default_date": col_info["default_date"],
            "default_value": col_info["default_value"],
            "preview": preview
        })
    except Exception as e:
        return jsonify({"error": f"Error loading sample dataset: {str(e)}"}), 500


@app.route('/api/forecast', methods=['POST'])
def run_forecast_api():
    data = request.get_json() or {}
    
    filename = data.get('filename')
    is_sample = data.get('is_sample', False)
    date_col = data.get('date_col')
    value_col = data.get('value_col')
    resample_freq = data.get('resample_freq', 'D')
    forecast_horizon = int(data.get('forecast_horizon', 30))
    auto_arima = data.get('auto_arima', True)
    
    p = int(data.get('p', 1))
    d = int(data.get('d', 1))
    q = int(data.get('q', 1))
    
    if is_sample or filename == "retail_sales_sample.csv":
        filepath = SAMPLE_FILE
        if not os.path.exists(filepath):
            generate_sample_sales_csv(filepath)
    else:
        if not filename:
            return jsonify({"error": "Filename is required."}), 400
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        if not os.path.exists(filepath):
            return jsonify({"error": "File not found on server."}), 400

    try:
        # Preprocess time series
        ts_series, summary = prepare_time_series(filepath, date_col, value_col, resample_freq)
        
        order = None if auto_arima else (p, d, q)
        
        # Train ARIMA model and compute forecast
        results = train_and_forecast(
            series=ts_series,
            forecast_horizon=forecast_horizon,
            order=order,
            auto_arima=auto_arima
        )
        
        return jsonify({
            "status": "success",
            "summary": summary,
            "forecast_results": results
        })
        
    except Exception as e:
        return jsonify({"error": f"Forecasting failed: {str(e)}"}), 500


@app.route('/api/export-csv', methods=['POST'])
def export_csv():
    data = request.get_json() or {}
    forecast_data = data.get('forecast', [])
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Date', 'Forecasted_Sales', 'Lower_Bound_95', 'Upper_Bound_95'])
    
    for row in forecast_data:
        writer.writerow([
            row.get('date', ''),
            row.get('forecast', 0.0),
            row.get('lower_bound', 0.0),
            row.get('upper_bound', 0.0)
        ])
        
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=sales_forecast_report.csv"}
    )

if __name__ == '__main__':
    # Ensure sample dataset exists at startup
    if not os.path.exists(SAMPLE_FILE):
        generate_sample_sales_csv(SAMPLE_FILE)
    print("Starting Sales Forecasting Web Application on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
