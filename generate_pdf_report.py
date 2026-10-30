import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
from reportlab.lib.units import inch

def create_project_pdf(output_path):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    primary_color = colors.HexColor("#1e293b")   # Dark Slate
    accent_color = colors.HexColor("#4f46e5")    # Indigo Accent
    secondary_color = colors.HexColor("#0284c7") # Cyan Accent
    light_bg = colors.HexColor("#f8fafc")        # Soft Light Grey
    border_color = colors.HexColor("#cbd5e1")    # Light Border

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=primary_color,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=accent_color,
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=6
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=accent_color,
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6
    )

    bold_body = ParagraphStyle(
        'BoldBody',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b")
    )

    story = []

    # Title & Header
    story.append(Paragraph("Sales Forecasting using Time Series Analysis", title_style))
    story.append(Paragraph("Full-Stack Web Application Project Report & Presentation Guide", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=accent_color, spaceAfter=15))

    # Executive Overview
    story.append(Paragraph("1. Project Abstract & Problem Statement", h1_style))
    abstract_text = (
        "Businesses frequently face operational challenges in predicting future sales demand accurately. "
        "Manual sales forecasting and static spreadsheets are time-consuming, prone to human error, and fail "
        "to capture underlying time-series trends and seasonal patterns. Inaccurate predictions lead to "
        "severe consequences such as <b>stock shortages (lost revenue)</b> or <b>overstocking (high holding costs and financial waste)</b>.<br/><br/>"
        "This project delivers an end-to-end full-stack web-based solution that automates sales forecasting using "
        "<b>Time Series Analysis (ARIMA / SARIMA)</b>. Users can upload any sales CSV dataset, auto-detect dates "
        "and revenue columns, visualize historical trends, train forecasting models, evaluate accuracy metrics in Indian Rupees (₹), "
        "and export projected forecasts."
    )
    story.append(Paragraph(abstract_text, body_style))

    # Tech Stack Table
    story.append(Paragraph("2. Technical Architecture & Tech Stack", h1_style))
    tech_data = [
        [Paragraph("<b>Layer</b>", bold_body), Paragraph("<b>Technology / Tools</b>", bold_body), Paragraph("<b>Purpose</b>", bold_body)],
        [Paragraph("Frontend UI", body_style), Paragraph("HTML5, CSS3, JavaScript (ES6+)", body_style), Paragraph("Responsive dark glassmorphism analytics dashboard", body_style)],
        [Paragraph("Data Charts", body_style), Paragraph("Chart.js", body_style), Paragraph("Interactive trend, backtest, and forecast visualization", body_style)],
        [Paragraph("Backend Framework", body_style), Paragraph("Python Flask (REST APIs)", body_style), Paragraph("Data ingestion, file handling, and API endpoints", body_style)],
        [Paragraph("Machine Learning", body_style), Paragraph("Statsmodels (ARIMA / SARIMA)", body_style), Paragraph("Time series model fitting, AIC optimization & prediction", body_style)],
        [Paragraph("Data Processing", body_style), Paragraph("Pandas, NumPy", body_style), Paragraph("CSV parsing, linear interpolation, frequency resampling", body_style)],
        [Paragraph("Model Evaluation", body_style), Paragraph("Scikit-Learn", body_style), Paragraph("Backtesting evaluation (MAE, RMSE, MAPE metrics)", body_style)]
    ]

    t_tech = Table(tech_data, colWidths=[1.3*inch, 2.0*inch, 3.7*inch])
    t_tech.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), light_bg),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_tech)
    story.append(Spacer(1, 10))

    # Core Features & Machine Learning Methodology
    story.append(Paragraph("3. Machine Learning Methodology (ARIMA / SARIMA)", h1_style))
    ml_explain = (
        "The system employs <b>ARIMA (AutoRegressive Integrated Moving Average)</b> time-series modeling:<br/>"
        "• <b>p (AutoRegressive Order)</b>: Uses linear combination of past sales values to predict future values.<br/>"
        "• <b>d (Differencing Degree)</b>: Subtracts consecutive values to make non-stationary trend data stationary.<br/>"
        "• <b>q (Moving Average Order)</b>: Uses past forecast errors in a regression-like model to smooth unexpected noise.<br/><br/>"
        "<b>Automatic Model Tuning (Auto-ARIMA)</b>: The engine grid-searches parameter combinations <i>(p, d, q)</i> "
        "and selects the optimal parameters by minimizing the <b>AIC (Akaike Information Criterion)</b> value."
    )
    story.append(Paragraph(ml_explain, body_style))

    # Evaluation Metrics Table
    story.append(Paragraph("4. Model Evaluation & Accuracy Metrics", h1_style))
    metrics_data = [
        [Paragraph("<b>Metric Name</b>", bold_body), Paragraph("<b>Mathematical Formula</b>", bold_body), Paragraph("<b>Business Meaning</b>", bold_body)],
        [Paragraph("MAE (Mean Absolute Error)", body_style), Paragraph("1/n ∑ |Actual - Predicted|", body_style), Paragraph("Average absolute dollar/rupee error per prediction.", body_style)],
        [Paragraph("RMSE (Root Mean Squared Error)", body_style), Paragraph("√( 1/n ∑ (Actual - Predicted)² )", body_style), Paragraph("Penalizes large forecast errors heavily.", body_style)],
        [Paragraph("MAPE (Mean Absolute % Error)", body_style), Paragraph("100%/n ∑ |(Actual - Pred)/Actual|", body_style), Paragraph("Percentage deviation relative to actual sales.", body_style)],
        [Paragraph("Accuracy Score (%)", body_style), Paragraph("max(0, 100% - MAPE)", body_style), Paragraph("Overall percentage model fit accuracy.", body_style)]
    ]

    t_metrics = Table(metrics_data, colWidths=[1.8*inch, 2.2*inch, 3.0*inch])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), light_bg),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 12))

    # Key Features
    story.append(Paragraph("5. Key Application Features", h1_style))
    features_text = (
        "1. <b>CSV Data Drag & Drop</b>: Auto-detects Date and Revenue columns with live preview snippet.<br/>"
        "2. <b>Resampling & Data Health</b>: Handles missing daily gaps via linear interpolation; resamples Daily, Weekly, Monthly.<br/>"
        "3. <b>Multi-Horizon Forecasting</b>: Project sales 7, 30, 90, or 180 days into the future.<br/>"
        "4. <b>95% Confidence Bounds</b>: Displays upper and lower risk bounds for inventory planning.<br/>"
        "5. <b>Indian Rupee (₹) Currency Support</b>: Formatted using Indian numbering locale (e.g. ₹1,45,719.30).<br/>"
        "6. <b>CSV Forecast Report Export</b>: Download generated predictions for business spreadsheet reporting."
    )
    story.append(Paragraph(features_text, body_style))
    story.append(Spacer(1, 10))

    # Project Structure
    story.append(Paragraph("6. Codebase File Structure", h1_style))
    struct_data = [
        [Paragraph("<b>File / Folder</b>", bold_body), Paragraph("<b>Functionality</b>", bold_body)],
        [Paragraph("app.py", body_style), Paragraph("Flask REST API routes (/api/upload, /api/forecast, /api/export-csv)", body_style)],
        [Paragraph("model.py", body_style), Paragraph("ARIMA model fitting, Auto-AIC tuning, backtest evaluation & confidence bounds", body_style)],
        [Paragraph("forecast.py", body_style), Paragraph("Data preprocessing, column auto-detection, date parsing, linear interpolation", body_style)],
        [Paragraph("templates/index.html", body_style), Paragraph("Glassmorphism dark theme dashboard user interface template", body_style)],
        [Paragraph("static/js/main.js", body_style), Paragraph("Chart.js integration, AJAX API handler, dynamic table & currency formatting", body_style)],
        [Paragraph("static/css/style.css", body_style), Paragraph("CSS glassmorphism styling, glowing cards, and responsive layout", body_style)]
    ]
    t_struct = Table(struct_data, colWidths=[2.2*inch, 4.8*inch])
    t_struct.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), light_bg),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_struct)
    story.append(Spacer(1, 14))

    # Presentation Guide / Questions Sir Might Ask
    story.append(Paragraph("7. Presentation & Q&A Guide for your Professor/Sir", h1_style))
    
    qa_list = [
        ("Q1: What is the main objective of your project?",
         "Answer: To develop a web application that takes historical sales CSV data and predicts future sales using Time Series ARIMA modeling to prevent stockouts and overstocking."),
        
        ("Q2: Why did you choose ARIMA over traditional linear regression?",
         "Answer: Standard linear regression assumes independent observations. Time series sales data has autocorrelation (today's sales depend on yesterday's sales) and trends. ARIMA specifically accounts for lagged values (AR) and lagged forecast errors (MA)."),
        
        ("Q3: How do you handle missing dates or gaps in sales data?",
         "Answer: In `forecast.py`, we resample the series to a fixed daily/weekly frequency and use linear interpolation (`interpolate(method='linear')`) followed by forward/backward fill."),
        
        ("Q4: How do you evaluate the model's accuracy?",
         "Answer: We perform a sequential 80-20 train-test split. We fit ARIMA on the training data, predict the test period, and compare against actual values using MAE, RMSE, and MAPE metrics."),
        
        ("Q5: What is the 95% Confidence Interval in your chart?",
         "Answer: It represents the statistical upper and lower uncertainty range within which future actual sales are 95% likely to fall, providing a safety threshold for inventory stock planning.")
    ]

    for q, a in qa_list:
        story.append(Paragraph(f"<b>{q}</b>", h2_style))
        story.append(Paragraph(a, body_style))
        story.append(Spacer(1, 3))

    doc.build(story)
    print(f"PDF successfully created at: {output_path}")

if __name__ == "__main__":
    out_pdf = os.path.join(os.path.dirname(__file__), "Sales_Forecasting_Project_Report.pdf")
    create_project_pdf(out_pdf)
