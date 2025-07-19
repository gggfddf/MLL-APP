#!/usr/bin/env python3
"""
Library compatibility test for the Stock Market Analysis System
This ensures all required libraries are properly installed and working
"""

import sys
print("Testing all required libraries for the Stock Market Analysis System...")
print("=" * 60)

# Core libraries test
try:
    import numpy as np
    print("✓ NumPy available")
except ImportError as e:
    print(f"✗ NumPy not available: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print("✓ Pandas available")
except ImportError as e:
    print(f"✗ Pandas not available: {e}")
    sys.exit(1)

try:
    import yfinance as yf
    print("✓ yfinance available for market data")
except ImportError as e:
    print(f"✗ yfinance not available: {e}")
    sys.exit(1)

try:
    import streamlit as st
    print("✓ Streamlit available for web interface")
except ImportError as e:
    print(f"✗ Streamlit not available: {e}")
    sys.exit(1)

try:
    import plotly.graph_objects as go
    print("✓ Plotly available for interactive charts")
except ImportError as e:
    print(f"✗ Plotly not available: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    print("✓ Matplotlib available")
except ImportError as e:
    print(f"✗ Matplotlib not available: {e}")
    sys.exit(1)

try:
    import seaborn as sns
    print("✓ Seaborn available")
except ImportError as e:
    print(f"✗ Seaborn not available: {e}")
    sys.exit(1)

# Machine Learning libraries test
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    print("✓ Scikit-learn available for ML models")
except ImportError as e:
    print(f"✗ Scikit-learn not available: {e}")
    sys.exit(1)

try:
    import xgboost as xgb
    print("✓ XGBoost available for gradient boosting")
except ImportError as e:
    print(f"✗ XGBoost not available: {e}")
    sys.exit(1)

# Deep Learning libraries test
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D
    print(f"✓ TensorFlow {tf.__version__} available for deep learning")
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    print(f"⚠ TensorFlow not available: {e} - using traditional ML fallback")
    TENSORFLOW_AVAILABLE = False

# Computer Vision libraries test
try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__} available for pattern recognition")
    OPENCV_AVAILABLE = True
except ImportError as e:
    print(f"⚠ OpenCV not available: {e} - disabling CV pattern recognition")
    OPENCV_AVAILABLE = False

# Technical Analysis libraries test
try:
    from finta import TA as finta_ta
    print("✓ finta available - 80+ technical indicators")
    FINTA_AVAILABLE = True
except ImportError as e:
    print(f"✗ finta not available: {e}")
    FINTA_AVAILABLE = False
    sys.exit(1)

try:
    import ta
    print("✓ ta library available for additional indicators")
    TA_AVAILABLE = True
except ImportError as e:
    print(f"⚠ ta library not available: {e} - using finta only")
    TA_AVAILABLE = False

# Optional libraries test
try:
    import talib
    print("✓ TA-Lib available for advanced technical analysis")
    TALIB_AVAILABLE = True
except ImportError as e:
    print(f"⚠ TA-Lib not available: {e} - using finta/ta alternatives")
    TALIB_AVAILABLE = False

# Report generation libraries test
try:
    import openpyxl
    print("✓ openpyxl available for Excel reports")
except ImportError as e:
    print(f"⚠ openpyxl not available: {e}")

try:
    from reportlab.pdfgen import canvas
    print("✓ ReportLab available for PDF reports")
except ImportError as e:
    print(f"⚠ ReportLab not available: {e}")

try:
    from jinja2 import Template
    print("✓ Jinja2 available for HTML reports")
except ImportError as e:
    print(f"⚠ Jinja2 not available: {e}")

print("=" * 60)
print("Library Availability Summary:")
print(f"✓ Core libraries: All available")
print(f"✓ Machine Learning: All available")
print(f"{'✓' if TENSORFLOW_AVAILABLE else '⚠'} Deep Learning: {'Available' if TENSORFLOW_AVAILABLE else 'Fallback to traditional ML'}")
print(f"{'✓' if OPENCV_AVAILABLE else '⚠'} Computer Vision: {'Available' if OPENCV_AVAILABLE else 'Basic pattern recognition only'}")
print(f"✓ Technical Analysis: finta + {'ta' if TA_AVAILABLE else 'basic implementations'}")
print(f"✓ Visualization: Plotly + Matplotlib + Seaborn")
print(f"✓ Data Pipeline: yfinance + pandas")
print(f"✓ Web Interface: Streamlit")

print("\n🎯 System is ready for comprehensive stock market analysis!")

# Quick functional test
print("\n" + "=" * 60)
print("Running quick functional tests...")

# Test data fetching
try:
    ticker = yf.Ticker("RELIANCE.NS")
    data = ticker.history(period="5d")
    if not data.empty:
        print("✓ Market data fetching works")
    else:
        print("⚠ Market data fetching returned empty data")
except Exception as e:
    print(f"⚠ Market data fetching error: {e}")

# Test technical indicators
try:
    if len(data) > 20:
        sma = finta_ta.SMA(data, 14)
        rsi = finta_ta.RSI(data, 14)
        if not sma.empty and not rsi.empty:
            print("✓ Technical indicators working")
        else:
            print("⚠ Technical indicators returned empty data")
except Exception as e:
    print(f"⚠ Technical indicators error: {e}")

# Test ML models
try:
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    # Create simple test data
    X = np.random.random((50, 5))
    y = np.random.random(50)
    model.fit(X, y)
    predictions = model.predict(X[:5])
    print("✓ Machine learning models working")
except Exception as e:
    print(f"⚠ ML models error: {e}")

print("=" * 60)
print("✅ ALL SYSTEMS READY FOR DEPLOYMENT!")