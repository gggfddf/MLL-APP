import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import json
import time
import io
import base64
from pathlib import Path
warnings.filterwarnings('ignore')

# Import custom modules
from models.deep_learning_engine import DeepLearningEngine
from models.pattern_discovery import PatternDiscoveryEngine
from models.technical_indicators import TechnicalIndicatorEngine
from models.prediction_engine import PredictionEngine
from data.data_pipeline import DataPipeline
from analysis.technical_analysis import TechnicalAnalyzer
from analysis.price_action_analysis import PriceActionAnalyzer
from visualization.chart_engine import ChartEngine
from reports.report_generator import ReportGenerator
from utils.market_utils import MarketUtils
from database.database_manager import DatabaseManager

# Page configuration
st.set_page_config(
    page_title="🚀 Advanced Stock Market Intelligence System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional trading interface
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #00ff88;
    }
    .prediction-box {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #00ff88;
    }
    .confidence-high { color: #00ff88; }
    .confidence-medium { color: #ffaa00; }
    .confidence-low { color: #ff4444; }
</style>
""", unsafe_allow_html=True)

def main():
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 ULTIMATE DEEP LEARNING STOCK MARKET ANALYSIS SYSTEM</h1>
        <p>Autonomous AI-Powered Market Intelligence for Indian Stocks</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for stock selection
    with st.sidebar:
        st.header("🎯 Stock Selection")
        
        # Stock symbol input (only editable parameter)
        stock_symbol = st.text_input(
            "Enter Stock Symbol (NSE Format)",
            value="RELIANCE.NS",
            help="Format: STOCKNAME.NS (e.g., RELIANCE.NS, TCS.NS, INFY.NS)",
            key=f"stock_input_{int(time.time())}"
        ).upper()
        
        # Analysis trigger
        analyze_button = st.button("🔍 Execute Deep Analysis", type="primary")
        
        st.markdown("---")
        st.markdown("""
        ### 🧠 System Features
        - **Deep Learning Engine**: LSTM + CNN + Transformers
        - **Autonomous Pattern Discovery**
        - **40+ Advanced Technical Indicators**
        - **ML-Powered Predictions**
        - **Real-time Market Intelligence**
        """)
    
    # Initialize system components
    if analyze_button and stock_symbol:
        with st.spinner("🚀 Initializing AI Market Intelligence System..."):
            try:
                # Initialize all system components
                data_pipeline = DataPipeline()
                deep_learning_engine = DeepLearningEngine()
                pattern_discovery = PatternDiscoveryEngine()
                technical_engine = TechnicalIndicatorEngine()
                prediction_engine = PredictionEngine()
                technical_analyzer = TechnicalAnalyzer()
                price_action_analyzer = PriceActionAnalyzer()
                chart_engine = ChartEngine()
                report_generator = ReportGenerator()
                market_utils = MarketUtils()
                db_manager = DatabaseManager()  # Database integration
                
                # Fetch and process data
                st.info("📊 Fetching live market data and historical analysis...")
                market_data = data_pipeline.fetch_comprehensive_data(stock_symbol)
                
                if market_data is None or market_data.empty:
                    st.error("❌ Unable to fetch data for the specified stock symbol. Please check the symbol format.")
                    return
                
                # Store market data in database
                st.info("💾 Storing data in database for analysis history...")
                db_manager.store_market_data(stock_symbol, market_data)
                
                # Display stock information
                stock_info = market_utils.get_stock_info(stock_symbol)
                display_stock_overview(stock_info, market_data)
                
                # Display database analytics
                display_database_analytics(db_manager, stock_symbol)
                
                # Main analysis tabs
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "🧠 Deep Learning Analysis",
                    "📊 Technical Analysis Report", 
                    "📈 Price Action Analysis Report",
                    "🎯 ML Predictions",
                    "📉 Advanced Charts",
                    "📋 Comprehensive Reports"
                ])
                
                with tab1:
                    display_deep_learning_analysis(deep_learning_engine, pattern_discovery, market_data)
                
                with tab2:
                    display_technical_analysis(technical_analyzer, technical_engine, market_data)
                
                with tab3:
                    display_price_action_analysis(price_action_analyzer, pattern_discovery, market_data)
                
                with tab4:
                    display_ml_predictions(prediction_engine, deep_learning_engine, market_data, db_manager, stock_symbol)
                
                with tab5:
                    display_advanced_charts(chart_engine, market_data, stock_symbol)
                
                with tab6:
                    display_comprehensive_reports(report_generator, market_data, stock_symbol)
                
            except Exception as e:
                st.error(f"❌ System Error: {str(e)}")
                st.info("Please try again or contact support if the issue persists.")
    else:
        # Display system overview when no analysis is running
        display_system_overview()

def display_system_overview():
    """Display system capabilities and overview"""
    st.markdown("""
    ## 🚀 Ultimate Deep Learning Stock Market Analysis System
    
    ### 🧠 Advanced AI Capabilities
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 🤖 Deep Learning Engine
        - **LSTM Networks**: Sequential pattern recognition
        - **CNN Layers**: Visual chart pattern analysis  
        - **Transformer Models**: Complex pattern relationships
        - **AutoEncoders**: Anomaly detection
        - **Ensemble Methods**: Multi-model predictions
        """)
    
    with col2:
        st.markdown("""
        #### 📊 Technical Analysis
        - **40+ Technical Indicators**
        - **Pattern Analysis Within Indicators**
        - **Multi-timeframe Analysis**
        - **Composite Signal Generation**
        - **Momentum & Volatility Analysis**
        """)
    
    with col3:
        st.markdown("""
        #### 📈 Price Action Analysis
        - **Autonomous Pattern Discovery**
        - **Candlestick Pattern Recognition**
        - **Support/Resistance Detection**
        - **Volume-Price Relationships**
        - **Market Structure Analysis**
        """)
    
    st.markdown("""
    ### 🎯 Key Features
    
    1. **Autonomous Learning**: System discovers patterns directly from data
    2. **Real-time Analysis**: Live Indian stock market data integration
    3. **ML Predictions**: Every analysis includes confidence-based predictions
    4. **Professional Charts**: Interactive candlestick visualizations
    5. **Comprehensive Reports**: Excel, PDF, JSON, and HTML outputs
    
    **Simply enter a stock symbol above and click 'Execute Deep Analysis' to begin!**
    """)

def display_stock_overview(stock_info, market_data):
    """Display stock overview and current metrics"""
    st.subheader("📊 Stock Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = market_data['Close'].iloc[-1]
    previous_close = market_data['Close'].iloc[-2]
    change = current_price - previous_close
    change_pct = (change / previous_close) * 100
    
    with col1:
        st.metric("Current Price", f"₹{current_price:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
    
    with col2:
        st.metric("Volume", f"{market_data['Volume'].iloc[-1]:,.0f}")
    
    with col3:
        st.metric("52W High", f"₹{market_data['High'].max():.2f}")
    
    with col4:
        st.metric("52W Low", f"₹{market_data['Low'].min():.2f}")

def display_database_analytics(db_manager, symbol):
    """Display database analytics and historical data"""
    st.subheader("💾 Database Analytics & Historical Intelligence")
    
    try:
        # Get analysis summary from database
        summary = db_manager.get_analysis_summary(symbol)
        
        if summary:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Sessions", summary.get('total_sessions', 0))
            
            with col2:
                patterns_count = len(summary.get('recent_patterns', []))
                st.metric("Patterns Found", patterns_count)
            
            with col3:
                if summary.get('latest_prediction'):
                    confidence = summary['latest_prediction'].get('confidence_level', 'Unknown')
                    st.metric("Latest Prediction", confidence)
                else:
                    st.metric("Latest Prediction", "None")
            
            with col4:
                if summary.get('latest_indicators'):
                    rsi = summary['latest_indicators'].get('rsi', 0)
                    st.metric("Latest RSI", f"{rsi:.1f}" if rsi else "N/A")
                else:
                    st.metric("Latest RSI", "N/A")
            
            # Display recent patterns
            if summary.get('recent_patterns'):
                with st.expander("🔍 Recent Patterns Discovered"):
                    patterns_df = pd.DataFrame(summary['recent_patterns'])
                    st.dataframe(patterns_df[['pattern_name', 'confidence', 'detection_date', 'strength']], 
                               use_container_width=True)
            
            # Historical data chart
            historical_data = db_manager.get_historical_data(symbol, days=30)
            if not historical_data.empty:
                with st.expander("📈 30-Day Historical Database Data"):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=historical_data.index,
                        y=historical_data['close_price'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#00ff88', width=2)
                    ))
                    fig.update_layout(
                        title="Historical Price Data from Database",
                        xaxis_title="Date",
                        yaxis_title="Price (₹)",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 No historical data found in database. This will be the first analysis session for this symbol.")
    
    except Exception as e:
        st.warning(f"⚠ Database analytics unavailable: {e}")
        st.info("Analysis will continue without database features.")

def display_deep_learning_analysis(deep_learning_engine, pattern_discovery, market_data):
    """Display comprehensive deep learning analysis"""
    st.subheader("🧠 Deep Learning Market Intelligence")
    
    with st.spinner("🤖 Training neural networks and discovering patterns..."):
        # Prepare data for deep learning
        features = deep_learning_engine.prepare_features(market_data)
        
        # Train ensemble models
        model_results = deep_learning_engine.train_ensemble_models(features, market_data)
        
        # Pattern discovery
        discovered_patterns = pattern_discovery.discover_autonomous_patterns(market_data)
        
        # Feature importance analysis
        feature_importance = deep_learning_engine.analyze_feature_importance(features, market_data)
        
        # Store patterns in database
        db_manager.store_discovered_patterns(stock_symbol, discovered_patterns)
    
    # Display model performance
    st.markdown("### 📈 Neural Network Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>LSTM Accuracy</h4>
            <h2 class="confidence-high">{:.1f}%</h2>
            <p>Sequential pattern recognition</p>
        </div>
        """.format(model_results.get('lstm_accuracy', 0) * 100), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>CNN Accuracy</h4>
            <h2 class="confidence-medium">{:.1f}%</h2>
            <p>Visual pattern analysis</p>
        </div>
        """.format(model_results.get('cnn_accuracy', 0) * 100), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>Ensemble Score</h4>
            <h2 class="confidence-high">{:.1f}%</h2>
            <p>Combined model confidence</p>
        </div>
        """.format(model_results.get('ensemble_accuracy', 0) * 100), unsafe_allow_html=True)
    
    # Discovered patterns
    st.markdown("### 🔍 Autonomous Pattern Discovery")
    
    if discovered_patterns:
        for pattern in discovered_patterns[:5]:  # Show top 5 patterns
            with st.expander(f"🎯 {pattern['name']} - Confidence: {pattern['confidence']:.1%}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Pattern Type**: {pattern['type']}")
                    st.write(f"**Frequency**: {pattern['frequency']} occurrences")
                    st.write(f"**Success Rate**: {pattern['success_rate']:.1%}")
                
                with col2:
                    st.write(f"**Timeframe**: {pattern['timeframe']}")
                    st.write(f"**Signal Strength**: {pattern['strength']}")
                    st.write(f"**Risk Level**: {pattern['risk_level']}")
    
    # Feature importance
    st.markdown("### 🎯 Feature Importance Analysis")
    
    if feature_importance:
        importance_df = pd.DataFrame(list(feature_importance.items()), 
                                   columns=['Feature', 'Importance'])
        importance_df = importance_df.sort_values('Importance', ascending=False).head(10)
        
        import plotly.express as px
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                     title="Top 10 Features Driving ML Predictions")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def display_technical_analysis(technical_analyzer, technical_engine, market_data):
    """Display comprehensive technical analysis report"""
    st.subheader("🔧 Technical Analysis Report")
    
    with st.spinner("📊 Analyzing 40+ technical indicators..."):
        # Generate technical analysis
        technical_results = technical_analyzer.generate_comprehensive_analysis(market_data)
        indicator_values = technical_engine.calculate_all_indicators(market_data)
        
    # Technical signals summary
    st.markdown("### 📊 Technical Signals Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    signals = technical_results.get('signals', {})
    
    with col1:
        bullish_signals = sum(1 for s in signals.values() if s == 'bullish')
        st.metric("Bullish Signals", bullish_signals, delta=None)
    
    with col2:
        bearish_signals = sum(1 for s in signals.values() if s == 'bearish')
        st.metric("Bearish Signals", bearish_signals, delta=None)
    
    with col3:
        neutral_signals = sum(1 for s in signals.values() if s == 'neutral')
        st.metric("Neutral Signals", neutral_signals, delta=None)
    
    with col4:
        overall_sentiment = technical_results.get('overall_sentiment', 'neutral')
        st.metric("Overall Sentiment", overall_sentiment.title())
    
    # Detailed indicator analysis
    st.markdown("### 📈 Detailed Indicator Analysis")
    
    # Create expandable sections for different indicator categories
    with st.expander("🎯 Momentum Indicators"):
        display_momentum_indicators(indicator_values)
    
    with st.expander("📊 Trend Indicators"):  
        display_trend_indicators(indicator_values)
    
    with st.expander("🌊 Volatility Indicators"):
        display_volatility_indicators(indicator_values)
    
    with st.expander("📶 Volume Indicators"):
        display_volume_indicators(indicator_values)

def display_price_action_analysis(price_action_analyzer, pattern_discovery, market_data):
    """Display comprehensive price action analysis report"""
    st.subheader("📈 Price Action Analysis Report")
    
    with st.spinner("🔍 Analyzing pure price action and patterns..."):
        # Generate price action analysis
        price_action_results = price_action_analyzer.analyze_comprehensive_price_action(market_data)
        candlestick_patterns = pattern_discovery.detect_candlestick_patterns(market_data)
        
    # Price action summary
    st.markdown("### 📊 Price Action Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        market_structure = price_action_results.get('market_structure', 'sideways')
        st.markdown(f"""
        <div class="metric-card">
            <h4>Market Structure</h4>
            <h3>{market_structure.title()}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        trend_strength = price_action_results.get('trend_strength', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h4>Trend Strength</h4>
            <h3>{trend_strength:.1f}/10</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        volume_confirmation = price_action_results.get('volume_confirmation', 'weak')
        st.markdown(f"""
        <div class="metric-card">
            <h4>Volume Confirmation</h4>
            <h3>{volume_confirmation.title()}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Support and resistance levels
    st.markdown("### 🎯 Support & Resistance Levels")
    
    sr_levels = price_action_results.get('support_resistance', {})
    
    if sr_levels:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🟢 Support Levels")
            for level in sr_levels.get('support', []):
                strength = "🔥" if level['strength'] > 0.8 else "⚡" if level['strength'] > 0.6 else "💫"
                st.write(f"{strength} ₹{level['price']:.2f} (Strength: {level['strength']:.1%})")
        
        with col2:
            st.markdown("#### 🔴 Resistance Levels")
            for level in sr_levels.get('resistance', []):
                strength = "🔥" if level['strength'] > 0.8 else "⚡" if level['strength'] > 0.6 else "💫"
                st.write(f"{strength} ₹{level['price']:.2f} (Strength: {level['strength']:.1%})")
    
    # Candlestick patterns
    st.markdown("### 🕯️ Detected Candlestick Patterns")
    
    if candlestick_patterns:
        for pattern in candlestick_patterns[-10:]:  # Show last 10 patterns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**{pattern['name']}**")
            
            with col2:
                st.write(f"Date: {pattern['date']}")
            
            with col3:
                confidence_class = "confidence-high" if pattern['confidence'] > 0.8 else "confidence-medium" if pattern['confidence'] > 0.6 else "confidence-low"
                st.markdown(f"<span class='{confidence_class}'>Confidence: {pattern['confidence']:.1%}</span>", unsafe_allow_html=True)

def display_ml_predictions(prediction_engine, deep_learning_engine, market_data, db_manager=None, stock_symbol=None):
    """Display ML-powered predictions with confidence scores"""
    st.subheader("🎯 Machine Learning Predictions")
    
    with st.spinner("🤖 Generating AI-powered market predictions..."):
        # Generate predictions
        predictions = prediction_engine.generate_ensemble_predictions(market_data)
        deep_predictions = deep_learning_engine.predict_future_movements(market_data)
        
        # Store predictions in database
        if db_manager and stock_symbol:
            db_manager.store_ml_predictions(stock_symbol, predictions)
        
    # Main prediction display
    st.markdown("### 🚀 Primary Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        direction_prob = predictions.get('direction_probability', 0.5)
        direction = "Upward" if direction_prob > 0.5 else "Downward"
        confidence_level = "High" if abs(direction_prob - 0.5) > 0.3 else "Medium" if abs(direction_prob - 0.5) > 0.15 else "Low"
        
        st.markdown(f"""
        <div class="prediction-box">
            <h3>Movement Direction</h3>
            <h2>{direction}</h2>
            <p>Probability: {direction_prob:.1%}</p>
            <p>Confidence: {confidence_level}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        target_price = predictions.get('target_price', market_data['Close'].iloc[-1])
        current_price = market_data['Close'].iloc[-1]
        price_change = ((target_price - current_price) / current_price) * 100
        
        st.markdown(f"""
        <div class="prediction-box">
            <h3>Target Price (1-5 days)</h3>
            <h2>₹{target_price:.2f}</h2>
            <p>Change: {price_change:+.2f}%</p>
            <p>Current: ₹{current_price:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed predictions
    st.markdown("### 📊 Detailed Prediction Analysis")
    
    pred_tabs = st.tabs(["🎯 Price Targets", "⏰ Time Horizons", "⚠️ Risk Assessment", "🔍 Scenario Analysis"])
    
    with pred_tabs[0]:
        st.markdown("#### Price Target Analysis")
        
        targets = predictions.get('price_targets', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Conservative Target", f"₹{targets.get('conservative', current_price):.2f}")
            st.metric("Probability", f"{targets.get('conservative_prob', 0):.1%}")
        
        with col2:
            st.metric("Moderate Target", f"₹{targets.get('moderate', current_price):.2f}")
            st.metric("Probability", f"{targets.get('moderate_prob', 0):.1%}")
        
        with col3:
            st.metric("Aggressive Target", f"₹{targets.get('aggressive', current_price):.2f}")
            st.metric("Probability", f"{targets.get('aggressive_prob', 0):.1%}")
    
    with pred_tabs[1]:
        st.markdown("#### Time Horizon Analysis")
        
        horizons = predictions.get('time_horizons', {})
        
        for horizon, data in horizons.items():
            with st.expander(f"📅 {horizon.title()} ({data.get('period', 'N/A')})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Expected Price**: ₹{data.get('price', current_price):.2f}")
                    st.write(f"**Confidence**: {data.get('confidence', 0):.1%}")
                
                with col2:
                    st.write(f"**Movement**: {data.get('direction', 'neutral').title()}")
                    st.write(f"**Volatility**: {data.get('volatility', 'medium').title()}")
    
    with pred_tabs[2]:
        st.markdown("#### Risk Assessment")
        
        risk_metrics = predictions.get('risk_assessment', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Level", risk_metrics.get('level', 'Medium'))
            st.metric("Stop Loss", f"₹{risk_metrics.get('stop_loss', current_price * 0.95):.2f}")
        
        with col2:
            st.metric("Max Drawdown Risk", f"{risk_metrics.get('max_drawdown', 0):.1%}")
            st.metric("Volatility Score", f"{risk_metrics.get('volatility_score', 5)}/10")
        
        with col3:
            st.metric("Risk-Reward Ratio", f"1:{risk_metrics.get('risk_reward_ratio', 1.5):.1f}")
            st.metric("Position Size Rec.", f"{risk_metrics.get('position_size', 1):.1%}")
    
    with pred_tabs[3]:
        st.markdown("#### Scenario Analysis")
        
        scenarios = predictions.get('scenarios', {})
        
        for scenario_name, scenario_data in scenarios.items():
            st.markdown(f"**{scenario_name.title()} Case**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"Price: ₹{scenario_data.get('price', current_price):.2f}")
            
            with col2:
                st.write(f"Probability: {scenario_data.get('probability', 0):.1%}")
            
            with col3:
                st.write(f"Timeframe: {scenario_data.get('timeframe', 'N/A')}")

def display_advanced_charts(chart_engine, market_data, stock_symbol):
    """Display advanced interactive charts"""
    st.subheader("📉 Professional Trading Charts")
    
    # Timeframe selection
    timeframe = st.selectbox("Select Timeframe", 
                           ["5min", "15min", "1day", "1week"], 
                           index=2)
    
    with st.spinner(f"📊 Generating professional {timeframe} charts..."):
        # Generate comprehensive chart
        fig = chart_engine.create_comprehensive_chart(market_data, 
                                                    title=f"{stock_symbol} - {timeframe.upper()} Analysis",
                                                    timeframe=timeframe)
        
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        
        # Additional chart types
        chart_tabs = st.tabs(["🕯️ Candlestick", "📊 Volume Analysis", "🎯 Pattern Detection", "📈 Indicator Analysis"])
        
        with chart_tabs[0]:
            candlestick_fig = chart_engine.create_candlestick_chart(market_data, stock_symbol)
            if candlestick_fig:
                st.plotly_chart(candlestick_fig, use_container_width=True)
        
        with chart_tabs[1]:
            volume_fig = chart_engine.create_volume_analysis_chart(market_data, stock_symbol)
            if volume_fig:
                st.plotly_chart(volume_fig, use_container_width=True)
        
        with chart_tabs[2]:
            pattern_fig = chart_engine.create_pattern_detection_chart(market_data, stock_symbol)
            if pattern_fig:
                st.plotly_chart(pattern_fig, use_container_width=True)
        
        with chart_tabs[3]:
            indicator_fig = chart_engine.create_indicator_analysis_chart(market_data, stock_symbol)
            if indicator_fig:
                st.plotly_chart(indicator_fig, use_container_width=True)

def display_comprehensive_reports(report_generator, market_data, stock_symbol):
    """Display and generate comprehensive reports"""
    st.subheader("📋 Comprehensive Analysis Reports")
    
    st.markdown("""
    Generate professional reports containing all analysis results, predictions, and insights.
    Reports include charts, detailed analysis, and actionable recommendations.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Generate Excel Report", type="primary"):
            with st.spinner("📊 Creating comprehensive Excel report..."):
                excel_buffer = report_generator.generate_excel_report(market_data, stock_symbol)
                if excel_buffer:
                    st.download_button(
                        label="⬇️ Download Excel Report",
                        data=excel_buffer,
                        file_name=f"{stock_symbol}_Analysis_Report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    st.success("✅ Excel report generated successfully!")
        
        if st.button("📄 Generate PDF Report"):
            with st.spinner("📄 Creating detailed PDF report..."):
                pdf_buffer = report_generator.generate_pdf_report(market_data, stock_symbol)
                if pdf_buffer:
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"{stock_symbol}_Analysis_Report.pdf",
                        mime="application/pdf"
                    )
                    st.success("✅ PDF report generated successfully!")
    
    with col2:
        if st.button("🌐 Generate HTML Report"):
            with st.spinner("🌐 Creating interactive HTML report..."):
                html_content = report_generator.generate_html_report(market_data, stock_symbol)
                if html_content:
                    st.download_button(
                        label="⬇️ Download HTML Report",
                        data=html_content.encode(),
                        file_name=f"{stock_symbol}_Analysis_Report.html",
                        mime="text/html"
                    )
                    st.success("✅ HTML report generated successfully!")
        
        if st.button("📁 Generate JSON Data"):
            with st.spinner("📁 Creating machine-readable JSON data..."):
                json_data = report_generator.generate_json_report(market_data, stock_symbol)
                if json_data:
                    st.download_button(
                        label="⬇️ Download JSON Data",
                        data=json.dumps(json_data, indent=2).encode(),
                        file_name=f"{stock_symbol}_Analysis_Data.json",
                        mime="application/json"
                    )
                    st.success("✅ JSON data exported successfully!")

# Helper functions for indicator displays
def display_momentum_indicators(indicators):
    """Display momentum indicators analysis"""
    st.markdown("#### RSI Analysis")
    rsi_value = indicators.get('RSI', 50)
    rsi_signal = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
    st.metric("RSI (14)", f"{rsi_value:.2f}", rsi_signal)
    
    st.markdown("#### Stochastic Analysis")
    stoch_k = indicators.get('STOCH_K', 50)
    stoch_d = indicators.get('STOCH_D', 50)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("%K", f"{stoch_k:.2f}")
    with col2:
        st.metric("%D", f"{stoch_d:.2f}")

def display_trend_indicators(indicators):
    """Display trend indicators analysis"""
    st.markdown("#### Moving Averages")
    sma_20 = indicators.get('SMA_20', 0)
    sma_50 = indicators.get('SMA_50', 0)
    ema_12 = indicators.get('EMA_12', 0)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("SMA 20", f"₹{sma_20:.2f}")
    with col2:
        st.metric("SMA 50", f"₹{sma_50:.2f}")
    with col3:
        st.metric("EMA 12", f"₹{ema_12:.2f}")

def display_volatility_indicators(indicators):
    """Display volatility indicators analysis"""
    st.markdown("#### Bollinger Bands")
    bb_upper = indicators.get('BB_UPPER', 0)
    bb_middle = indicators.get('BB_MIDDLE', 0)
    bb_lower = indicators.get('BB_LOWER', 0)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Upper Band", f"₹{bb_upper:.2f}")
    with col2:
        st.metric("Middle Band", f"₹{bb_middle:.2f}")
    with col3:
        st.metric("Lower Band", f"₹{bb_lower:.2f}")

def display_volume_indicators(indicators):
    """Display volume indicators analysis"""
    st.markdown("#### Volume Analysis")
    obv = indicators.get('OBV', 0)
    volume_sma = indicators.get('VOLUME_SMA', 0)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("OBV", f"{obv:,.0f}")
    with col2:
        st.metric("Volume SMA", f"{volume_sma:,.0f}")

if __name__ == "__main__":
    main()

def display_stock_overview(stock_info, market_data):
    """Display stock overview and current market status"""
    st.subheader("📊 Stock Overview & Market Status")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    current_price = market_data['Close'].iloc[-1]
    prev_close = market_data['Close'].iloc[-2]
    price_change = current_price - prev_close
    price_change_pct = (price_change / prev_close) * 100
    
    with col1:
        st.metric("Current Price", f"₹{current_price:.2f}", f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
    
    with col2:
        st.metric("Volume", f"{market_data['Volume'].iloc[-1]:,.0f}")
    
    with col3:
        high_52w = market_data['High'].rolling(252).max().iloc[-1]
        st.metric("52W High", f"₹{high_52w:.2f}")
    
    with col4:
        low_52w = market_data['Low'].rolling(252).min().iloc[-1]
        st.metric("52W Low", f"₹{low_52w:.2f}")
    
    with col5:
        volatility = market_data['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
        st.metric("Volatility (20D)", f"{volatility:.1f}%")

def display_deep_learning_analysis(deep_learning_engine, pattern_discovery, market_data):
    """Display deep learning analysis results"""
    st.subheader("🧠 Deep Learning Market Intelligence")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔮 Neural Network Ensemble Analysis")
        
        # Train and analyze with deep learning models
        with st.spinner("Training neural network ensemble..."):
            dl_results = deep_learning_engine.analyze_market(market_data)
            
        st.markdown("#### Model Performance Metrics")
        for model_name, metrics in dl_results['model_performance'].items():
            st.write(f"**{model_name}**: Accuracy {metrics['accuracy']:.2f}%, Loss: {metrics['loss']:.4f}")
        
        st.markdown("#### Feature Importance (Top 10)")
        feature_importance = dl_results['feature_importance']
        for feature, importance in feature_importance.items():
            st.write(f"• **{feature}**: {importance:.3f}")
    
    with col2:
        st.markdown("### 🔍 Autonomous Pattern Discovery")
        
        # Discover patterns autonomously
        with st.spinner("Discovering market patterns..."):
            patterns = pattern_discovery.discover_patterns(market_data)
        
        st.markdown("#### Newly Discovered Patterns")
        for pattern in patterns['discovered_patterns']:
            confidence_class = "confidence-high" if pattern['confidence'] > 0.7 else "confidence-medium" if pattern['confidence'] > 0.5 else "confidence-low"
            st.markdown(f"""
            <div class="metric-card">
                <strong>{pattern['name']}</strong><br>
                Confidence: <span class="{confidence_class}">{pattern['confidence']:.2f}</span><br>
                Occurrences: {pattern['occurrences']}<br>
                Success Rate: {pattern['success_rate']:.1f}%
            </div>
            """, unsafe_allow_html=True)

def display_technical_analysis(technical_analyzer, technical_engine, market_data):
    """Display comprehensive technical analysis report"""
    st.subheader("📊 Technical Analysis Intelligence Report")
    
    # Generate technical analysis
    with st.spinner("Computing 40+ advanced technical indicators..."):
        tech_analysis = technical_analyzer.generate_comprehensive_analysis(market_data)
        indicators = technical_engine.calculate_all_indicators(market_data)
    
    # Technical Summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📈 Trend Analysis")
        trend_strength = tech_analysis['trend_strength']
        trend_direction = tech_analysis['trend_direction']
        
        st.markdown(f"""
        <div class="metric-card">
            <strong>Primary Trend</strong>: {trend_direction}<br>
            <strong>Strength</strong>: {trend_strength}/10<br>
            <strong>Confidence</strong>: {tech_analysis['trend_confidence']:.1f}%
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Momentum Indicators")
        for indicator, value in tech_analysis['momentum_signals'].items():
            signal_class = "confidence-high" if value['signal'] == 'Strong Buy' else "confidence-medium" if 'Buy' in value['signal'] else "confidence-low"
            st.markdown(f"""
            <div class="metric-card">
                <strong>{indicator}</strong>: <span class="{signal_class}">{value['signal']}</span><br>
                Value: {value['value']:.2f}
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("### 🎯 Support & Resistance")
        levels = tech_analysis['support_resistance']
        st.markdown(f"""
        <div class="metric-card">
            <strong>Resistance Levels</strong>:<br>
            R1: ₹{levels['resistance'][0]:.2f}<br>
            R2: ₹{levels['resistance'][1]:.2f}<br>
            <strong>Support Levels</strong>:<br>
            S1: ₹{levels['support'][0]:.2f}<br>
            S2: ₹{levels['support'][1]:.2f}
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed indicator analysis
    st.markdown("### 📋 Comprehensive Indicator Analysis")
    
    # Create indicator summary table
    indicator_df = pd.DataFrame([
        {
            'Indicator': name,
            'Current Value': data['current_value'],
            'Signal': data['signal'],
            'Pattern': data['pattern_detected'],
            'Confidence': f"{data['confidence']:.1f}%"
        }
        for name, data in indicators.items()
    ])
    
    st.dataframe(indicator_df, use_container_width=True)

def display_price_action_analysis(price_action_analyzer, pattern_discovery, market_data):
    """Display comprehensive price action analysis report"""
    st.subheader("📈 Price Action Analysis Intelligence Report")
    
    # Generate price action analysis
    with st.spinner("Analyzing price action patterns and market structure..."):
        price_analysis = price_action_analyzer.analyze_price_action(market_data)
        candlestick_patterns = pattern_discovery.detect_candlestick_patterns(market_data)
        chart_patterns = pattern_discovery.detect_chart_patterns(market_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🕯️ Candlestick Pattern Analysis")
        
        st.markdown("#### Traditional Patterns Detected")
        for pattern in candlestick_patterns['traditional_patterns']:
            reliability_class = "confidence-high" if pattern['reliability'] > 70 else "confidence-medium" if pattern['reliability'] > 50 else "confidence-low"
            st.markdown(f"""
            <div class="metric-card">
                <strong>{pattern['name']}</strong><br>
                Type: {pattern['type']}<br>
                Reliability: <span class="{reliability_class}">{pattern['reliability']:.1f}%</span><br>
                Timeframe: {pattern['timeframe']}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### ML-Discovered Patterns")
        for pattern in candlestick_patterns['ml_discovered_patterns']:
            st.markdown(f"""
            <div class="metric-card">
                <strong>{pattern['custom_name']}</strong><br>
                Formation: {pattern['formation_type']}<br>
                Success Rate: {pattern['historical_success']:.1f}%<br>
                Occurrences: {pattern['total_occurrences']}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Chart Pattern Recognition")
        
        for pattern in chart_patterns['detected_patterns']:
            completion_class = "confidence-high" if pattern['completion_probability'] > 0.7 else "confidence-medium" if pattern['completion_probability'] > 0.5 else "confidence-low"
            st.markdown(f"""
            <div class="metric-card">
                <strong>{pattern['pattern_type']}</strong><br>
                Stage: {pattern['formation_stage']}<br>
                Completion Probability: <span class="{completion_class}">{pattern['completion_probability']:.2f}</span><br>
                Breakout Direction: {pattern['expected_breakout']}<br>
                Target: ₹{pattern['price_target']:.2f}
            </div>
            """, unsafe_allow_html=True)
    
    # Volume-Price Analysis
    st.markdown("### 📊 Volume-Price Relationship Analysis")
    
    volume_analysis = price_analysis['volume_price_analysis']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <strong>Volume Trend</strong>: {volume_analysis['volume_trend']}<br>
            <strong>Price-Volume Correlation</strong>: {volume_analysis['correlation']:.3f}<br>
            <strong>Accumulation/Distribution</strong>: {volume_analysis['accumulation_distribution']}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <strong>Volume Breakouts</strong>: {volume_analysis['volume_breakouts']}<br>
            <strong>Average Volume (20D)</strong>: {volume_analysis['avg_volume_20d']:,.0f}<br>
            <strong>Volume Ratio</strong>: {volume_analysis['volume_ratio']:.2f}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <strong>Volume Confirmation</strong>: {volume_analysis['volume_confirmation']}<br>
            <strong>Buying Pressure</strong>: {volume_analysis['buying_pressure']:.1f}%<br>
            <strong>Selling Pressure</strong>: {volume_analysis['selling_pressure']:.1f}%
        </div>
        """, unsafe_allow_html=True)

def display_ml_predictions(prediction_engine, market_data):
    """Display ML-powered predictions with confidence scores"""
    st.subheader("🎯 Machine Learning Predictions & Intelligence")
    
    # Generate predictions
    with st.spinner("Generating ML-powered market predictions..."):
        predictions = prediction_engine.generate_predictions(market_data)
    
    # Main prediction display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔮 Primary Market Prediction")
        
        main_prediction = predictions['primary_prediction']
        confidence_class = "confidence-high" if main_prediction['confidence'] > 0.75 else "confidence-medium" if main_prediction['confidence'] > 0.5 else "confidence-low"
        
        st.markdown(f"""
        <div class="prediction-box">
            <h3>Direction: {main_prediction['direction']}</h3>
            <p><strong>Probability</strong>: <span class="{confidence_class}">{main_prediction['probability']:.1f}%</span></p>
            <p><strong>Target Price Range</strong>: ₹{main_prediction['target_range']['low']:.2f} - ₹{main_prediction['target_range']['high']:.2f}</p>
            <p><strong>Time Horizon</strong>: {main_prediction['time_horizon']}</p>
            <p><strong>Confidence Level</strong>: <span class="{confidence_class}">{main_prediction['confidence_level']} ({main_prediction['confidence']:.1f}%)</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature importance for prediction
        st.markdown("### 🔍 Prediction Feature Importance")
        feature_importance = predictions['feature_importance']
        
        importance_df = pd.DataFrame([
            {'Feature': feature, 'Importance': f"{importance:.3f}", 'Contribution': f"{importance*100:.1f}%"}
            for feature, importance in feature_importance.items()
        ])
        st.dataframe(importance_df, use_container_width=True)
    
    with col2:
        st.markdown("### ⚠️ Risk Assessment")
        
        risk_assessment = predictions['risk_assessment']
        risk_class = "confidence-low" if risk_assessment['risk_level'] == 'High' else "confidence-medium" if risk_assessment['risk_level'] == 'Medium' else "confidence-high"
        
        st.markdown(f"""
        <div class="metric-card">
            <strong>Risk Level</strong>: <span class="{risk_class}">{risk_assessment['risk_level']}</span><br>
            <strong>Volatility Risk</strong>: {risk_assessment['volatility_risk']:.1f}%<br>
            <strong>Stop Loss</strong>: ₹{risk_assessment['stop_loss']:.2f}<br>
            <strong>Risk-Reward Ratio</strong>: {risk_assessment['risk_reward_ratio']:.2f}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Scenario Analysis")
        scenarios = predictions['scenario_analysis']
        
        for scenario_name, scenario_data in scenarios.items():
            st.markdown(f"""
            <div class="metric-card">
                <strong>{scenario_name}</strong><br>
                Price: ₹{scenario_data['price']:.2f}<br>
                Probability: {scenario_data['probability']:.1f}%
            </div>
            """, unsafe_allow_html=True)
    
    # Pattern trigger conditions
    st.markdown("### 🎯 Pattern Trigger Conditions")
    
    trigger_conditions = predictions['trigger_conditions']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Entry Conditions")
        for condition in trigger_conditions['entry_conditions']:
            st.write(f"• {condition}")
    
    with col2:
        st.markdown("#### Confirmation Signals")
        for signal in trigger_conditions['confirmation_signals']:
            st.write(f"• {signal}")
    
    with col3:
        st.markdown("#### Exit Conditions")
        for condition in trigger_conditions['exit_conditions']:
            st.write(f"• {condition}")

def display_advanced_charts(chart_engine, market_data):
    """Display professional candlestick charts with advanced features"""
    st.subheader("📉 Professional Candlestick Chart Analysis")
    
    # Generate advanced charts
    with st.spinner("Generating professional trading charts..."):
        charts = chart_engine.generate_comprehensive_charts(market_data)
    
    # Multi-timeframe analysis
    timeframes = ['5T', '15T', '1D', '1W']
    timeframe_names = ['5-Minute', '15-Minute', 'Daily', 'Weekly']
    
    selected_timeframe = st.selectbox(
        "Select Timeframe Analysis",
        options=timeframe_names,
        index=2  # Default to Daily
    )
    
    timeframe_key = timeframes[timeframe_names.index(selected_timeframe)]
    
    # Main candlestick chart with overlays
    fig = charts['candlestick_charts'][timeframe_key]
    st.plotly_chart(fig, use_container_width=True)
    
    # Technical indicator charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 RSI & MACD Analysis")
        rsi_macd_fig = charts['indicator_charts']['rsi_macd']
        st.plotly_chart(rsi_macd_fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Bollinger Bands & Volume")
        bb_volume_fig = charts['indicator_charts']['bollinger_volume']
        st.plotly_chart(bb_volume_fig, use_container_width=True)
    
    # Pattern detection overlays
    st.markdown("### 🔍 Pattern Detection Visualization")
    pattern_fig = charts['pattern_charts']['detected_patterns']
    st.plotly_chart(pattern_fig, use_container_width=True)

def display_comprehensive_reports(report_generator, market_data, stock_symbol):
    """Display and generate comprehensive reports"""
    st.subheader("📋 Comprehensive Analysis Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Excel Report", type="primary"):
            with st.spinner("Generating comprehensive Excel report..."):
                excel_file = report_generator.generate_excel_report(market_data, stock_symbol)
                
                with open(excel_file, "rb") as file:
                    st.download_button(
                        label="Download Excel Report",
                        data=file.read(),
                        file_name=f"{stock_symbol}_analysis_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
    
    with col2:
        if st.button("📄 Generate PDF Reports", type="primary"):
            with st.spinner("Generating PDF analysis reports..."):
                pdf_files = report_generator.generate_pdf_reports(market_data, stock_symbol)
                
                for report_name, pdf_file in pdf_files.items():
                    with open(pdf_file, "rb") as file:
                        st.download_button(
                            label=f"Download {report_name} PDF",
                            data=file.read(),
                            file_name=f"{stock_symbol}_{report_name.lower().replace(' ', '_')}_report.pdf",
                            mime="application/pdf"
                        )
    
    with col3:
        if st.button("📁 Generate JSON Export", type="primary"):
            with st.spinner("Generating JSON data export..."):
                json_file = report_generator.generate_json_export(market_data, stock_symbol)
                
                with open(json_file, "rb") as file:
                    st.download_button(
                        label="Download JSON Export",
                        data=file.read(),
                        file_name=f"{stock_symbol}_data_export.json",
                        mime="application/json"
                    )
    
    # Executive Summary
    st.markdown("### 📋 Executive Summary")
    
    summary = report_generator.generate_executive_summary(market_data, stock_symbol)
    
    st.markdown(f"""
    <div class="prediction-box">
        <h4>Key Insights & Recommendations</h4>
        <p><strong>Overall Rating</strong>: {summary['overall_rating']}</p>
        <p><strong>Primary Recommendation</strong>: {summary['primary_recommendation']}</p>
        <p><strong>Risk Level</strong>: {summary['risk_level']}</p>
        <p><strong>Time Horizon</strong>: {summary['time_horizon']}</p>
        
        <h5>Key Findings:</h5>
        <ul>
            {''.join([f'<li>{finding}</li>' for finding in summary['key_findings']])}
        </ul>
        
        <h5>Action Items:</h5>
        <ul>
            {''.join([f'<li>{action}</li>' for action in summary['action_items']])}
        </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
