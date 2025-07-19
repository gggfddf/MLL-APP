import os
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import pandas as pd
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DatabaseManager:
    """
    Comprehensive Database Manager for Stock Market Analysis System
    Handles storage of market data, analysis results, patterns, and predictions
    """
    
    def __init__(self):
        self.connection = None
        self.setup_connection()
        self.create_tables()
    
    def setup_connection(self):
        """Setup PostgreSQL database connection"""
        try:
            self.connection = psycopg2.connect(
                host=os.environ.get('PGHOST', 'localhost'),
                database=os.environ.get('PGDATABASE', 'stock_analysis'),
                user=os.environ.get('PGUSER', 'postgres'),
                password=os.environ.get('PGPASSWORD', ''),
                port=os.environ.get('PGPORT', 5432)
            )
            self.connection.autocommit = True
            print("✓ Database connection established")
        except Exception as e:
            print(f"⚠ Database connection failed: {e}")
            self.connection = None
    
    def create_tables(self):
        """Create all necessary tables for the stock analysis system"""
        if not self.connection:
            return
        
        tables = {
            'stock_data': """
                CREATE TABLE IF NOT EXISTS stock_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    date DATE NOT NULL,
                    open_price DECIMAL(12,4),
                    high_price DECIMAL(12,4),
                    low_price DECIMAL(12,4),
                    close_price DECIMAL(12,4),
                    volume BIGINT,
                    returns DECIMAL(8,6),
                    volatility DECIMAL(8,6),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                );
            """,
            
            'technical_indicators': """
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    date DATE NOT NULL,
                    rsi DECIMAL(8,4),
                    macd DECIMAL(8,4),
                    macd_signal DECIMAL(8,4),
                    bb_upper DECIMAL(12,4),
                    bb_middle DECIMAL(12,4),
                    bb_lower DECIMAL(12,4),
                    sma_20 DECIMAL(12,4),
                    ema_12 DECIMAL(12,4),
                    stochastic_k DECIMAL(8,4),
                    williams_r DECIMAL(8,4),
                    atr DECIMAL(8,4),
                    adx DECIMAL(8,4),
                    cci DECIMAL(8,4),
                    obv BIGINT,
                    indicator_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                );
            """,
            
            'discovered_patterns': """
                CREATE TABLE IF NOT EXISTS discovered_patterns (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    pattern_name VARCHAR(100) NOT NULL,
                    pattern_type VARCHAR(50) NOT NULL,
                    detection_date DATE NOT NULL,
                    confidence DECIMAL(4,3),
                    success_rate DECIMAL(4,3),
                    frequency INTEGER,
                    timeframe VARCHAR(20),
                    strength VARCHAR(20),
                    risk_level VARCHAR(20),
                    pattern_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """,
            
            'ml_predictions': """
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    prediction_date DATE NOT NULL,
                    prediction_horizon INTEGER,
                    predicted_price DECIMAL(12,4),
                    direction_probability DECIMAL(4,3),
                    confidence_level VARCHAR(20),
                    model_type VARCHAR(50),
                    feature_importance JSONB,
                    prediction_data JSONB,
                    actual_price DECIMAL(12,4),
                    prediction_accuracy DECIMAL(4,3),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """,
            
            'analysis_sessions': """
                CREATE TABLE IF NOT EXISTS analysis_sessions (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    analysis_type VARCHAR(50),
                    data_points INTEGER,
                    patterns_found INTEGER,
                    indicators_calculated INTEGER,
                    ml_accuracy DECIMAL(4,3),
                    session_data JSONB,
                    execution_time DECIMAL(8,3)
                );
            """,
            
            'portfolio_tracking': """
                CREATE TABLE IF NOT EXISTS portfolio_tracking (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    position_type VARCHAR(10) CHECK (position_type IN ('long', 'short', 'watch')),
                    entry_price DECIMAL(12,4),
                    entry_date DATE,
                    quantity INTEGER,
                    stop_loss DECIMAL(12,4),
                    target_price DECIMAL(12,4),
                    current_price DECIMAL(12,4),
                    pnl DECIMAL(12,4),
                    roi_percentage DECIMAL(8,4),
                    status VARCHAR(20) DEFAULT 'active',
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """,
            
            'market_alerts': """
                CREATE TABLE IF NOT EXISTS market_alerts (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    alert_type VARCHAR(50) NOT NULL,
                    condition_type VARCHAR(50),
                    trigger_value DECIMAL(12,4),
                    current_value DECIMAL(12,4),
                    alert_message TEXT,
                    is_triggered BOOLEAN DEFAULT FALSE,
                    trigger_date TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """
        }
        
        try:
            cursor = self.connection.cursor()
            for table_name, create_sql in tables.items():
                cursor.execute(create_sql)
                print(f"✓ Table '{table_name}' ready")
            
            # Create indexes for better performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_stock_data_symbol_date ON stock_data(symbol, date);",
                "CREATE INDEX IF NOT EXISTS idx_indicators_symbol_date ON technical_indicators(symbol, date);",
                "CREATE INDEX IF NOT EXISTS idx_patterns_symbol ON discovered_patterns(symbol);",
                "CREATE INDEX IF NOT EXISTS idx_predictions_symbol_date ON ml_predictions(symbol, prediction_date);",
                "CREATE INDEX IF NOT EXISTS idx_sessions_symbol ON analysis_sessions(symbol);",
                "CREATE INDEX IF NOT EXISTS idx_portfolio_symbol ON portfolio_tracking(symbol);"
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            cursor.close()
            print("✓ Database schema created successfully")
            
        except Exception as e:
            print(f"⚠ Error creating tables: {e}")
    
    def store_market_data(self, symbol, data_df):
        """Store market data in the database"""
        if not self.connection or data_df.empty:
            return False
        
        try:
            cursor = self.connection.cursor()
            
            for index, row in data_df.iterrows():
                cursor.execute("""
                    INSERT INTO stock_data (symbol, date, open_price, high_price, low_price, 
                                          close_price, volume, returns, volatility)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, date) DO UPDATE SET
                        open_price = EXCLUDED.open_price,
                        high_price = EXCLUDED.high_price,
                        low_price = EXCLUDED.low_price,
                        close_price = EXCLUDED.close_price,
                        volume = EXCLUDED.volume,
                        returns = EXCLUDED.returns,
                        volatility = EXCLUDED.volatility
                """, (
                    symbol,
                    index.date() if hasattr(index, 'date') else index,
                    float(row.get('Open', 0)),
                    float(row.get('High', 0)),
                    float(row.get('Low', 0)),
                    float(row.get('Close', 0)),
                    int(row.get('Volume', 0)),
                    float(row.get('Returns', 0)) if pd.notna(row.get('Returns', 0)) else None,
                    float(row.get('Volatility', 0)) if pd.notna(row.get('Volatility', 0)) else None
                ))
            
            cursor.close()
            print(f"✓ Stored {len(data_df)} market data records for {symbol}")
            return True
            
        except Exception as e:
            print(f"⚠ Error storing market data: {e}")
            return False
    
    def store_technical_indicators(self, symbol, date, indicators):
        """Store technical indicators in the database"""
        if not self.connection:
            return False
        
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT INTO technical_indicators (
                    symbol, date, rsi, macd, macd_signal, bb_upper, bb_middle, bb_lower,
                    sma_20, ema_12, stochastic_k, williams_r, atr, adx, cci, obv, indicator_data
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, date) DO UPDATE SET
                    rsi = EXCLUDED.rsi,
                    macd = EXCLUDED.macd,
                    macd_signal = EXCLUDED.macd_signal,
                    bb_upper = EXCLUDED.bb_upper,
                    bb_middle = EXCLUDED.bb_middle,
                    bb_lower = EXCLUDED.bb_lower,
                    sma_20 = EXCLUDED.sma_20,
                    ema_12 = EXCLUDED.ema_12,
                    stochastic_k = EXCLUDED.stochastic_k,
                    williams_r = EXCLUDED.williams_r,
                    atr = EXCLUDED.atr,
                    adx = EXCLUDED.adx,
                    cci = EXCLUDED.cci,
                    obv = EXCLUDED.obv,
                    indicator_data = EXCLUDED.indicator_data
            """, (
                symbol,
                date,
                indicators.get('RSI'),
                indicators.get('MACD'),
                indicators.get('MACD_SIGNAL'),
                indicators.get('BB_UPPER'),
                indicators.get('BB_MIDDLE'),
                indicators.get('BB_LOWER'),
                indicators.get('SMA_20'),
                indicators.get('EMA_12'),
                indicators.get('STOCHASTIC_K'),
                indicators.get('WILLIAMS_R'),
                indicators.get('ATR'),
                indicators.get('ADX'),
                indicators.get('CCI'),
                indicators.get('OBV'),
                Json(indicators)  # Store full indicator data as JSON
            ))
            
            cursor.close()
            return True
            
        except Exception as e:
            print(f"⚠ Error storing technical indicators: {e}")
            return False
    
    def store_discovered_patterns(self, symbol, patterns):
        """Store discovered patterns in the database"""
        if not self.connection or not patterns:
            return False
        
        try:
            cursor = self.connection.cursor()
            
            for pattern in patterns:
                cursor.execute("""
                    INSERT INTO discovered_patterns (
                        symbol, pattern_name, pattern_type, detection_date, confidence,
                        success_rate, frequency, timeframe, strength, risk_level, pattern_data
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    symbol,
                    pattern.get('name'),
                    pattern.get('type'),
                    datetime.now().date(),
                    pattern.get('confidence'),
                    pattern.get('success_rate'),
                    pattern.get('frequency'),
                    pattern.get('timeframe'),
                    pattern.get('strength'),
                    pattern.get('risk_level'),
                    Json(pattern)
                ))
            
            cursor.close()
            print(f"✓ Stored {len(patterns)} patterns for {symbol}")
            return True
            
        except Exception as e:
            print(f"⚠ Error storing patterns: {e}")
            return False
    
    def store_ml_predictions(self, symbol, predictions):
        """Store ML predictions in the database"""
        if not self.connection:
            return False
        
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT INTO ml_predictions (
                    symbol, prediction_date, prediction_horizon, predicted_price,
                    direction_probability, confidence_level, model_type, feature_importance, prediction_data
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                symbol,
                datetime.now().date(),
                predictions.get('horizon_days', 5),
                predictions.get('target_price'),
                predictions.get('direction_probability'),
                predictions.get('confidence_level'),
                'ensemble_ml',
                Json(predictions.get('feature_importance', {})),
                Json(predictions)
            ))
            
            cursor.close()
            return True
            
        except Exception as e:
            print(f"⚠ Error storing predictions: {e}")
            return False
    
    def get_historical_data(self, symbol, days=30):
        """Retrieve historical data from database"""
        if not self.connection:
            return pd.DataFrame()
        
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM stock_data 
                WHERE symbol = %s AND date >= %s 
                ORDER BY date DESC
            """, (symbol, datetime.now().date() - timedelta(days=days)))
            
            rows = cursor.fetchall()
            cursor.close()
            
            if rows:
                df = pd.DataFrame(rows)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"⚠ Error retrieving historical data: {e}")
            return pd.DataFrame()
    
    def get_analysis_summary(self, symbol):
        """Get comprehensive analysis summary for a symbol"""
        if not self.connection:
            return {}
        
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            # Get latest indicators
            cursor.execute("""
                SELECT * FROM technical_indicators 
                WHERE symbol = %s 
                ORDER BY date DESC LIMIT 1
            """, (symbol,))
            latest_indicators = cursor.fetchone()
            
            # Get recent patterns
            cursor.execute("""
                SELECT * FROM discovered_patterns 
                WHERE symbol = %s 
                ORDER BY detection_date DESC LIMIT 5
            """, (symbol,))
            recent_patterns = cursor.fetchall()
            
            # Get latest prediction
            cursor.execute("""
                SELECT * FROM ml_predictions 
                WHERE symbol = %s 
                ORDER BY prediction_date DESC LIMIT 1
            """, (symbol,))
            latest_prediction = cursor.fetchone()
            
            # Get analysis sessions count
            cursor.execute("""
                SELECT COUNT(*) as session_count FROM analysis_sessions 
                WHERE symbol = %s
            """, (symbol,))
            session_count = cursor.fetchone()
            
            cursor.close()
            
            return {
                'latest_indicators': dict(latest_indicators) if latest_indicators else {},
                'recent_patterns': [dict(p) for p in recent_patterns],
                'latest_prediction': dict(latest_prediction) if latest_prediction else {},
                'total_sessions': session_count['session_count'] if session_count else 0
            }
            
        except Exception as e:
            print(f"⚠ Error retrieving analysis summary: {e}")
            return {}
    
    def log_analysis_session(self, symbol, session_data):
        """Log an analysis session"""
        if not self.connection:
            return False
        
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                INSERT INTO analysis_sessions (
                    symbol, analysis_type, data_points, patterns_found,
                    indicators_calculated, ml_accuracy, session_data, execution_time
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                symbol,
                session_data.get('analysis_type', 'comprehensive'),
                session_data.get('data_points', 0),
                session_data.get('patterns_found', 0),
                session_data.get('indicators_calculated', 0),
                session_data.get('ml_accuracy', 0),
                Json(session_data),
                session_data.get('execution_time', 0)
            ))
            
            cursor.close()
            return True
            
        except Exception as e:
            print(f"⚠ Error logging session: {e}")
            return False
    
    def close_connection(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("✓ Database connection closed")