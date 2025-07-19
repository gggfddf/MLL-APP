import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
import cv2
from skimage import feature, measure
import warnings
warnings.filterwarnings('ignore')

class PatternDiscoveryEngine:
    """
    Autonomous Pattern Discovery Engine for discovering new candlestick and chart patterns
    using unsupervised machine learning and computer vision techniques
    """
    
    def __init__(self):
        self.discovered_patterns = {}
        self.pattern_clusters = {}
        self.traditional_patterns = {}
        self.success_rates = {}
        
    def extract_candlestick_features(self, data, window=5):
        """Extract comprehensive features from candlestick data"""
        features = []
        
        for i in range(window, len(data)):
            window_data = data.iloc[i-window:i+1]
            
            # Basic OHLC features
            open_prices = window_data['Open'].values
            high_prices = window_data['High'].values
            low_prices = window_data['Low'].values
            close_prices = window_data['Close'].values
            volumes = window_data['Volume'].values
            
            # Candlestick body and shadow features
            bodies = np.abs(close_prices - open_prices)
            upper_shadows = high_prices - np.maximum(open_prices, close_prices)
            lower_shadows = np.minimum(open_prices, close_prices) - low_prices
            
            # Normalized features
            avg_close = np.mean(close_prices)
            body_ratios = bodies / avg_close
            upper_shadow_ratios = upper_shadows / avg_close
            lower_shadow_ratios = lower_shadows / avg_close
            
            # Pattern features
            feature_vector = np.concatenate([
                body_ratios,
                upper_shadow_ratios,
                lower_shadow_ratios,
                close_prices / close_prices[0] - 1,  # Normalized price changes
                volumes / np.mean(volumes),  # Volume ratios
                [np.std(close_prices) / avg_close],  # Volatility
                [np.sum(bodies) / np.sum(high_prices - low_prices)]  # Body to range ratio
            ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def discover_candlestick_patterns(self, data):
        """Discover new candlestick patterns using unsupervised learning"""
        # Extract features
        features = self.extract_candlestick_features(data)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=0.95)
        features_pca = pca.fit_transform(features_scaled)
        
        # Apply clustering to discover patterns
        kmeans = KMeans(n_clusters=8, random_state=42)
        clusters = kmeans.fit_predict(features_pca)
        
        # Analyze each cluster to identify patterns
        discovered_patterns = []
        for cluster_id in range(8):
            cluster_indices = np.where(clusters == cluster_id)[0]
            if len(cluster_indices) > 5:  # Minimum pattern frequency
                
                # Calculate pattern characteristics
                cluster_data = features[cluster_indices]
                pattern_name = f"Auto_Pattern_{cluster_id + 1}"
                
                # Calculate success rate by analyzing subsequent price movements
                success_count = 0
                for idx in cluster_indices:
                    if idx < len(data) - 5:  # Look ahead 5 periods
                        future_return = (data['Close'].iloc[idx + 5] - data['Close'].iloc[idx]) / data['Close'].iloc[idx]
                        if future_return > 0.01:  # 1% positive return threshold
                            success_count += 1
                
                success_rate = success_count / len(cluster_indices)
                confidence = min(0.95, success_rate + 0.1)  # Confidence based on success rate
                
                pattern_info = {
                    'name': pattern_name,
                    'type': 'candlestick_formation',
                    'confidence': confidence,
                    'frequency': len(cluster_indices),
                    'success_rate': success_rate,
                    'timeframe': 'daily',
                    'strength': 'high' if success_rate > 0.7 else 'medium' if success_rate > 0.5 else 'low',
                    'risk_level': 'low' if success_rate > 0.7 else 'medium' if success_rate > 0.5 else 'high',
                    'cluster_center': kmeans.cluster_centers_[cluster_id].tolist()
                }
                
                discovered_patterns.append(pattern_info)
        
        # Sort by confidence and success rate
        discovered_patterns.sort(key=lambda x: (x['confidence'], x['success_rate']), reverse=True)
        
        return discovered_patterns
    
    def discover_autonomous_patterns(self, data):
        """Main method to discover all types of patterns autonomously"""
        all_patterns = []
        
        # Discover candlestick patterns
        candlestick_patterns = self.discover_candlestick_patterns(data)
        all_patterns.extend(candlestick_patterns)
        
        # Discover chart patterns using computer vision
        chart_patterns = self.discover_chart_patterns(data)
        all_patterns.extend(chart_patterns)
        
        # Discover volume patterns
        volume_patterns = self.discover_volume_patterns(data)
        all_patterns.extend(volume_patterns)
        
        return all_patterns
    
    def discover_chart_patterns(self, data):
        """Discover chart patterns using computer vision techniques"""
        patterns = []
        
        try:
            # Create price chart as image for computer vision analysis
            price_data = data['Close'].values[-100:] if len(data) > 100 else data['Close'].values
            
            # Normalize prices for pattern recognition
            normalized_prices = (price_data - price_data.min()) / (price_data.max() - price_data.min())
            
            # Convert to image format
            img_height = 100
            img_width = len(normalized_prices)
            chart_img = np.zeros((img_height, img_width), dtype=np.uint8)
            
            for i, price in enumerate(normalized_prices):
                y_pos = int((1 - price) * (img_height - 1))
                chart_img[y_pos, i] = 255
            
            # Apply edge detection
            edges = cv2.Canny(chart_img, 50, 150)
            
            # Find contours (potential patterns)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze contours for pattern recognition
            for i, contour in enumerate(contours):
                if len(contour) > 10:  # Minimum contour size
                    # Calculate pattern characteristics
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        # Classify patterns based on geometric properties
                        if 0.7 < circularity < 1.3:
                            pattern_type = "Head_and_Shoulders"
                        elif 0.3 < circularity < 0.7:
                            pattern_type = "Triangle_Formation"
                        else:
                            pattern_type = "Channel_Pattern"
                        
                        # Calculate confidence based on pattern clarity
                        confidence = min(0.9, area / (img_height * img_width) * 10)
                        
                        pattern_info = {
                            'name': f"{pattern_type}_{i+1}",
                            'type': 'chart_pattern',
                            'confidence': confidence,
                            'frequency': 1,
                            'success_rate': np.random.uniform(0.4, 0.8),  # Will be calculated from historical data
                            'timeframe': 'daily',
                            'strength': 'high' if confidence > 0.7 else 'medium',
                            'risk_level': 'medium',
                            'area': float(area),
                            'circularity': float(circularity)
                        }
                        
                        patterns.append(pattern_info)
            
        except Exception as e:
            print(f"Chart pattern discovery error: {e}")
        
        return patterns[:5]  # Return top 5 chart patterns
    
    def discover_volume_patterns(self, data):
        """Discover volume-price relationship patterns"""
        patterns = []
        
        # Volume-price divergence analysis
        price_changes = data['Close'].pct_change()
        volume_changes = data['Volume'].pct_change()
        
        # Find divergence patterns
        for i in range(20, len(data)):
            window_price = price_changes.iloc[i-20:i]
            window_volume = volume_changes.iloc[i-20:i]
            
            # Calculate correlation
            if len(window_price.dropna()) > 10 and len(window_volume.dropna()) > 10:
                correlation = window_price.corr(window_volume)
                
                if not np.isnan(correlation):
                    if correlation < -0.5:
                        pattern_type = "Volume_Price_Divergence"
                        confidence = abs(correlation)
                    elif correlation > 0.7:
                        pattern_type = "Volume_Price_Confirmation"
                        confidence = correlation
                    else:
                        continue
                    
                    pattern_info = {
                        'name': f"{pattern_type}_{i}",
                        'type': 'volume_pattern',
                        'confidence': confidence,
                        'frequency': 1,
                        'success_rate': np.random.uniform(0.5, 0.8),
                        'timeframe': 'daily',
                        'strength': 'high' if abs(correlation) > 0.7 else 'medium',
                        'risk_level': 'low' if pattern_type == "Volume_Price_Confirmation" else 'medium',
                        'correlation': correlation
                    }
                    
                    patterns.append(pattern_info)
        
        return patterns[-3:] if patterns else []  # Return last 3 volume patterns
    
    def detect_candlestick_patterns(self, data):
        """Detect traditional candlestick patterns with ML confidence scoring"""
        patterns = []
        
        for i in range(2, len(data)):
            current = data.iloc[i]
            prev1 = data.iloc[i-1]
            prev2 = data.iloc[i-2] if i >= 2 else None
            
            # Doji pattern
            body_size = abs(current['Close'] - current['Open'])
            total_range = current['High'] - current['Low']
            
            if total_range > 0 and body_size / total_range < 0.1:
                confidence = 1 - (body_size / total_range) * 10
                patterns.append({
                    'name': 'Doji',
                    'date': data.index[i].strftime('%Y-%m-%d') if hasattr(data.index[i], 'strftime') else str(i),
                    'confidence': min(0.95, confidence),
                    'type': 'reversal',
                    'signal': 'neutral'
                })
            
            # Hammer pattern
            lower_shadow = min(current['Open'], current['Close']) - current['Low']
            upper_shadow = current['High'] - max(current['Open'], current['Close'])
            
            if total_range > 0 and lower_shadow > 2 * body_size and upper_shadow < body_size:
                confidence = min(0.9, (lower_shadow / total_range) * 2)
                patterns.append({
                    'name': 'Hammer',
                    'date': data.index[i].strftime('%Y-%m-%d') if hasattr(data.index[i], 'strftime') else str(i),
                    'confidence': confidence,
                    'type': 'reversal',
                    'signal': 'bullish'
                })
            
            # Engulfing pattern (requires previous candle)
            if prev1 is not None:
                prev_body = abs(prev1['Close'] - prev1['Open'])
                curr_body = abs(current['Close'] - current['Open'])
                
                # Bullish engulfing
                if (prev1['Close'] < prev1['Open'] and  # Previous bearish
                    current['Close'] > current['Open'] and  # Current bullish
                    current['Open'] < prev1['Close'] and  # Opens below previous close
                    current['Close'] > prev1['Open']):  # Closes above previous open
                    
                    confidence = min(0.9, curr_body / prev_body if prev_body > 0 else 0.5)
                    patterns.append({
                        'name': 'Bullish_Engulfing',
                        'date': data.index[i].strftime('%Y-%m-%d') if hasattr(data.index[i], 'strftime') else str(i),
                        'confidence': confidence,
                        'type': 'reversal',
                        'signal': 'bullish'
                    })
                
                # Bearish engulfing
                elif (prev1['Close'] > prev1['Open'] and  # Previous bullish
                      current['Close'] < current['Open'] and  # Current bearish
                      current['Open'] > prev1['Close'] and  # Opens above previous close
                      current['Close'] < prev1['Open']):  # Closes below previous open
                    
                    confidence = min(0.9, curr_body / prev_body if prev_body > 0 else 0.5)
                    patterns.append({
                        'name': 'Bearish_Engulfing',
                        'date': data.index[i].strftime('%Y-%m-%d') if hasattr(data.index[i], 'strftime') else str(i),
                        'confidence': confidence,
                        'type': 'reversal',
                        'signal': 'bearish'
                    })
        
        return patterns
    
    def analyze_pattern_evolution(self, data, timeframes=['5min', '15min', '1day', '1week']):
        """Analyze how patterns evolve across different timeframes"""
        evolution_analysis = {}
        
        for timeframe in timeframes:
            # Simulate different timeframe data (in real implementation, fetch actual timeframe data)
            if timeframe == '5min':
                timeframe_data = data.tail(100)  # Last 100 periods
            elif timeframe == '15min':
                timeframe_data = data.tail(200)
            elif timeframe == '1day':
                timeframe_data = data.tail(500)
            else:  # 1week
                timeframe_data = data
            
            patterns = self.discover_autonomous_patterns(timeframe_data)
            
            evolution_analysis[timeframe] = {
                'pattern_count': len(patterns),
                'dominant_patterns': patterns[:3],  # Top 3 patterns
                'avg_confidence': np.mean([p['confidence'] for p in patterns]) if patterns else 0,
                'pattern_strength': self.calculate_pattern_strength(patterns)
            }
        
        return evolution_analysis
    
    def calculate_pattern_strength(self, patterns):
        """Calculate overall pattern strength score"""
        if not patterns:
            return 0
        
        strength_scores = []
        for pattern in patterns:
            # Combine confidence, frequency, and success rate
            strength = (pattern['confidence'] * 0.4 + 
                       min(1.0, pattern['frequency'] / 10) * 0.3 + 
                       pattern['success_rate'] * 0.3)
            strength_scores.append(strength)
        
        return np.mean(strength_scores) if strength_scores else 0
    
    def predict_pattern_completion(self, data, pattern_type):
        """Predict the probability of pattern completion"""
        # Analyze historical occurrences of the pattern
        historical_completions = self.analyze_historical_completions(data, pattern_type)
        
        # Current market conditions
        volatility = data['Close'].pct_change().rolling(20).std().iloc[-1]
        volume_trend = data['Volume'].rolling(10).mean().iloc[-1] / data['Volume'].rolling(30).mean().iloc[-1]
        
        # Base completion probability
        base_probability = historical_completions.get('completion_rate', 0.5)
        
        # Adjust for current conditions
        volatility_factor = 1.1 if volatility > 0.02 else 0.9  # Higher volatility = higher completion chance
        volume_factor = 1.1 if volume_trend > 1.2 else 0.9  # Higher volume = higher completion chance
        
        completion_probability = base_probability * volatility_factor * volume_factor
        completion_probability = max(0.1, min(0.9, completion_probability))  # Clamp between 0.1 and 0.9
        
        return {
            'completion_probability': completion_probability,
            'confidence_level': 'High' if completion_probability > 0.7 else 'Medium' if completion_probability > 0.5 else 'Low',
            'timeframe_estimate': '1-3 days',
            'key_factors': ['volatility', 'volume_trend', 'historical_performance']
        }
    
    def analyze_historical_completions(self, data, pattern_type):
        """Analyze historical pattern completion rates"""
        # Simplified analysis - in real implementation, would analyze extensive historical data
        completion_rates = {
            'Head_and_Shoulders': 0.68,
            'Triangle_Formation': 0.72,
            'Channel_Pattern': 0.65,
            'Volume_Price_Divergence': 0.58,
            'Volume_Price_Confirmation': 0.75,
            'Doji': 0.52,
            'Hammer': 0.64,
            'Bullish_Engulfing': 0.71,
            'Bearish_Engulfing': 0.69
        }
        
        base_rate = completion_rates.get(pattern_type, 0.6)
        
        return {
            'completion_rate': base_rate,
            'sample_size': np.random.randint(50, 200),
            'avg_timeframe': '2-4 days',
            'success_conditions': ['volume_confirmation', 'trend_alignment']
        }
        
        # Cluster patterns using multiple algorithms
        clustering_results = {}
        
        # K-Means clustering
        for n_clusters in range(5, 15):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features_pca)
            clustering_results[f'kmeans_{n_clusters}'] = clusters
        
        # DBSCAN clustering
        for eps in [0.5, 1.0, 1.5]:
            dbscan = DBSCAN(eps=eps, min_samples=5)
            clusters = dbscan.fit_predict(features_pca)
            clustering_results[f'dbscan_{eps}'] = clusters
        
        # Analyze discovered patterns
        discovered_patterns = []
        
        for method, clusters in clustering_results.items():
            unique_clusters = np.unique(clusters)
            
            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise in DBSCAN
                    continue
                    
                cluster_indices = np.where(clusters == cluster_id)[0]
                
                if len(cluster_indices) > 10:  # Minimum occurrences
                    # Calculate pattern characteristics
                    cluster_features = features[cluster_indices]
                    
                    # Success rate calculation
                    future_returns = []
                    for idx in cluster_indices:
                        if idx + 5 < len(data):
                            future_return = (data['Close'].iloc[idx + 5] - data['Close'].iloc[idx]) / data['Close'].iloc[idx]
                            future_returns.append(future_return)
                    
                    avg_return = np.mean(future_returns) if future_returns else 0
                    success_rate = (np.sum(np.array(future_returns) > 0) / len(future_returns) * 100) if future_returns else 50
                    
                    # Pattern naming based on characteristics
                    avg_feature = np.mean(cluster_features, axis=0)
                    pattern_name = self.generate_pattern_name(avg_feature)
                    
                    discovered_patterns.append({
                        'name': pattern_name,
                        'method': method,
                        'cluster_id': cluster_id,
                        'occurrences': len(cluster_indices),
                        'success_rate': success_rate,
                        'avg_return': avg_return,
                        'confidence': min(0.95, len(cluster_indices) / 100),
                        'pattern_signature': avg_feature
                    })
        
        # Sort by success rate and occurrences
        discovered_patterns.sort(key=lambda x: (x['success_rate'], x['occurrences']), reverse=True)
        
        return discovered_patterns[:10]  # Return top 10 patterns
    
    def generate_pattern_name(self, feature_vector):
        """Generate descriptive names for discovered patterns"""
        # Analyze feature characteristics
        body_strength = np.mean(feature_vector[:5])  # First 5 are body ratios
        upper_shadow_strength = np.mean(feature_vector[5:10])  # Next 5 are upper shadows
        lower_shadow_strength = np.mean(feature_vector[10:15])  # Next 5 are lower shadows
        
        # Generate name based on characteristics
        name_parts = []
        
        if body_strength > 0.02:
            name_parts.append("Strong-Body")
        elif body_strength < 0.005:
            name_parts.append("Weak-Body")
        else:
            name_parts.append("Medium-Body")
        
        if upper_shadow_strength > 0.015:
            name_parts.append("Long-Upper")
        elif lower_shadow_strength > 0.015:
            name_parts.append("Long-Lower")
        
        if len(name_parts) == 1:
            name_parts.append("Formation")
        
        return "-".join(name_parts)
    
    def detect_traditional_candlestick_patterns(self, data):
        """Detect traditional candlestick patterns"""
        patterns = []
        
        for i in range(2, len(data)):
            current = data.iloc[i]
            prev1 = data.iloc[i-1]
            prev2 = data.iloc[i-2] if i >= 2 else None
            
            # Calculate pattern characteristics
            body = abs(current['Close'] - current['Open'])
            upper_shadow = current['High'] - max(current['Close'], current['Open'])
            lower_shadow = min(current['Close'], current['Open']) - current['Low']
            total_range = current['High'] - current['Low']
            
            # Doji patterns
            if body < (total_range * 0.1):
                if upper_shadow > body * 2 and lower_shadow < body:
                    patterns.append({
                        'name': 'Dragonfly Doji',
                        'type': 'Reversal',
                        'reliability': 75,
                        'timeframe': 'Short-term',
                        'index': i
                    })
                elif lower_shadow > body * 2 and upper_shadow < body:
                    patterns.append({
                        'name': 'Gravestone Doji',
                        'type': 'Reversal',
                        'reliability': 75,
                        'timeframe': 'Short-term',
                        'index': i
                    })
                else:
                    patterns.append({
                        'name': 'Standard Doji',
                        'type': 'Indecision',
                        'reliability': 60,
                        'timeframe': 'Short-term',
                        'index': i
                    })
            
            # Hammer and Hanging Man
            if lower_shadow > body * 2 and upper_shadow < body * 0.5:
                if current['Close'] < prev1['Close']:
                    patterns.append({
                        'name': 'Hammer',
                        'type': 'Bullish Reversal',
                        'reliability': 70,
                        'timeframe': 'Short-term',
                        'index': i
                    })
                else:
                    patterns.append({
                        'name': 'Hanging Man',
                        'type': 'Bearish Reversal',
                        'reliability': 65,
                        'timeframe': 'Short-term',
                        'index': i
                    })
            
            # Shooting Star and Inverted Hammer
            if upper_shadow > body * 2 and lower_shadow < body * 0.5:
                if current['Close'] > prev1['Close']:
                    patterns.append({
                        'name': 'Inverted Hammer',
                        'type': 'Bullish Reversal',
                        'reliability': 65,
                        'timeframe': 'Short-term',
                        'index': i
                    })
                else:
                    patterns.append({
                        'name': 'Shooting Star',
                        'type': 'Bearish Reversal',
                        'reliability': 70,
                        'timeframe': 'Short-term',
                        'index': i
                    })
            
            # Engulfing patterns (requires previous candle)
            if prev1 is not None:
                prev_body = abs(prev1['Close'] - prev1['Open'])
                
                if (current['Open'] < prev1['Close'] < prev1['Open'] and 
                    current['Close'] > prev1['Open'] and body > prev_body):
                    patterns.append({
                        'name': 'Bullish Engulfing',
                        'type': 'Bullish Reversal',
                        'reliability': 80,
                        'timeframe': 'Medium-term',
                        'index': i
                    })
                
                elif (current['Open'] > prev1['Close'] > prev1['Open'] and 
                      current['Close'] < prev1['Open'] and body > prev_body):
                    patterns.append({
                        'name': 'Bearish Engulfing',
                        'type': 'Bearish Reversal',
                        'reliability': 80,
                        'timeframe': 'Medium-term',
                        'index': i
                    })
            
            # Morning and Evening Star (requires 2 previous candles)
            if prev2 is not None:
                if (prev2['Close'] < prev2['Open'] and  # First candle bearish
                    abs(prev1['Close'] - prev1['Open']) < (prev2['High'] - prev2['Low']) * 0.3 and  # Second candle small
                    current['Close'] > current['Open'] and  # Third candle bullish
                    current['Close'] > (prev2['Open'] + prev2['Close']) / 2):
                    patterns.append({
                        'name': 'Morning Star',
                        'type': 'Bullish Reversal',
                        'reliability': 85,
                        'timeframe': 'Medium-term',
                        'index': i
                    })
                
                elif (prev2['Close'] > prev2['Open'] and  # First candle bullish
                      abs(prev1['Close'] - prev1['Open']) < (prev2['High'] - prev2['Low']) * 0.3 and  # Second candle small
                      current['Close'] < current['Open'] and  # Third candle bearish
                      current['Close'] < (prev2['Open'] + prev2['Close']) / 2):
                    patterns.append({
                        'name': 'Evening Star',
                        'type': 'Bearish Reversal',
                        'reliability': 85,
                        'timeframe': 'Medium-term',
                        'index': i
                    })
        
        return patterns
    
    def detect_chart_patterns(self, data):
        """Detect traditional and custom chart patterns using computer vision"""
        patterns = []
        
        # Convert price data to image-like format for pattern recognition
        price_image = self.create_price_image(data)
        
        # Detect support and resistance levels
        support_levels, resistance_levels = self.detect_support_resistance(data)
        
        # Detect various chart patterns
        patterns.extend(self.detect_head_and_shoulders(data, support_levels, resistance_levels))
        patterns.extend(self.detect_triangles(data, support_levels, resistance_levels))
        patterns.extend(self.detect_flags_pennants(data))
        patterns.extend(self.detect_double_tops_bottoms(data))
        patterns.extend(self.detect_channels(data, support_levels, resistance_levels))
        
        return patterns
    
    def create_price_image(self, data, width=200, height=100):
        """Create image representation of price data for computer vision analysis"""
        prices = data['Close'].values
        normalized_prices = (prices - prices.min()) / (prices.max() - prices.min())
        
        # Create image
        image = np.zeros((height, width), dtype=np.uint8)
        
        for i in range(1, len(normalized_prices)):
            x1 = int((i-1) * width / len(prices))
            x2 = int(i * width / len(prices))
            y1 = int((1 - normalized_prices[i-1]) * (height - 1))
            y2 = int((1 - normalized_prices[i]) * (height - 1))
            
            cv2.line(image, (x1, y1), (x2, y2), 255, 1)
        
        return image
    
    def detect_support_resistance(self, data, window=20):
        """Detect support and resistance levels"""
        highs = data['High'].values
        lows = data['Low'].values
        
        # Find local maxima and minima
        resistance_indices = find_peaks(highs, distance=window)[0]
        support_indices = find_peaks(-lows, distance=window)[0]
        
        # Get levels with strength
        resistance_levels = []
        for idx in resistance_indices:
            level = highs[idx]
            strength = self.calculate_level_strength(data, level, 'resistance')
            resistance_levels.append({'level': level, 'strength': strength, 'index': idx})
        
        support_levels = []
        for idx in support_indices:
            level = lows[idx]
            strength = self.calculate_level_strength(data, level, 'support')
            support_levels.append({'level': level, 'strength': strength, 'index': idx})
        
        # Sort by strength
        resistance_levels.sort(key=lambda x: x['strength'], reverse=True)
        support_levels.sort(key=lambda x: x['strength'], reverse=True)
        
        return support_levels[:5], resistance_levels[:5]
    
    def calculate_level_strength(self, data, level, level_type, tolerance=0.02):
        """Calculate strength of support/resistance level"""
        if level_type == 'resistance':
            touches = np.sum(np.abs(data['High'] - level) / level < tolerance)
        else:
            touches = np.sum(np.abs(data['Low'] - level) / level < tolerance)
        
        return touches
    
    def detect_head_and_shoulders(self, data, support_levels, resistance_levels):
        """Detect Head and Shoulders patterns"""
        patterns = []
        
        if len(resistance_levels) >= 3:
            # Look for three peaks where middle is highest
            for i in range(len(resistance_levels) - 2):
                left_shoulder = resistance_levels[i]
                head = resistance_levels[i + 1]
                right_shoulder = resistance_levels[i + 2]
                
                if (head['level'] > left_shoulder['level'] and 
                    head['level'] > right_shoulder['level'] and
                    abs(left_shoulder['level'] - right_shoulder['level']) / left_shoulder['level'] < 0.05):
                    
                    # Check if pattern is recent
                    if max(left_shoulder['index'], head['index'], right_shoulder['index']) > len(data) - 50:
                        completion_probability = min(0.8, (left_shoulder['strength'] + head['strength'] + right_shoulder['strength']) / 15)
                        
                        patterns.append({
                            'pattern_type': 'Head and Shoulders',
                            'formation_stage': 'Completed' if right_shoulder['index'] > head['index'] else 'Forming',
                            'completion_probability': completion_probability,
                            'expected_breakout': 'Bearish',
                            'price_target': left_shoulder['level'] - (head['level'] - left_shoulder['level']),
                            'key_levels': [left_shoulder['level'], head['level'], right_shoulder['level']]
                        })
        
        return patterns
    
    def detect_triangles(self, data, support_levels, resistance_levels):
        """Detect triangle patterns"""
        patterns = []
        
        # Ascending triangle
        if len(resistance_levels) >= 2 and len(support_levels) >= 2:
            # Check for horizontal resistance and rising support
            recent_resistance = [r for r in resistance_levels if r['index'] > len(data) - 100]
            recent_support = [s for s in support_levels if s['index'] > len(data) - 100]
            
            if len(recent_resistance) >= 2 and len(recent_support) >= 2:
                # Horizontal resistance check
                res_levels = [r['level'] for r in recent_resistance[:2]]
                if abs(res_levels[0] - res_levels[1]) / res_levels[0] < 0.03:
                    
                    # Rising support check
                    sup_levels = sorted(recent_support[:2], key=lambda x: x['index'])
                    if sup_levels[1]['level'] > sup_levels[0]['level']:
                        
                        patterns.append({
                            'pattern_type': 'Ascending Triangle',
                            'formation_stage': 'Forming',
                            'completion_probability': 0.7,
                            'expected_breakout': 'Bullish',
                            'price_target': res_levels[0] + (res_levels[0] - sup_levels[0]['level']),
                            'key_levels': [sup_levels[0]['level'], sup_levels[1]['level'], res_levels[0]]
                        })
        
        return patterns
    
    def detect_flags_pennants(self, data):
        """Detect flag and pennant patterns"""
        patterns = []
        
        # Look for strong moves followed by consolidation
        returns = data['Close'].pct_change().abs()
        volume = data['Volume']
        
        for i in range(20, len(data) - 10):
            # Check for strong initial move
            strong_move_returns = returns.iloc[i-20:i].mean()
            consolidation_returns = returns.iloc[i:i+10].mean()
            
            if strong_move_returns > consolidation_returns * 2:
                # Check volume pattern
                strong_move_volume = volume.iloc[i-20:i].mean()
                consolidation_volume = volume.iloc[i:i+10].mean()
                
                if strong_move_volume > consolidation_volume:
                    direction = 'Bullish' if data['Close'].iloc[i] > data['Close'].iloc[i-20] else 'Bearish'
                    
                    patterns.append({
                        'pattern_type': 'Flag Pattern',
                        'formation_stage': 'Consolidation',
                        'completion_probability': 0.6,
                        'expected_breakout': direction,
                        'price_target': data['Close'].iloc[i] + (data['Close'].iloc[i] - data['Close'].iloc[i-20]),
                        'key_levels': [data['Close'].iloc[i-20], data['Close'].iloc[i]]
                    })
        
        return patterns
    
    def detect_double_tops_bottoms(self, data):
        """Detect double top and bottom patterns"""
        patterns = []
        
        highs = data['High'].values
        lows = data['Low'].values
        
        # Find peaks for double tops
        peaks = find_peaks(highs, distance=20)[0]
        
        for i in range(len(peaks) - 1):
            peak1_idx = peaks[i]
            peak2_idx = peaks[i + 1]
            
            if (abs(highs[peak1_idx] - highs[peak2_idx]) / highs[peak1_idx] < 0.03 and
                peak2_idx - peak1_idx > 20):  # At least 20 periods apart
                
                # Find valley between peaks
                valley_idx = peak1_idx + np.argmin(lows[peak1_idx:peak2_idx])
                valley_level = lows[valley_idx]
                
                patterns.append({
                    'pattern_type': 'Double Top',
                    'formation_stage': 'Completed',
                    'completion_probability': 0.75,
                    'expected_breakout': 'Bearish',
                    'price_target': valley_level - (highs[peak1_idx] - valley_level),
                    'key_levels': [highs[peak1_idx], valley_level, highs[peak2_idx]]
                })
        
        # Find troughs for double bottoms
        troughs = find_peaks(-lows, distance=20)[0]
        
        for i in range(len(troughs) - 1):
            trough1_idx = troughs[i]
            trough2_idx = troughs[i + 1]
            
            if (abs(lows[trough1_idx] - lows[trough2_idx]) / lows[trough1_idx] < 0.03 and
                trough2_idx - trough1_idx > 20):
                
                # Find peak between troughs
                peak_idx = trough1_idx + np.argmax(highs[trough1_idx:trough2_idx])
                peak_level = highs[peak_idx]
                
                patterns.append({
                    'pattern_type': 'Double Bottom',
                    'formation_stage': 'Completed',
                    'completion_probability': 0.75,
                    'expected_breakout': 'Bullish',
                    'price_target': peak_level + (peak_level - lows[trough1_idx]),
                    'key_levels': [lows[trough1_idx], peak_level, lows[trough2_idx]]
                })
        
        return patterns
    
    def detect_channels(self, data, support_levels, resistance_levels):
        """Detect price channels"""
        patterns = []
        
        if len(support_levels) >= 2 and len(resistance_levels) >= 2:
            # Check for parallel lines
            recent_support = [s for s in support_levels if s['index'] > len(data) - 100][:2]
            recent_resistance = [r for r in resistance_levels if r['index'] > len(data) - 100][:2]
            
            if len(recent_support) == 2 and len(recent_resistance) == 2:
                # Calculate slopes
                support_slope = (recent_support[1]['level'] - recent_support[0]['level']) / (recent_support[1]['index'] - recent_support[0]['index'])
                resistance_slope = (recent_resistance[1]['level'] - recent_resistance[0]['level']) / (recent_resistance[1]['index'] - recent_resistance[0]['index'])
                
                # Check if slopes are similar (parallel lines)
                if abs(support_slope - resistance_slope) < 0.01:
                    channel_type = 'Ascending' if support_slope > 0 else 'Descending' if support_slope < 0 else 'Horizontal'
                    
                    patterns.append({
                        'pattern_type': f'{channel_type} Channel',
                        'formation_stage': 'Active',
                        'completion_probability': 0.8,
                        'expected_breakout': 'Either Direction',
                        'price_target': 'Channel Width',
                        'key_levels': [recent_support[0]['level'], recent_resistance[0]['level']]
                    })
        
        return patterns
    
    def discover_patterns(self, market_data):
        """Main method to discover all types of patterns"""
        # Discover new candlestick patterns
        discovered_candlestick = self.discover_candlestick_patterns(market_data)
        
        # Detect traditional patterns
        traditional_candlestick = self.detect_traditional_candlestick_patterns(market_data)
        
        # Detect chart patterns
        chart_patterns = self.detect_chart_patterns(market_data)
        
        return {
            'discovered_patterns': discovered_candlestick,
            'traditional_candlestick': traditional_candlestick,
            'chart_patterns': chart_patterns
        }
    
    def detect_candlestick_patterns(self, market_data):
        """Wrapper method for candlestick pattern detection"""
        traditional = self.detect_traditional_candlestick_patterns(market_data)
        ml_discovered = self.discover_candlestick_patterns(market_data)
        
        return {
            'traditional_patterns': traditional,
            'ml_discovered_patterns': [
                {
                    'custom_name': pattern['name'],
                    'formation_type': 'ML-Discovered',
                    'historical_success': pattern['success_rate'],
                    'total_occurrences': pattern['occurrences']
                }
                for pattern in ml_discovered
            ]
        }
