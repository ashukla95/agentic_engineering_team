```python
import sqlite3
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import threading
import time
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import json
import logging
from typing import List, Dict, Optional, Tuple
import gradio as gr
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Alert:
    id: Optional[int]
    user_id: int
    stock_symbol: str
    alert_type: str  # 'price', 'volume', 'market_cap'
    condition: str  # 'above', 'below', 'change_percent'
    threshold: float
    is_active: bool = True

@dataclass
class WatchlistItem:
    id: Optional[int]
    user_id: int
    stock_symbol: str
    category: str
    notes: str
    date_added: datetime

class DatabaseManager:
    def __init__(self, db_path='watchlist.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Watchlist table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS watchlist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    stock_symbol TEXT NOT NULL,
                    category TEXT DEFAULT 'General',
                    notes TEXT DEFAULT '',
                    date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    UNIQUE(user_id, stock_symbol)
                )
            ''')
            
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    stock_symbol TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Stock data cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_data (
                    symbol TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    close_price REAL,
                    volume INTEGER,
                    market_cap REAL,
                    PRIMARY KEY (symbol, timestamp)
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_watchlist_user ON watchlist(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_user ON alerts(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_data_symbol ON stock_data(symbol)')
            
            conn.commit()
    
    def add_user(self, username: str, email: str) -> int:
        """Add a new user and return user_id"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (username, email) VALUES (?, ?)', (username, email))
            return cursor.lastrowid
    
    def get_user_id(self, username: str) -> Optional[int]:
        """Get user ID by username"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            result = cursor.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
            return result[0] if result else None
    
    def add_to_watchlist(self, watchlist_item: WatchlistItem) -> bool:
        """Add stock to user's watchlist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO watchlist (user_id, stock_symbol, category, notes)
                    VALUES (?, ?, ?, ?)
                ''', (watchlist_item.user_id, watchlist_item.stock_symbol, 
                     watchlist_item.category, watchlist_item.notes))
                return True
        except Exception as e:
            logger.error(f"Error adding to watchlist: {e}")
            return False
    
    def get_watchlist(self, user_id: int) -> List[WatchlistItem]:
        """Get user's watchlist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            results = cursor.execute('''
                SELECT id, user_id, stock_symbol, category, notes, date_added
                FROM watchlist WHERE user_id = ?
            ''', (user_id,)).fetchall()
            
            return [WatchlistItem(id=r[0], user_id=r[1], stock_symbol=r[2], 
                                category=r[3], notes=r[4], 
                                date_added=datetime.fromisoformat(r[5]))
                   for r in results]
    
    def remove_from_watchlist(self, user_id: int, stock_symbol: str) -> bool:
        """Remove stock from watchlist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM watchlist WHERE user_id = ? AND stock_symbol = ?',
                             (user_id, stock_symbol))
                return True
        except Exception as e:
            logger.error(f"Error removing from watchlist: {e}")
            return False
    
    def add_alert(self, alert: Alert) -> bool:
        """Add a new alert"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO alerts (user_id, stock_symbol, alert_type, condition, threshold, is_active)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (alert.user_id, alert.stock_symbol, alert.alert_type, 
                     alert.condition, alert.threshold, alert.is_active))
                return True
        except Exception as e:
            logger.error(f"Error adding alert: {e}")
            return False
    
    def get_alerts(self, user_id: int) -> List[Alert]:
        """Get user's active alerts"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            results = cursor.execute('''
                SELECT id, user_id, stock_symbol, alert_type, condition, threshold, is_active
                FROM alerts WHERE user_id = ? AND is_active = 1
            ''', (user_id,)).fetchall()
            
            return [Alert(id=r[0], user_id=r[1], stock_symbol=r[2], 
                         alert_type=r[3], condition=r[4], threshold=r[5], is_active=r[6])
                   for r in results]
    
    def store_stock_data(self, symbol: str, data: Dict) -> bool:
        """Store stock data in cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO stock_data 
                    (symbol, timestamp, open_price, high_price, low_price, close_price, volume, market_cap)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, datetime.now(), data.get('open'), data.get('high'),
                     data.get('low'), data.get('close'), data.get('volume'), data.get('market_cap')))
                return True
        except Exception as e:
            logger.error(f"Error storing stock data: {e}")
            return False

class StockDataService:
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 60  # 1 minute cache
    
    def get_stock_data(self, symbol: str) -> Optional[Dict]:
        """Fetch real-time stock data"""
        try:
            # Check cache first
            if symbol in self.cache:
                cached_data, timestamp = self.cache[symbol]
                if (datetime.now() - timestamp).seconds < self.cache_timeout:
                    return cached_data
            
            # Fetch from Yahoo Finance
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period='1d', interval='1m')
            
            if hist.empty:
                return None
            
            latest = hist.iloc[-1]
            
            data = {
                'symbol': symbol,
                'current_price': float(latest['Close']),
                'open': float(latest['Open']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'close': float(latest['Close']),
                'volume': int(latest['Volume']),
                'market_cap': info.get('marketCap', 0),
                'previous_close': info.get('previousClose', 0),
                'change': float(latest['Close']) - info.get('previousClose', 0),
                'change_percent': ((float(latest['Close']) - info.get('previousClose', 0)) / 
                                 info.get('previousClose', 1)) * 100,
                'timestamp': datetime.now()
            }
            
            # Update cache
            self.cache[symbol] = (data, datetime.now())
            return data
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, period: str = '1y') -> pd.DataFrame:
        """Get historical stock data for AI analysis"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            return hist
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

class AIPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for ML model"""
        if df.empty or len(df) < 10:
            return np.array([]), np.array([])
        
        # Calculate technical indicators
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Create features
        features = []
        targets = []
        
        for i in range(20, len(df) - 1):  # Need enough history for indicators
            feature_row = [
                df.iloc[i]['MA_5'],
                df.iloc[i]['MA_20'],
                df.iloc[i]['RSI'],
                df.iloc[i]['Price_Change'],
                df.iloc[i]['Volume_Change'],
                df.iloc[i]['Volume'],
                df.iloc[i]['High'] - df.iloc[i]['Low'],  # Daily range
            ]
            
            if not any(pd.isna(feature_row)):
                features.append(feature_row)
                # Predict next day's price change
                targets.append(df.iloc[i + 1]['Close'] - df.iloc[i]['Close'])
        
        return np.array(features), np.array(targets)
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def train_model(self, symbol: str, historical_data: pd.DataFrame) -> bool:
        """Train prediction model for a specific stock"""
        try:
            features, targets = self.prepare_features(historical_data)
            
            if len(features) == 0:
                logger.warning(f"Insufficient data to train model for {symbol}")
                return False
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Train model
            model = LinearRegression()
            model.fit(features_scaled, targets)
            
            # Store model and scaler
            self.models[symbol] = model
            self.scalers[symbol] = scaler
            
            logger.info(f"Model trained successfully for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            return False
    
    def predict_trend(self, symbol: str, current_data: Dict, historical_data: pd.DataFrame) -> Dict:
        """Predict stock trend"""
        try:
            if symbol not in self.models:
                # Train model if not exists
                if not self.train_model(symbol, historical_data):
                    return {'prediction': 'unknown', 'confidence': 0.0, 'message': 'Insufficient data'}
            
            # Prepare current features (using last 20 days)
            if len(historical_data) < 20:
                return {'prediction': 'unknown', 'confidence': 0.0, 'message': 'Insufficient historical data'}
            
            recent_data = historical_data.tail(20).copy()
            recent_data.loc[len(recent_data)] = {
                'Close': current_data['current_price'],
                'Volume': current_data['volume'],
                'High': current_data['high'],
                'Low': current_data['low'],
                'Open': current_data['open']
            }
            
            features, _ = self.prepare_features(recent_data)
            
            if len(features) == 0:
                return {'prediction': 'unknown', 'confidence': 0.0, 'message': 'Unable to generate features'}
            
            # Make prediction
            last_features = features[-1].reshape(1, -1)
            last_features_scaled = self.scalers[symbol].transform(last_features)
            prediction = self.models[symbol].predict(last_features_scaled)[0]
            
            # Determine trend
            if prediction > current_data['current_price'] * 0.02:  # > 2% increase
                trend = 'bullish'
                confidence = min(abs(prediction) / current_data['current_price'] * 100, 95.0)
            elif prediction < -current_data['current_price'] * 0.02:  # > 2% decrease
                trend = 'bearish'
                confidence = min(abs(prediction) / current_data['current_price'] * 100, 95.0)