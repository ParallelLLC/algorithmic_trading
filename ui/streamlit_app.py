"""
Streamlit UI for Algorithmic Trading System

A comprehensive web interface for:
- Real-time market data visualization
- Trading strategy configuration
- FinRL model training and evaluation
- Portfolio management
- Risk monitoring
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import yaml
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import asyncio
import threading
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import with error handling for deployment
try:
    from agentic_ai_system.main import load_config
    from agentic_ai_system.data_ingestion import load_data, validate_data, add_technical_indicators
    from agentic_ai_system.finrl_agent import FinRLAgent, FinRLConfig
    from agentic_ai_system.alpaca_broker import AlpacaBroker
    from agentic_ai_system.orchestrator import run_backtest, run_live_trading
    DEPLOYMENT_MODE = False
except ImportError as e:
    st.warning(f"⚠️ Some modules not available in deployment mode: {e}")
    DEPLOYMENT_MODE = True
    
    # Mock functions for deployment
    def load_config(config_file):
        return {
            'trading': {'symbol': 'AAPL', 'capital': 100000, 'timeframe': '1d'},
            'execution': {'broker_api': 'alpaca_paper'},
            'finrl': {'algorithm': 'PPO'},
            'risk': {'max_drawdown': 0.1}
        }
    
    def load_data(config):
        # Generate sample data for deployment
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        prices = 150 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        })
        return data
    
    def add_technical_indicators(data):
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        return data
    
    class FinRLAgent:
        def __init__(self, config):
            self.config = config
        
        def train(self, data, config, total_timesteps, use_real_broker=False):
            return {'success': True, 'message': 'Training completed (demo mode)'}
    
    class FinRLConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class AlpacaBroker:
        def __init__(self, config):
            self.config = config
        
        def get_account_info(self):
            return {
                'portfolio_value': 100000,
                'equity': 102500,
                'cash': 50000,
                'buying_power': 50000
            }
        
        def get_positions(self):
            return []
    
    def run_backtest(config, data):
        return {
            'success': True,
            'total_return': 0.025,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.05,
            'total_trades': 15
        }
    
    def run_live_trading(config, data):
        return {'success': True, 'message': 'Live trading started (demo mode)'}

# Page configuration
st.set_page_config(
    page_title="Algorithmic Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

class TradingUI:
    def __init__(self):
        self.config = None
        self.data = None
        self.alpaca_broker = None
        self.finrl_agent = None
        self.session_state = st.session_state
        
        # Initialize session state
        if 'trading_active' not in self.session_state:
            self.session_state.trading_active = False
        if 'current_portfolio' not in self.session_state:
            self.session_state.current_portfolio = {}
        if 'trading_history' not in self.session_state:
            self.session_state.trading_history = []
    
    def load_configuration(self):
        """Load and display configuration"""
        st.sidebar.header("⚙️ Configuration")
        
        # Config file selector
        config_files = [f for f in os.listdir('.') if f.endswith('.yaml') or f.endswith('.yml')]
        selected_config = st.sidebar.selectbox(
            "Select Configuration File",
            config_files,
            index=0 if 'config.yaml' in config_files else 0
        )
        
        if st.sidebar.button("Load Configuration"):
            try:
                self.config = load_config(selected_config)
                st.sidebar.success(f"✅ Configuration loaded: {selected_config}")
                return True
            except Exception as e:
                st.sidebar.error(f"❌ Error loading config: {e}")
                return False
        
        return False
    
    def display_system_status(self):
        """Display system status and metrics"""
        st.header("📊 System Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Trading Status",
                value="🟢 Active" if self.session_state.trading_active else "🔴 Inactive",
                delta="Running" if self.session_state.trading_active else "Stopped"
            )
        
        with col2:
            if self.config:
                st.metric(
                    label="Capital",
                    value=f"${self.config['trading']['capital']:,}",
                    delta="Available"
                )
            else:
                st.metric(label="Capital", value="Not Loaded")
        
        with col3:
            if self.alpaca_broker:
                try:
                    account_info = self.alpaca_broker.get_account_info()
                    if account_info:
                        st.metric(
                            label="Portfolio Value",
                            value=f"${float(account_info['portfolio_value']):,.2f}",
                            delta=f"{float(account_info['equity']) - float(account_info['portfolio_value']):,.2f}"
                        )
                except:
                    st.metric(label="Portfolio Value", value="Not Connected")
            else:
                st.metric(label="Portfolio Value", value="Not Connected")
        
        with col4:
            if self.data is not None:
                st.metric(
                    label="Data Points",
                    value=f"{len(self.data):,}",
                    delta=f"Latest: {self.data['timestamp'].max().strftime('%Y-%m-%d')}"
                )
            else:
                st.metric(label="Data Points", value="Not Loaded")
    
    def data_ingestion_panel(self):
        """Data ingestion and visualization panel"""
        st.header("📥 Data Ingestion")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if self.config:
                if st.button("Load Market Data"):
                    with st.spinner("Loading data..."):
                        try:
                            self.data = load_data(self.config)
                            if self.data is not None and not self.data.empty:
                                st.success(f"✅ Loaded {len(self.data)} data points")
                                
                                # Add technical indicators
                                self.data = add_technical_indicators(self.data)
                                st.info(f"✅ Added technical indicators")
                            else:
                                st.error("❌ Failed to load data")
                        except Exception as e:
                            st.error(f"❌ Error loading data: {e}")
        
        with col2:
            if self.data is not None:
                st.subheader("Data Summary")
                st.write(f"**Symbol:** {self.config['trading']['symbol']}")
                st.write(f"**Timeframe:** {self.config['trading']['timeframe']}")
                st.write(f"**Date Range:** {self.data['timestamp'].min().strftime('%Y-%m-%d')} to {self.data['timestamp'].max().strftime('%Y-%m-%d')}")
                st.write(f"**Price Range:** ${self.data['close'].min():.2f} - ${self.data['close'].max():.2f}")
        
        # Data visualization
        if self.data is not None:
            st.subheader("📈 Market Data Visualization")
            
            # Chart type selector
            chart_type = st.selectbox(
                "Chart Type",
                ["Candlestick", "Line", "OHLC", "Volume"]
            )
            
            if chart_type == "Candlestick":
                fig = go.Figure(data=[go.Candlestick(
                    x=self.data['timestamp'],
                    open=self.data['open'],
                    high=self.data['high'],
                    low=self.data['low'],
                    close=self.data['close']
                )])
                fig.update_layout(
                    title=f"{self.config['trading']['symbol']} Candlestick Chart",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Line":
                fig = px.line(self.data, x='timestamp', y='close', 
                             title=f"{self.config['trading']['symbol']} Price Chart")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Volume":
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=self.data['timestamp'],
                    y=self.data['volume'],
                    name='Volume'
                ))
                fig.update_layout(
                    title=f"{self.config['trading']['symbol']} Volume Chart",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def alpaca_integration_panel(self):
        """Alpaca broker integration panel"""
        st.header("🏦 Alpaca Integration")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Connect to Alpaca"):
                if self.config and self.config['execution']['broker_api'] in ['alpaca_paper', 'alpaca_live']:
                    with st.spinner("Connecting to Alpaca..."):
                        try:
                            self.alpaca_broker = AlpacaBroker(self.config)
                            account_info = self.alpaca_broker.get_account_info()
                            if account_info:
                                st.success("✅ Connected to Alpaca")
                                self.session_state.alpaca_connected = True
                            else:
                                st.error("❌ Failed to connect to Alpaca")
                        except Exception as e:
                            st.error(f"❌ Connection error: {e}")
                else:
                    st.warning("⚠️ Alpaca not configured in settings")
        
        with col2:
            if st.button("Disconnect from Alpaca"):
                self.alpaca_broker = None
                self.session_state.alpaca_connected = False
                st.success("✅ Disconnected from Alpaca")
        
        # Account information display
        if self.alpaca_broker:
            st.subheader("Account Information")
            
            try:
                account_info = self.alpaca_broker.get_account_info()
                if account_info:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Buying Power",
                            value=f"${float(account_info['buying_power']):,.2f}"
                        )
                    
                    with col2:
                        st.metric(
                            label="Portfolio Value",
                            value=f"${float(account_info['portfolio_value']):,.2f}"
                        )
                    
                    with col3:
                        st.metric(
                            label="Equity",
                            value=f"${float(account_info['equity']):,.2f}"
                        )
                    
                    # Market hours
                    market_hours = self.alpaca_broker.get_market_hours()
                    if market_hours:
                        status_color = "🟢" if market_hours['is_open'] else "🔴"
                        st.info(f"{status_color} Market Status: {'OPEN' if market_hours['is_open'] else 'CLOSED'}")
                        
                        if market_hours['next_open']:
                            st.write(f"Next Open: {market_hours['next_open']}")
                        if market_hours['next_close']:
                            st.write(f"Next Close: {market_hours['next_close']}")
                
                # Current positions
                positions = self.alpaca_broker.get_positions()
                if positions:
                    st.subheader("Current Positions")
                    positions_df = pd.DataFrame(positions)
                    st.dataframe(positions_df)
                else:
                    st.info("No current positions")
                    
            except Exception as e:
                st.error(f"Error fetching account info: {e}")
    
    def finrl_training_panel(self):
        """FinRL model training panel"""
        st.header("🧠 FinRL Model Training")
        
        if not self.data is not None:
            st.warning("⚠️ Please load market data first")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Training Configuration")
            
            # Training parameters
            algorithm = st.selectbox(
                "Algorithm",
                ["PPO", "A2C", "DDPG", "TD3"],
                index=0
            )
            
            learning_rate = st.slider(
                "Learning Rate",
                min_value=0.0001,
                max_value=0.01,
                value=0.0003,
                step=0.0001,
                format="%.4f"
            )
            
            total_timesteps = st.slider(
                "Total Timesteps",
                min_value=1000,
                max_value=1000000,
                value=100000,
                step=1000
            )
            
            batch_size = st.selectbox(
                "Batch Size",
                [32, 64, 128, 256],
                index=1
            )
        
        with col2:
            st.subheader("Training Controls")
            
            if st.button("Start Training", type="primary"):
                if self.data is not None:
                    with st.spinner("Training FinRL model..."):
                        try:
                            # Create FinRL config
                            finrl_config = FinRLConfig(
                                algorithm=algorithm,
                                learning_rate=learning_rate,
                                batch_size=batch_size,
                                buffer_size=1000000,
                                learning_starts=100,
                                gamma=0.99,
                                tau=0.005,
                                train_freq=1,
                                gradient_steps=1,
                                verbose=1,
                                tensorboard_log='logs/finrl_tensorboard'
                            )
                            
                            # Initialize agent
                            self.finrl_agent = FinRLAgent(finrl_config)
                            
                            # Train the agent
                            result = self.finrl_agent.train(
                                data=self.data,
                                config=self.config,
                                total_timesteps=total_timesteps,
                                use_real_broker=False
                            )
                            
                            if result['success']:
                                st.success("✅ Training completed successfully!")
                                st.write(f"Model saved: {result['model_path']}")
                                self.session_state.model_trained = True
                            else:
                                st.error("❌ Training failed")
                                
                        except Exception as e:
                            st.error(f"❌ Training error: {e}")
        
        # Training progress and metrics
        if hasattr(self.session_state, 'model_trained') and self.session_state.model_trained:
            st.subheader("Model Performance")
            
            if st.button("Evaluate Model"):
                if self.finrl_agent:
                    with st.spinner("Evaluating model..."):
                        try:
                            # Use last 100 data points for evaluation
                            eval_data = self.data.tail(100)
                            prediction_result = self.finrl_agent.predict(
                                data=eval_data,
                                config=self.config,
                                use_real_broker=False
                            )
                            
                            if prediction_result['success']:
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        label="Initial Value",
                                        value=f"${prediction_result['initial_value']:,.2f}"
                                    )
                                
                                with col2:
                                    st.metric(
                                        label="Final Value",
                                        value=f"${prediction_result['final_value']:,.2f}"
                                    )
                                
                                with col3:
                                    return_pct = prediction_result['total_return'] * 100
                                    st.metric(
                                        label="Total Return",
                                        value=f"{return_pct:.2f}%",
                                        delta=f"{return_pct:.2f}%"
                                    )
                                
                                st.write(f"Total Trades: {prediction_result['total_trades']}")
                            else:
                                st.error("❌ Model evaluation failed")
                                
                        except Exception as e:
                            st.error(f"❌ Evaluation error: {e}")
    
    def trading_controls_panel(self):
        """Trading controls and execution panel"""
        st.header("🎯 Trading Controls")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Backtesting")
            
            if st.button("Run Backtest"):
                if self.data is not None and self.config:
                    with st.spinner("Running backtest..."):
                        try:
                            result = run_backtest(self.config, self.data)
                            if result['success']:
                                st.success("✅ Backtest completed")
                                
                                # Display backtest results
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        label="Total Return",
                                        value=f"{result['total_return']:.2%}"
                                    )
                                
                                with col2:
                                    st.metric(
                                        label="Sharpe Ratio",
                                        value=f"{result['sharpe_ratio']:.2f}"
                                    )
                                
                                with col3:
                                    st.metric(
                                        label="Max Drawdown",
                                        value=f"{result['max_drawdown']:.2%}"
                                    )
                                
                                # Store results in session state
                                self.session_state.backtest_results = result
                            else:
                                st.error("❌ Backtest failed")
                                
                        except Exception as e:
                            st.error(f"❌ Backtest error: {e}")
        
        with col2:
            st.subheader("Live Trading")
            
            if st.button("Start Live Trading", type="primary"):
                if self.config and self.alpaca_broker:
                    self.session_state.trading_active = True
                    st.success("✅ Live trading started")
                    
                    # Start trading in background thread
                    def run_trading():
                        try:
                            run_live_trading(self.config, self.data)
                        except Exception as e:
                            st.error(f"Trading error: {e}")
                    
                    trading_thread = threading.Thread(target=run_trading)
                    trading_thread.daemon = True
                    trading_thread.start()
                else:
                    st.warning("⚠️ Please configure Alpaca connection first")
            
            if st.button("Stop Live Trading"):
                self.session_state.trading_active = False
                st.success("✅ Live trading stopped")
    
    def portfolio_monitoring_panel(self):
        """Portfolio monitoring and analytics panel"""
        st.header("📊 Portfolio Monitoring")
        
        if not self.alpaca_broker:
            st.warning("⚠️ Connect to Alpaca to view portfolio")
            return
        
        try:
            # Portfolio overview
            account_info = self.alpaca_broker.get_account_info()
            if account_info:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="Total Value",
                        value=f"${float(account_info['portfolio_value']):,.2f}"
                    )
                
                with col2:
                    st.metric(
                        label="Cash",
                        value=f"${float(account_info['cash']):,.2f}"
                    )
                
                with col3:
                    st.metric(
                        label="Buying Power",
                        value=f"${float(account_info['buying_power']):,.2f}"
                    )
                
                with col4:
                    equity = float(account_info['equity'])
                    portfolio_value = float(account_info['portfolio_value'])
                    pnl = equity - portfolio_value
                    st.metric(
                        label="P&L",
                        value=f"${pnl:,.2f}",
                        delta=f"{pnl:,.2f}"
                    )
            
            # Positions table
            positions = self.alpaca_broker.get_positions()
            if positions:
                st.subheader("Current Positions")
                
                positions_df = pd.DataFrame(positions)
                if not positions_df.empty:
                    # Calculate additional metrics
                    positions_df['market_value'] = positions_df['quantity'].astype(float) * positions_df['current_price'].astype(float)
                    positions_df['unrealized_pl'] = positions_df['unrealized_pl'].astype(float)
                    positions_df['unrealized_plpc'] = positions_df['unrealized_plpc'].astype(float)
                    
                    # Display positions
                    st.dataframe(
                        positions_df[['symbol', 'quantity', 'current_price', 'market_value', 'unrealized_pl', 'unrealized_plpc']],
                        use_container_width=True
                    )
                    
                    # Position chart
                    fig = px.pie(
                        positions_df, 
                        values='market_value', 
                        names='symbol',
                        title="Portfolio Allocation"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No positions found")
            else:
                st.info("No current positions")
                
        except Exception as e:
            st.error(f"Error fetching portfolio data: {e}")
    
    def run(self):
        """Main UI application"""
        # Header
        st.markdown('<h1 class="main-header">🤖 Algorithmic Trading System</h1>', unsafe_allow_html=True)
        
        # Load configuration
        if self.load_configuration():
            self.config = load_config('config.yaml')
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Select Page",
            ["Dashboard", "Data Ingestion", "Alpaca Integration", "FinRL Training", "Trading Controls", "Portfolio Monitoring"]
        )
        
        # Display system status
        self.display_system_status()
        
        # Page routing
        if page == "Dashboard":
            st.header("📊 Dashboard")
            
            if self.config:
                st.subheader("System Configuration")
                config_col1, config_col2 = st.columns(2)
                
                with config_col1:
                    st.write(f"**Symbol:** {self.config['trading']['symbol']}")
                    st.write(f"**Capital:** ${self.config['trading']['capital']:,}")
                    st.write(f"**Timeframe:** {self.config['trading']['timeframe']}")
                
                with config_col2:
                    st.write(f"**Broker:** {self.config['execution']['broker_api']}")
                    st.write(f"**FinRL Algorithm:** {self.config['finrl']['algorithm']}")
                    st.write(f"**Risk Max Drawdown:** {self.config['risk']['max_drawdown']:.1%}")
            
            # Quick actions
            st.subheader("Quick Actions")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Load Data", type="primary"):
                    if self.config:
                        with st.spinner("Loading data..."):
                            self.data = load_data(self.config)
                            if self.data is not None:
                                st.success("✅ Data loaded successfully")
            
            with col2:
                if st.button("Connect Alpaca"):
                    if self.config and self.config['execution']['broker_api'] in ['alpaca_paper', 'alpaca_live']:
                        with st.spinner("Connecting..."):
                            self.alpaca_broker = AlpacaBroker(self.config)
                            st.success("✅ Connected to Alpaca")
            
            with col3:
                if st.button("Start Training"):
                    if self.data is not None:
                        st.info("Navigate to FinRL Training page to configure and start training")
        
        elif page == "Data Ingestion":
            self.data_ingestion_panel()
        
        elif page == "Alpaca Integration":
            self.alpaca_integration_panel()
        
        elif page == "FinRL Training":
            self.finrl_training_panel()
        
        elif page == "Trading Controls":
            self.trading_controls_panel()
        
        elif page == "Portfolio Monitoring":
            self.portfolio_monitoring_panel()

def main():
    """Main application entry point"""
    ui = TradingUI()
    ui.run()

def create_streamlit_app():
    """Create and return a Streamlit trading application"""
    return TradingUI()

if __name__ == "__main__":
    main() 