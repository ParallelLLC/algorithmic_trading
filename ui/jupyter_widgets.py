"""
Jupyter Widgets UI for Algorithmic Trading System

Interactive notebook interface for:
- Data exploration and visualization
- Strategy development and testing
- Model training and evaluation
- Real-time trading simulation
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
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

from agentic_ai_system.main import load_config
from agentic_ai_system.data_ingestion import load_data, validate_data, add_technical_indicators
from agentic_ai_system.finrl_agent import FinRLAgent, FinRLConfig
from agentic_ai_system.alpaca_broker import AlpacaBroker
from agentic_ai_system.orchestrator import run_backtest, run_live_trading

class TradingJupyterUI:
    def __init__(self):
        self.config = None
        self.data = None
        self.alpaca_broker = None
        self.finrl_agent = None
        self.trading_active = False
        
        self.setup_widgets()
    
    def setup_widgets(self):
        """Setup all interactive widgets"""
        
        # Configuration widgets
        self.config_file = widgets.Text(
            value='config.yaml',
            description='Config File:',
            style={'description_width': '120px'}
        )
        
        self.load_config_btn = widgets.Button(
            description='Load Configuration',
            button_style='primary',
            icon='cog'
        )
        
        self.config_output = widgets.Output()
        
        # Data widgets
        self.data_source = widgets.Dropdown(
            options=['csv', 'alpaca', 'synthetic'],
            value='csv',
            description='Data Source:',
            style={'description_width': '120px'}
        )
        
        self.symbol_input = widgets.Text(
            value='AAPL',
            description='Symbol:',
            style={'description_width': '120px'}
        )
        
        self.timeframe_input = widgets.Dropdown(
            options=['1m', '5m', '15m', '1h', '1d'],
            value='1m',
            description='Timeframe:',
            style={'description_width': '120px'}
        )
        
        self.load_data_btn = widgets.Button(
            description='Load Data',
            button_style='success',
            icon='database'
        )
        
        self.data_output = widgets.Output()
        
        # Alpaca widgets
        self.alpaca_api_key = widgets.Password(
            description='API Key:',
            style={'description_width': '120px'}
        )
        
        self.alpaca_secret_key = widgets.Password(
            description='Secret Key:',
            style={'description_width': '120px'}
        )
        
        self.connect_alpaca_btn = widgets.Button(
            description='Connect to Alpaca',
            button_style='info',
            icon='link'
        )
        
        self.alpaca_output = widgets.Output()
        
        # FinRL widgets
        self.finrl_algorithm = widgets.Dropdown(
            options=['PPO', 'A2C', 'DDPG', 'TD3'],
            value='PPO',
            description='Algorithm:',
            style={'description_width': '120px'}
        )
        
        self.learning_rate = widgets.FloatSlider(
            value=0.0003,
            min=0.0001,
            max=0.01,
            step=0.0001,
            description='Learning Rate:',
            style={'description_width': '120px'},
            readout_format='.4f'
        )
        
        self.training_steps = widgets.IntSlider(
            value=100000,
            min=1000,
            max=1000000,
            step=1000,
            description='Training Steps:',
            style={'description_width': '120px'}
        )
        
        self.batch_size = widgets.Dropdown(
            options=[32, 64, 128, 256],
            value=64,
            description='Batch Size:',
            style={'description_width': '120px'}
        )
        
        self.start_training_btn = widgets.Button(
            description='Start Training',
            button_style='warning',
            icon='play'
        )
        
        self.finrl_output = widgets.Output()
        
        # Trading widgets
        self.capital_input = widgets.IntText(
            value=100000,
            description='Capital ($):',
            style={'description_width': '120px'}
        )
        
        self.order_size_input = widgets.IntText(
            value=10,
            description='Order Size:',
            style={'description_width': '120px'}
        )
        
        self.start_trading_btn = widgets.Button(
            description='Start Trading',
            button_style='danger',
            icon='rocket'
        )
        
        self.stop_trading_btn = widgets.Button(
            description='Stop Trading',
            button_style='danger',
            icon='stop'
        )
        
        self.trading_output = widgets.Output()
        
        # Backtesting widgets
        self.run_backtest_btn = widgets.Button(
            description='Run Backtest',
            button_style='primary',
            icon='chart-line'
        )
        
        self.backtest_output = widgets.Output()
        
        # Chart widgets
        self.chart_type = widgets.Dropdown(
            options=['Candlestick', 'Line', 'Volume', 'Technical Indicators'],
            value='Candlestick',
            description='Chart Type:',
            style={'description_width': '120px'}
        )
        
        self.chart_output = widgets.Output()
        
        # Setup callbacks
        self.load_config_btn.on_click(self.on_load_config)
        self.load_data_btn.on_click(self.on_load_data)
        self.connect_alpaca_btn.on_click(self.on_connect_alpaca)
        self.start_training_btn.on_click(self.on_start_training)
        self.start_trading_btn.on_click(self.on_start_trading)
        self.stop_trading_btn.on_click(self.on_stop_trading)
        self.run_backtest_btn.on_click(self.on_run_backtest)
        self.chart_type.observe(self.on_chart_type_change, names='value')
    
    def on_load_config(self, b):
        """Handle configuration loading"""
        with self.config_output:
            clear_output()
            try:
                self.config = load_config(self.config_file.value)
                print(f"✅ Configuration loaded from {self.config_file.value}")
                print(f"Symbol: {self.config['trading']['symbol']}")
                print(f"Capital: ${self.config['trading']['capital']:,}")
                print(f"Timeframe: {self.config['trading']['timeframe']}")
                print(f"Broker: {self.config['execution']['broker_api']}")
            except Exception as e:
                print(f"❌ Error loading configuration: {e}")
    
    def on_load_data(self, b):
        """Handle data loading"""
        with self.data_output:
            clear_output()
            try:
                if self.config:
                    # Update config with widget values
                    self.config['data_source']['type'] = self.data_source.value
                    self.config['trading']['symbol'] = self.symbol_input.value
                    self.config['trading']['timeframe'] = self.timeframe_input.value
                    
                    print(f"Loading data for {self.symbol_input.value}...")
                    self.data = load_data(self.config)
                    
                    if self.data is not None and not self.data.empty:
                        print(f"✅ Loaded {len(self.data)} data points")
                        print(f"Date range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
                        print(f"Price range: ${self.data['close'].min():.2f} - ${self.data['close'].max():.2f}")
                        
                        # Add technical indicators
                        self.data = add_technical_indicators(self.data)
                        print(f"✅ Added technical indicators")
                        
                        # Update chart
                        self.update_chart()
                    else:
                        print("❌ Failed to load data")
                else:
                    print("⚠️ Please load configuration first")
            except Exception as e:
                print(f"❌ Error loading data: {e}")
    
    def on_connect_alpaca(self, b):
        """Handle Alpaca connection"""
        with self.alpaca_output:
            clear_output()
            try:
                if self.alpaca_api_key.value and self.alpaca_secret_key.value:
                    # Update config with API keys
                    if self.config:
                        self.config['alpaca']['api_key'] = self.alpaca_api_key.value
                        self.config['alpaca']['secret_key'] = self.alpaca_secret_key.value
                        self.config['execution']['broker_api'] = 'alpaca_paper'
                        
                        print("Connecting to Alpaca...")
                        self.alpaca_broker = AlpacaBroker(self.config)
                        
                        account_info = self.alpaca_broker.get_account_info()
                        if account_info:
                            print("✅ Connected to Alpaca")
                            print(f"Account ID: {account_info['account_id']}")
                            print(f"Status: {account_info['status']}")
                            print(f"Buying Power: ${account_info['buying_power']:,.2f}")
                            print(f"Portfolio Value: ${account_info['portfolio_value']:,.2f}")
                        else:
                            print("❌ Failed to connect to Alpaca")
                    else:
                        print("⚠️ Please load configuration first")
                else:
                    print("⚠️ Please enter Alpaca API credentials")
            except Exception as e:
                print(f"❌ Error connecting to Alpaca: {e}")
    
    def on_start_training(self, b):
        """Handle FinRL training"""
        with self.finrl_output:
            clear_output()
            try:
                if self.data is not None:
                    print("Starting FinRL training...")
                    
                    # Create FinRL config
                    finrl_config = FinRLConfig(
                        algorithm=self.finrl_algorithm.value,
                        learning_rate=self.learning_rate.value,
                        batch_size=self.batch_size.value,
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
                        total_timesteps=self.training_steps.value,
                        use_real_broker=False
                    )
                    
                    if result['success']:
                        print("✅ Training completed successfully!")
                        print(f"Algorithm: {result['algorithm']}")
                        print(f"Timesteps: {result['total_timesteps']:,}")
                        print(f"Model saved: {result['model_path']}")
                    else:
                        print("❌ Training failed")
                else:
                    print("⚠️ Please load data first")
            except Exception as e:
                print(f"❌ Error during training: {e}")
    
    def on_start_trading(self, b):
        """Handle trading start"""
        with self.trading_output:
            clear_output()
            try:
                if self.config and self.alpaca_broker:
                    print("Starting live trading...")
                    self.trading_active = True
                    
                    # Update config with widget values
                    self.config['trading']['capital'] = self.capital_input.value
                    self.config['execution']['order_size'] = self.order_size_input.value
                    
                    # Start trading in background thread
                    def run_trading():
                        try:
                            run_live_trading(self.config, self.data)
                        except Exception as e:
                            print(f"Trading error: {e}")
                    
                    trading_thread = threading.Thread(target=run_trading)
                    trading_thread.daemon = True
                    trading_thread.start()
                    
                    print("✅ Live trading started")
                else:
                    print("⚠️ Please load configuration and connect to Alpaca first")
            except Exception as e:
                print(f"❌ Error starting trading: {e}")
    
    def on_stop_trading(self, b):
        """Handle trading stop"""
        with self.trading_output:
            clear_output()
            self.trading_active = False
            print("✅ Trading stopped")
    
    def on_run_backtest(self, b):
        """Handle backtesting"""
        with self.backtest_output:
            clear_output()
            try:
                if self.config and self.data is not None:
                    print("Running backtest...")
                    
                    # Update config with widget values
                    self.config['trading']['capital'] = self.capital_input.value
                    
                    result = run_backtest(self.config, self.data)
                    
                    if result['success']:
                        print("✅ Backtest completed")
                        print(f"Total Return: {result['total_return']:.2%}")
                        print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
                        print(f"Max Drawdown: {result['max_drawdown']:.2%}")
                        print(f"Total Trades: {result['total_trades']}")
                    else:
                        print("❌ Backtest failed")
                else:
                    print("⚠️ Please load configuration and data first")
            except Exception as e:
                print(f"❌ Error during backtest: {e}")
    
    def on_chart_type_change(self, change):
        """Handle chart type change"""
        if self.data is not None:
            self.update_chart()
    
    def update_chart(self):
        """Update the chart display"""
        with self.chart_output:
            clear_output()
            
            if self.data is None:
                return
            
            if self.chart_type.value == "Candlestick":
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
                display(fig)
            
            elif self.chart_type.value == "Line":
                fig = px.line(self.data, x='timestamp', y='close',
                             title=f"{self.config['trading']['symbol']} Price Chart")
                fig.update_layout(height=500)
                display(fig)
            
            elif self.chart_type.value == "Volume":
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
                display(fig)
            
            elif self.chart_type.value == "Technical Indicators":
                fig = go.Figure()
                
                # Add price
                fig.add_trace(go.Scatter(
                    x=self.data['timestamp'],
                    y=self.data['close'],
                    name='Close Price',
                    line=dict(color='blue')
                ))
                
                # Add moving averages if available
                if 'sma_20' in self.data.columns:
                    fig.add_trace(go.Scatter(
                        x=self.data['timestamp'],
                        y=self.data['sma_20'],
                        name='SMA 20',
                        line=dict(color='orange')
                    ))
                
                if 'sma_50' in self.data.columns:
                    fig.add_trace(go.Scatter(
                        x=self.data['timestamp'],
                        y=self.data['sma_50'],
                        name='SMA 50',
                        line=dict(color='red')
                    ))
                
                fig.update_layout(
                    title=f"{self.config['trading']['symbol']} Technical Indicators",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=500
                )
                display(fig)
    
    def display_interface(self):
        """Display the complete Jupyter interface"""
        
        # Header
        display(HTML("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1>🤖 Algorithmic Trading System</h1>
            <p>Interactive Jupyter Interface for Trading Analysis</p>
        </div>
        """))
        
        # Configuration section
        display(HTML("<h2>⚙️ Configuration</h2>"))
        config_widgets = widgets.VBox([
            widgets.HBox([self.config_file, self.load_config_btn]),
            self.config_output
        ])
        display(config_widgets)
        
        # Data section
        display(HTML("<h2>📥 Data Management</h2>"))
        data_widgets = widgets.VBox([
            widgets.HBox([self.data_source, self.symbol_input, self.timeframe_input]),
            widgets.HBox([self.load_data_btn]),
            self.data_output
        ])
        display(data_widgets)
        
        # Alpaca section
        display(HTML("<h2>🏦 Alpaca Integration</h2>"))
        alpaca_widgets = widgets.VBox([
            widgets.HBox([self.alpaca_api_key, self.alpaca_secret_key]),
            widgets.HBox([self.connect_alpaca_btn]),
            self.alpaca_output
        ])
        display(alpaca_widgets)
        
        # FinRL section
        display(HTML("<h2>🧠 FinRL Training</h2>"))
        finrl_widgets = widgets.VBox([
            widgets.HBox([self.finrl_algorithm, self.learning_rate]),
            widgets.HBox([self.training_steps, self.batch_size]),
            widgets.HBox([self.start_training_btn]),
            self.finrl_output
        ])
        display(finrl_widgets)
        
        # Trading section
        display(HTML("<h2>🎯 Trading Controls</h2>"))
        trading_widgets = widgets.VBox([
            widgets.HBox([self.capital_input, self.order_size_input]),
            widgets.HBox([self.start_trading_btn, self.stop_trading_btn]),
            self.trading_output
        ])
        display(trading_widgets)
        
        # Backtesting section
        display(HTML("<h2>📊 Backtesting</h2>"))
        backtest_widgets = widgets.VBox([
            widgets.HBox([self.run_backtest_btn]),
            self.backtest_output
        ])
        display(backtest_widgets)
        
        # Chart section
        display(HTML("<h2>📈 Data Visualization</h2>"))
        chart_widgets = widgets.VBox([
            widgets.HBox([self.chart_type]),
            self.chart_output
        ])
        display(chart_widgets)

def create_jupyter_interface():
    """Create and return the Jupyter interface"""
    ui = TradingJupyterUI()
    return ui 