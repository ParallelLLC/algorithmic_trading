"""
Dash UI for Algorithmic Trading System

Enterprise-grade interactive dashboard with:
- Real-time market data visualization
- Advanced trading analytics
- Portfolio management
- Risk monitoring
- Strategy backtesting
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from dash_bootstrap_components import themes
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import yaml
import os
import sys
from datetime import datetime, timedelta
import asyncio
import threading
import time
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_ai_system.main import load_config
from agentic_ai_system.data_ingestion import load_data, validate_data, add_technical_indicators
from agentic_ai_system.finrl_agent import FinRLAgent, FinRLConfig
from agentic_ai_system.alpaca_broker import AlpacaBroker
from agentic_ai_system.orchestrator import run_backtest, run_live_trading

class TradingDashApp:
    def __init__(self):
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[
                themes.BOOTSTRAP,
                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
            ],
            suppress_callback_exceptions=True
        )
        
        self.config = None
        self.data = None
        self.alpaca_broker = None
        self.finrl_agent = None
        
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup the main application layout"""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1([
                        html.I(className="fas fa-chart-line me-3"),
                        "Algorithmic Trading System"
                    ], className="text-primary mb-4 text-center")
                ])
            ]),
            
            # Navigation tabs
            dbc.Tabs([
                dbc.Tab(self.create_dashboard_tab(), label="Dashboard", tab_id="dashboard"),
                dbc.Tab(self.create_data_tab(), label="Data", tab_id="data"),
                dbc.Tab(self.create_trading_tab(), label="Trading", tab_id="trading"),
                dbc.Tab(self.create_analytics_tab(), label="Analytics", tab_id="analytics"),
                dbc.Tab(self.create_portfolio_tab(), label="Portfolio", tab_id="portfolio"),
                dbc.Tab(self.create_settings_tab(), label="Settings", tab_id="settings")
            ], id="tabs", active_tab="dashboard"),
            
            # Store components for data persistence
            dcc.Store(id="config-store"),
            dcc.Store(id="data-store"),
            dcc.Store(id="alpaca-store"),
            dcc.Store(id="finrl-store"),
            dcc.Store(id="trading-status-store"),
            
            # Interval for real-time updates
            dcc.Interval(
                id="interval-component",
                interval=5*1000,  # 5 seconds
                n_intervals=0
            )
        ], fluid=True)
    
    def create_dashboard_tab(self):
        """Create the main dashboard tab"""
        return dbc.Container([
            # System status cards
            dbc.Row([
                dbc.Col(self.create_status_card("Trading Status", "Active", "success"), width=3),
                dbc.Col(self.create_status_card("Portfolio Value", "$100,000", "info"), width=3),
                dbc.Col(self.create_status_card("Daily P&L", "+$1,250", "success"), width=3),
                dbc.Col(self.create_status_card("Risk Level", "Low", "warning"), width=3)
            ], className="mb-4"),
            
            # Charts row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Price Chart"),
                        dbc.CardBody([
                            dcc.Graph(id="price-chart", style={"height": "400px"})
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Portfolio Allocation"),
                        dbc.CardBody([
                            dcc.Graph(id="allocation-chart", style={"height": "400px"})
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            # Trading activity and alerts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Recent Trades"),
                        dbc.CardBody([
                            html.Div(id="trades-table")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("System Alerts"),
                        dbc.CardBody([
                            html.Div(id="alerts-list")
                        ])
                    ])
                ], width=6)
            ])
        ])
    
    def create_data_tab(self):
        """Create the data management tab"""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Data Configuration"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Data Source"),
                                    dbc.Select(
                                        id="data-source-select",
                                        options=[
                                            {"label": "CSV File", "value": "csv"},
                                            {"label": "Alpaca API", "value": "alpaca"},
                                            {"label": "Synthetic Data", "value": "synthetic"}
                                        ],
                                        value="csv"
                                    )
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Symbol"),
                                    dbc.Input(
                                        id="symbol-input",
                                        type="text",
                                        value="AAPL",
                                        placeholder="Enter symbol"
                                    )
                                ], width=4),
                                dbc.Col([
                                    dbc.Label("Timeframe"),
                                    dbc.Select(
                                        id="timeframe-select",
                                        options=[
                                            {"label": "1 Minute", "value": "1m"},
                                            {"label": "5 Minutes", "value": "5m"},
                                            {"label": "15 Minutes", "value": "15m"},
                                            {"label": "1 Hour", "value": "1h"},
                                            {"label": "1 Day", "value": "1d"}
                                        ],
                                        value="1m"
                                    )
                                ], width=4)
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Load Data", id="load-data-btn", color="primary", className="me-2"),
                                    dbc.Button("Refresh Data", id="refresh-data-btn", color="secondary")
                                ])
                            ])
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Data Statistics"),
                        dbc.CardBody([
                            html.Div(id="data-stats")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Data visualization
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.Span("Market Data Visualization"),
                            dbc.ButtonGroup([
                                dbc.Button("Candlestick", id="candlestick-btn", size="sm"),
                                dbc.Button("Line", id="line-btn", size="sm"),
                                dbc.Button("Volume", id="volume-btn", size="sm")
                            ], className="float-end")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id="market-chart", style={"height": "500px"})
                        ])
                    ])
                ])
            ])
        ])
    
    def create_trading_tab(self):
        """Create the trading controls tab"""
        return dbc.Container([
            # Trading configuration
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Trading Configuration"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Capital"),
                                    dbc.Input(
                                        id="capital-input",
                                        type="number",
                                        value=100000,
                                        step=1000
                                    )
                                ], width=3),
                                dbc.Col([
                                    dbc.Label("Order Size"),
                                    dbc.Input(
                                        id="order-size-input",
                                        type="number",
                                        value=10,
                                        step=1
                                    )
                                ], width=3),
                                dbc.Col([
                                    dbc.Label("Max Position"),
                                    dbc.Input(
                                        id="max-position-input",
                                        type="number",
                                        value=100,
                                        step=10
                                    )
                                ], width=3),
                                dbc.Col([
                                    dbc.Label("Max Drawdown"),
                                    dbc.Input(
                                        id="max-drawdown-input",
                                        type="number",
                                        value=0.05,
                                        step=0.01,
                                        min=0,
                                        max=1
                                    )
                                ], width=3)
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Start Trading", id="start-trading-btn", color="success", className="me-2"),
                                    dbc.Button("Stop Trading", id="stop-trading-btn", color="danger", className="me-2"),
                                    dbc.Button("Emergency Stop", id="emergency-stop-btn", color="warning")
                                ])
                            ])
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Alpaca Connection"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("API Key"),
                                    dbc.Input(
                                        id="alpaca-api-key",
                                        type="password",
                                        placeholder="Enter Alpaca API key"
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Secret Key"),
                                    dbc.Input(
                                        id="alpaca-secret-key",
                                        type="password",
                                        placeholder="Enter Alpaca secret key"
                                    )
                                ], width=6)
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Connect", id="connect-alpaca-btn", color="primary", className="me-2"),
                                    dbc.Button("Disconnect", id="disconnect-alpaca-btn", color="secondary")
                                ])
                            ])
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Trading activity
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Live Trading Activity"),
                        dbc.CardBody([
                            html.Div(id="trading-activity")
                        ])
                    ])
                ])
            ])
        ])
    
    def create_analytics_tab(self):
        """Create the analytics tab"""
        return dbc.Container([
            # FinRL training
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("FinRL Model Training"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Algorithm"),
                                    dbc.Select(
                                        id="finrl-algorithm-select",
                                        options=[
                                            {"label": "PPO", "value": "PPO"},
                                            {"label": "A2C", "value": "A2C"},
                                            {"label": "DDPG", "value": "DDPG"},
                                            {"label": "TD3", "value": "TD3"}
                                        ],
                                        value="PPO"
                                    )
                                ], width=3),
                                dbc.Col([
                                    dbc.Label("Learning Rate"),
                                    dbc.Input(
                                        id="learning-rate-input",
                                        type="number",
                                        value=0.0003,
                                        step=0.0001,
                                        min=0.0001,
                                        max=0.01
                                    )
                                ], width=3),
                                dbc.Col([
                                    dbc.Label("Training Steps"),
                                    dbc.Input(
                                        id="training-steps-input",
                                        type="number",
                                        value=100000,
                                        step=1000
                                    )
                                ], width=3),
                                dbc.Col([
                                    dbc.Label("Batch Size"),
                                    dbc.Select(
                                        id="batch-size-select",
                                        options=[
                                            {"label": "32", "value": 32},
                                            {"label": "64", "value": 64},
                                            {"label": "128", "value": 128},
                                            {"label": "256", "value": 256}
                                        ],
                                        value=64
                                    )
                                ], width=3)
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Start Training", id="start-training-btn", color="primary", className="me-2"),
                                    dbc.Button("Stop Training", id="stop-training-btn", color="danger")
                                ])
                            ])
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Training Progress"),
                        dbc.CardBody([
                            dbc.Progress(id="training-progress", value=0, className="mb-3"),
                            html.Div(id="training-metrics")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Backtesting
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Strategy Backtesting"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Run Backtest", id="run-backtest-btn", color="primary", className="me-2"),
                                    dbc.Button("Export Results", id="export-backtest-btn", color="secondary")
                                ])
                            ]),
                            html.Div(id="backtest-results")
                        ])
                    ])
                ])
            ])
        ])
    
    def create_portfolio_tab(self):
        """Create the portfolio management tab"""
        return dbc.Container([
            # Portfolio overview
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Portfolio Overview"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H4("Total Value", className="text-muted"),
                                    html.H3(id="total-value", children="$100,000")
                                ], width=3),
                                dbc.Col([
                                    html.H4("Cash", className="text-muted"),
                                    html.H3(id="cash-value", children="$25,000")
                                ], width=3),
                                dbc.Col([
                                    html.H4("Invested", className="text-muted"),
                                    html.H3(id="invested-value", children="$75,000")
                                ], width=3),
                                dbc.Col([
                                    html.H4("P&L", className="text-muted"),
                                    html.H3(id="pnl-value", children="+$1,250", className="text-success")
                                ], width=3)
                            ])
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Positions and allocation
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Current Positions"),
                        dbc.CardBody([
                            html.Div(id="positions-table")
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Allocation Chart"),
                        dbc.CardBody([
                            dcc.Graph(id="portfolio-allocation-chart", style={"height": "300px"})
                        ])
                    ])
                ], width=4)
            ])
        ])
    
    def create_settings_tab(self):
        """Create the settings tab"""
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("System Configuration"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Config File"),
                                    dbc.Input(
                                        id="config-file-input",
                                        type="text",
                                        value="config.yaml",
                                        placeholder="Enter config file path"
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Log Level"),
                                    dbc.Select(
                                        id="log-level-select",
                                        options=[
                                            {"label": "DEBUG", "value": "DEBUG"},
                                            {"label": "INFO", "value": "INFO"},
                                            {"label": "WARNING", "value": "WARNING"},
                                            {"label": "ERROR", "value": "ERROR"}
                                        ],
                                        value="INFO"
                                    )
                                ], width=6)
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Load Config", id="load-config-btn", color="primary", className="me-2"),
                                    dbc.Button("Save Config", id="save-config-btn", color="success")
                                ])
                            ])
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("System Status"),
                        dbc.CardBody([
                            html.Div(id="system-status")
                        ])
                    ])
                ], width=6)
            ])
        ])
    
    def create_status_card(self, title, value, color):
        """Create a status card component"""
        return dbc.Card([
            dbc.CardBody([
                html.H5(title, className="card-title text-muted"),
                html.H3(value, className=f"text-{color}")
            ])
        ])
    
    def setup_callbacks(self):
        """Setup all Dash callbacks"""
        
        @self.app.callback(
            Output("config-store", "data"),
            Input("load-config-btn", "n_clicks"),
            State("config-file-input", "value"),
            prevent_initial_call=True
        )
        def load_configuration(n_clicks, config_file):
            if n_clicks:
                try:
                    config = load_config(config_file)
                    return config
                except Exception as e:
                    return {"error": str(e)}
            return dash.no_update
        
        @self.app.callback(
            Output("data-store", "data"),
            Input("load-data-btn", "n_clicks"),
            State("config-store", "data"),
            prevent_initial_call=True
        )
        def load_market_data(n_clicks, config):
            if n_clicks and config:
                try:
                    data = load_data(config)
                    if data is not None:
                        return data.to_dict('records')
                except Exception as e:
                    return {"error": str(e)}
            return dash.no_update
        
        @self.app.callback(
            Output("price-chart", "figure"),
            Input("data-store", "data"),
            Input("interval-component", "n_intervals")
        )
        def update_price_chart(data, n_intervals):
            if data and isinstance(data, list):
                df = pd.DataFrame(data)
                if not df.empty:
                    fig = go.Figure(data=[go.Candlestick(
                        x=df['timestamp'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close']
                    )])
                    fig.update_layout(
                        title="Market Data",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=400
                    )
                    return fig
            return go.Figure()
        
        @self.app.callback(
            Output("allocation-chart", "figure"),
            Input("alpaca-store", "data"),
            Input("interval-component", "n_intervals")
        )
        def update_allocation_chart(alpaca_data, n_intervals):
            # Mock portfolio allocation data
            labels = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'Cash']
            values = [30, 25, 20, 15, 10]
            
            fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
            fig.update_layout(
                title="Portfolio Allocation",
                height=400
            )
            return fig
        
        @self.app.callback(
            Output("trading-activity", "children"),
            Input("interval-component", "n_intervals")
        )
        def update_trading_activity(n_intervals):
            # Mock trading activity
            trades = [
                {"time": "09:30:15", "symbol": "AAPL", "action": "BUY", "quantity": 10, "price": 150.25},
                {"time": "09:35:22", "symbol": "GOOGL", "action": "SELL", "quantity": 5, "price": 2750.50},
                {"time": "09:40:08", "symbol": "MSFT", "action": "BUY", "quantity": 15, "price": 320.75}
            ]
            
            table_rows = []
            for trade in trades:
                color = "success" if trade["action"] == "BUY" else "danger"
                table_rows.append(
                    dbc.Row([
                        dbc.Col(trade["time"], width=2),
                        dbc.Col(trade["symbol"], width=2),
                        dbc.Col(trade["action"], width=2, className=f"text-{color}"),
                        dbc.Col(str(trade["quantity"]), width=2),
                        dbc.Col(f"${trade['price']:.2f}", width=2),
                        dbc.Col(f"${trade['quantity'] * trade['price']:.2f}", width=2)
                    ], className="mb-2")
                )
            
            return table_rows

def create_dash_app():
    """Create and return the Dash application"""
    app = TradingDashApp()
    return app.app

if __name__ == "__main__":
    app = create_dash_app()
    app.run_server(debug=True, host="0.0.0.0", port=8050) 