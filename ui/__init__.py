"""
UI Package for Algorithmic Trading System

This package provides multiple UI options:
- Streamlit: Quick prototyping and data science workflows
- Dash: Enterprise-grade interactive dashboards
- Jupyter: Notebook-based interfaces
- WebSocket: Real-time trading interfaces
"""

__version__ = "1.0.0"
__author__ = "Algorithmic Trading Team"

from .streamlit_app import create_streamlit_app, TradingUI
from .dash_app import create_dash_app, TradingDashApp
from .jupyter_widgets import create_jupyter_interface, TradingJupyterUI
from .websocket_server import create_websocket_server, TradingWebSocketServer

__all__ = [
    "create_streamlit_app",
    "create_dash_app", 
    "create_jupyter_interface",
    "create_websocket_server",
    "TradingUI",
    "TradingDashApp",
    "TradingJupyterUI",
    "TradingWebSocketServer"
] 