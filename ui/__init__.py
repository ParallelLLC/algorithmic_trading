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

from .streamlit_app import create_streamlit_app
from .dash_app import create_dash_app
from .jupyter_widgets import create_jupyter_interface
from .websocket_server import TradingWebSocketServer

__all__ = [
    "create_streamlit_app",
    "create_dash_app", 
    "create_jupyter_interface",
    "TradingWebSocketServer"
] 