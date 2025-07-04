"""
Test UI integration for the algorithmic trading system
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_ui_imports():
    """Test that UI modules can be imported"""
    try:
        from ui import create_streamlit_app, create_dash_app, create_jupyter_interface, TradingWebSocketServer
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import UI modules: {e}")

def test_streamlit_app_creation():
    """Test Streamlit app creation"""
    try:
        from ui.streamlit_app import TradingUI
        ui = TradingUI()
        assert ui is not None
        assert hasattr(ui, 'config')
        assert hasattr(ui, 'data')
        assert hasattr(ui, 'alpaca_broker')
    except Exception as e:
        pytest.fail(f"Failed to create Streamlit UI: {e}")

def test_dash_app_creation():
    """Test Dash app creation"""
    try:
        from ui.dash_app import TradingDashApp
        app = TradingDashApp()
        assert app is not None
        assert hasattr(app, 'app')
        assert hasattr(app, 'config')
    except Exception as e:
        pytest.fail(f"Failed to create Dash app: {e}")

def test_jupyter_ui_creation():
    """Test Jupyter UI creation"""
    try:
        from ui.jupyter_widgets import TradingJupyterUI
        ui = TradingJupyterUI()
        assert ui is not None
        assert hasattr(ui, 'config')
        assert hasattr(ui, 'data')
    except Exception as e:
        pytest.fail(f"Failed to create Jupyter UI: {e}")

def test_websocket_server_creation():
    """Test WebSocket server creation"""
    try:
        from ui.websocket_server import TradingWebSocketServer
        server = TradingWebSocketServer(host="localhost", port=8765)
        assert server is not None
        assert server.host == "localhost"
        assert server.port == 8765
        assert hasattr(server, 'clients')
    except Exception as e:
        pytest.fail(f"Failed to create WebSocket server: {e}")

def test_ui_launcher_imports():
    """Test UI launcher imports"""
    try:
        import ui_launcher
        assert hasattr(ui_launcher, 'check_dependencies')
        assert hasattr(ui_launcher, 'launch_streamlit')
        assert hasattr(ui_launcher, 'launch_dash')
        assert hasattr(ui_launcher, 'launch_jupyter')
        assert hasattr(ui_launcher, 'launch_websocket_server')
    except Exception as e:
        pytest.fail(f"Failed to import UI launcher: {e}")

@patch('subprocess.run')
def test_ui_launcher_functions(mock_run):
    """Test UI launcher functions"""
    mock_run.return_value = MagicMock()
    
    try:
        import ui_launcher
        
        # Test dependency check
        result = ui_launcher.check_dependencies()
        assert isinstance(result, bool)
        
        # Test launcher functions (they should not raise exceptions)
        ui_launcher.launch_streamlit()
        ui_launcher.launch_dash()
        ui_launcher.launch_jupyter()
        ui_launcher.launch_websocket_server()
        
    except Exception as e:
        pytest.fail(f"Failed to test UI launcher functions: {e}")

def test_ui_configuration():
    """Test UI configuration loading"""
    try:
        from agentic_ai_system.main import load_config
        config = load_config()
        
        # Check if UI-related config can be added
        config['ui'] = {
            'streamlit': {
                'server_port': 8501,
                'server_address': "0.0.0.0"
            },
            'dash': {
                'server_port': 8050,
                'server_address': "0.0.0.0"
            }
        }
        
        assert 'ui' in config
        assert 'streamlit' in config['ui']
        assert 'dash' in config['ui']
        
    except Exception as e:
        pytest.fail(f"Failed to test UI configuration: {e}")

def test_ui_dependencies():
    """Test that UI dependencies are available"""
    required_packages = [
        'streamlit',
        'dash',
        'plotly',
        'ipywidgets'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        pytest.skip(f"Missing UI dependencies: {missing_packages}")
    else:
        assert True

if __name__ == "__main__":
    pytest.main([__file__]) 