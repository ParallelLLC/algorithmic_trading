# UI Integration Guide

This guide covers the comprehensive UI system for the Algorithmic Trading project, providing multiple interface options for different use cases.

## 🎯 UI Options Overview

### 1. **Streamlit UI** - Quick Prototyping
- **Best for**: Data scientists, quick experiments, rapid prototyping
- **Features**: Interactive widgets, real-time data visualization, easy configuration
- **Port**: 8501
- **URL**: http://localhost:8501

### 2. **Dash UI** - Enterprise Dashboards
- **Best for**: Production dashboards, real-time monitoring, complex analytics
- **Features**: Advanced charts, real-time updates, professional styling
- **Port**: 8050
- **URL**: http://localhost:8050

### 3. **Jupyter UI** - Interactive Notebooks
- **Best for**: Research, experimentation, educational purposes
- **Features**: Interactive widgets, code execution, rich documentation
- **Port**: 8888
- **URL**: http://localhost:8888

### 4. **WebSocket Server** - Real-time Data
- **Best for**: Real-time trading signals, live data streaming
- **Features**: WebSocket API, real-time updates, trading signals
- **Port**: 8765
- **URL**: ws://localhost:8765

## 🚀 Quick Start

### Prerequisites
```bash
# Install UI dependencies
pip install -r requirements.txt

# Verify installation
python -c "import streamlit, dash, plotly, ipywidgets; print('✅ All UI dependencies installed')"
```

### Launch Individual UIs

#### Streamlit (Recommended for beginners)
```bash
python ui_launcher.py streamlit
```

#### Dash (Recommended for production)
```bash
python ui_launcher.py dash
```

#### Jupyter Lab
```bash
python ui_launcher.py jupyter
```

#### WebSocket Server
```bash
python ui_launcher.py websocket
```

#### Launch All UIs
```bash
python ui_launcher.py all
```

## 📊 Streamlit UI Features

### Dashboard
- **System Status**: Real-time trading status, portfolio value, P&L
- **Configuration Management**: Load and modify trading parameters
- **Quick Actions**: One-click data loading, Alpaca connection, model training

### Data Ingestion
- **Multiple Sources**: CSV, Alpaca API, Synthetic data
- **Data Validation**: Automatic data quality checks
- **Technical Indicators**: Automatic calculation of moving averages, RSI, MACD
- **Interactive Charts**: Candlestick, line, volume charts with Plotly

### Alpaca Integration
- **Account Connection**: Secure API key management
- **Market Status**: Real-time market hours and status
- **Position Monitoring**: Current positions and portfolio value
- **Order Management**: Buy/sell order execution

### FinRL Training
- **Algorithm Selection**: PPO, A2C, DDPG, TD3
- **Hyperparameter Tuning**: Learning rate, batch size, training steps
- **Training Progress**: Real-time training metrics and progress
- **Model Evaluation**: Performance metrics and backtesting

### Trading Controls
- **Live Trading**: Start/stop live trading with Alpaca
- **Backtesting**: Historical strategy testing
- **Risk Management**: Position sizing and drawdown limits
- **Emergency Stop**: Immediate trading halt

### Portfolio Monitoring
- **Real-time Portfolio**: Live portfolio value and P&L
- **Position Analysis**: Individual position performance
- **Allocation Charts**: Portfolio allocation visualization
- **Risk Metrics**: Sharpe ratio, drawdown analysis

## 📈 Dash UI Features

### Enterprise Dashboard
- **Professional Styling**: Bootstrap themes and responsive design
- **Real-time Updates**: Live data streaming and updates
- **Advanced Charts**: Interactive Plotly charts with zoom, pan, hover
- **Multi-page Navigation**: Tabbed interface for different functions

### Advanced Analytics
- **Technical Analysis**: Advanced charting with indicators
- **Performance Metrics**: Comprehensive trading performance analysis
- **Risk Management**: Advanced risk monitoring and alerts
- **Strategy Comparison**: Multiple strategy backtesting and comparison

### Real-time Monitoring
- **Live Trading Activity**: Real-time trade execution monitoring
- **System Alerts**: Automated alerts for important events
- **Portfolio Tracking**: Live portfolio updates and analysis
- **Market Data**: Real-time market data visualization

## 📓 Jupyter UI Features

### Interactive Development
- **Widget-based Interface**: Interactive controls for all functions
- **Code Execution**: Direct Python code execution and experimentation
- **Data Exploration**: Interactive data analysis and visualization
- **Model Development**: Iterative model training and testing

### Research Tools
- **Notebook Integration**: Rich documentation and code examples
- **Data Analysis**: Pandas and NumPy integration
- **Visualization**: Matplotlib, Seaborn, Plotly integration
- **Experiment Tracking**: Training history and model comparison

## 🔌 WebSocket API

### Real-time Data Streaming
```javascript
// Connect to WebSocket server
const ws = new WebSocket('ws://localhost:8765');

// Listen for market data updates
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.type === 'market_data') {
        console.log('Price:', data.price);
        console.log('Volume:', data.volume);
    }
    
    if (data.type === 'trading_signal') {
        console.log('Signal:', data.signal);
    }
    
    if (data.type === 'portfolio_update') {
        console.log('Portfolio:', data.account);
    }
};
```

### Available Message Types
- `market_data`: Real-time price and volume data
- `trading_signal`: FinRL model trading signals
- `portfolio_update`: Account and position updates
- `trading_status`: Trading system status
- `system_alert`: System alerts and notifications

## 🛠️ Configuration

### Environment Variables
```bash
# Alpaca API credentials
export ALPACA_API_KEY="your_api_key"
export ALPACA_SECRET_KEY="your_secret_key"

# UI configuration
export STREAMLIT_SERVER_PORT=8501
export DASH_SERVER_PORT=8050
export JUPYTER_PORT=8888
export WEBSOCKET_PORT=8765
```

### Configuration File
```yaml
# config.yaml
ui:
  streamlit:
    server_port: 8501
    server_address: "0.0.0.0"
    theme: "light"
  
  dash:
    server_port: 8050
    server_address: "0.0.0.0"
    theme: "bootstrap"
  
  jupyter:
    port: 8888
    ip: "0.0.0.0"
    token: ""
  
  websocket:
    host: "0.0.0.0"
    port: 8765
    max_connections: 100
```

## 🔧 Customization

### Adding Custom Charts
```python
# In ui/streamlit_app.py
def create_custom_chart(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=data['custom_indicator'],
        name='Custom Indicator'
    ))
    return fig
```

### Custom Trading Strategies
```python
# In ui/dash_app.py
def custom_strategy(data, config):
    # Implement your custom strategy
    signals = []
    for i in range(len(data)):
        if data['sma_20'][i] > data['sma_50'][i]:
            signals.append('BUY')
        else:
            signals.append('SELL')
    return signals
```

### WebSocket Custom Messages
```python
# In ui/websocket_server.py
async def broadcast_custom_message(self, message_type, data):
    message = {
        "type": message_type,
        "timestamp": datetime.now().isoformat(),
        "data": data
    }
    await self.broadcast(message)
```

## 🚀 Deployment

### Docker Deployment
```bash
# Build UI-enabled Docker image
docker build -t trading-ui .

# Run with UI ports exposed
docker run -p 8501:8501 -p 8050:8050 -p 8888:8888 -p 8765:8765 trading-ui
```

### Production Deployment
```bash
# Using Gunicorn for production
pip install gunicorn

# Start Dash app with Gunicorn
gunicorn -w 4 -b 0.0.0.0:8050 ui.dash_app:app

# Start Streamlit with production settings
streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

### Cloud Deployment
```bash
# Deploy to Heroku
heroku create trading-ui-app
git push heroku main

# Deploy to AWS
aws ecs create-service --cluster trading-cluster --service-name trading-ui
```

## 🔍 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port
lsof -i :8501

# Kill process
kill -9 <PID>

# Or use different port
python ui_launcher.py streamlit --port 8502
```

#### Missing Dependencies
```bash
# Install missing packages
pip install streamlit dash plotly ipywidgets

# Or reinstall all requirements
pip install -r requirements.txt
```

#### Alpaca Connection Issues
```bash
# Check API credentials
echo $ALPACA_API_KEY
echo $ALPACA_SECRET_KEY

# Test connection
python -c "from agentic_ai_system.alpaca_broker import AlpacaBroker; print('Connection test')"
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with debug output
python ui_launcher.py streamlit --debug
```

## 📚 API Reference

### Streamlit Functions
- `create_streamlit_app()`: Create Streamlit application
- `TradingUI.run()`: Run the main UI application
- `load_configuration()`: Load trading configuration
- `display_system_status()`: Show system status

### Dash Functions
- `create_dash_app()`: Create Dash application
- `TradingDashApp.setup_layout()`: Setup dashboard layout
- `TradingDashApp.setup_callbacks()`: Setup interactive callbacks

### Jupyter Functions
- `create_jupyter_interface()`: Create Jupyter interface
- `TradingJupyterUI.display_interface()`: Display interactive widgets
- `TradingJupyterUI.update_chart()`: Update chart displays

### WebSocket Functions
- `create_websocket_server()`: Create WebSocket server
- `TradingWebSocketServer.broadcast()`: Broadcast messages
- `TradingWebSocketServer.handle_client_message()`: Handle client messages

## 🤝 Contributing

### Adding New UI Features
1. Create feature branch: `git checkout -b feature/new-ui-feature`
2. Implement feature in appropriate UI module
3. Add tests in `tests/ui/` directory
4. Update documentation
5. Submit pull request

### UI Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints for all functions
- Include docstrings for all classes and methods
- Write unit tests for new features
- Update documentation for new features

## 📞 Support

For UI-related issues:
1. Check the troubleshooting section
2. Review the logs in `logs/ui/` directory
3. Create an issue on GitHub with detailed error information
4. Include system information and error logs

## 🔄 Updates

### UI Version History
- **v1.0.0**: Initial UI implementation with Streamlit, Dash, Jupyter, and WebSocket
- **v1.1.0**: Added real-time data streaming and advanced charts
- **v1.2.0**: Enhanced portfolio monitoring and risk management
- **v1.3.0**: Added custom strategy development tools

### Upcoming Features
- **v1.4.0**: Machine learning model visualization
- **v1.5.0**: Advanced backtesting interface
- **v1.6.0**: Multi-asset portfolio management
- **v1.7.0**: Social trading features 