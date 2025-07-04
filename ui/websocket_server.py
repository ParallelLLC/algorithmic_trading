"""
WebSocket Server for Real-time Trading Data

Provides real-time updates for:
- Market data streaming
- Trading signals
- Portfolio updates
- System alerts
"""

import asyncio
import websockets
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_ai_system.main import load_config
from agentic_ai_system.data_ingestion import load_data, add_technical_indicators
from agentic_ai_system.alpaca_broker import AlpacaBroker
from agentic_ai_system.finrl_agent import FinRLAgent, FinRLConfig

class TradingWebSocketServer:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.config = None
        self.alpaca_broker = None
        self.finrl_agent = None
        self.trading_active = False
        self.market_data = None
        self.portfolio_data = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def register(self, websocket):
        """Register a new client"""
        self.clients.add(websocket)
        self.logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        # Send initial data
        await self.send_initial_data(websocket)
    
    async def unregister(self, websocket):
        """Unregister a client"""
        self.clients.remove(websocket)
        self.logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def send_initial_data(self, websocket):
        """Send initial data to new client"""
        initial_data = {
            "type": "initial_data",
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "portfolio": self.portfolio_data,
            "trading_status": self.trading_active
        }
        await websocket.send(json.dumps(initial_data))
    
    async def broadcast(self, message):
        """Broadcast message to all connected clients"""
        if self.clients:
            message_str = json.dumps(message)
            await asyncio.gather(
                *[client.send(message_str) for client in self.clients],
                return_exceptions=True
            )
    
    async def handle_market_data(self):
        """Handle real-time market data updates"""
        while True:
            try:
                if self.config and self.alpaca_broker:
                    # Get real-time market data
                    symbol = self.config['trading']['symbol']
                    
                    # Get current price
                    current_price = await self.get_current_price(symbol)
                    
                    if current_price:
                        market_update = {
                            "type": "market_data",
                            "timestamp": datetime.now().isoformat(),
                            "symbol": symbol,
                            "price": current_price,
                            "volume": await self.get_current_volume(symbol)
                        }
                        
                        await self.broadcast(market_update)
                        self.logger.info(f"Broadcasted market data for {symbol}: ${current_price}")
                
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                self.logger.error(f"Error in market data handler: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def handle_portfolio_updates(self):
        """Handle portfolio updates"""
        while True:
            try:
                if self.alpaca_broker:
                    # Get portfolio information
                    account_info = self.alpaca_broker.get_account_info()
                    positions = self.alpaca_broker.get_positions()
                    
                    if account_info:
                        portfolio_update = {
                            "type": "portfolio_update",
                            "timestamp": datetime.now().isoformat(),
                            "account": {
                                "buying_power": float(account_info['buying_power']),
                                "portfolio_value": float(account_info['portfolio_value']),
                                "equity": float(account_info['equity']),
                                "cash": float(account_info['cash'])
                            },
                            "positions": positions if positions else []
                        }
                        
                        await self.broadcast(portfolio_update)
                        self.portfolio_data = portfolio_update
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in portfolio updates: {e}")
                await asyncio.sleep(10)  # Wait before retrying
    
    async def handle_trading_signals(self):
        """Handle trading signals from FinRL agent"""
        while True:
            try:
                if self.trading_active and self.finrl_agent and self.market_data is not None:
                    # Generate trading signals
                    signal = await self.generate_trading_signal()
                    
                    if signal:
                        signal_update = {
                            "type": "trading_signal",
                            "timestamp": datetime.now().isoformat(),
                            "signal": signal
                        }
                        
                        await self.broadcast(signal_update)
                        self.logger.info(f"Broadcasted trading signal: {signal}")
                
                await asyncio.sleep(10)  # Generate signals every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in trading signals: {e}")
                await asyncio.sleep(30)  # Wait before retrying
    
    async def get_current_price(self, symbol):
        """Get current price for symbol"""
        try:
            if self.alpaca_broker:
                # Get latest price from Alpaca
                latest_trade = self.alpaca_broker.get_latest_trade(symbol)
                if latest_trade:
                    return float(latest_trade['p'])
            return None
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return None
    
    async def get_current_volume(self, symbol):
        """Get current volume for symbol"""
        try:
            if self.alpaca_broker:
                # Get latest trade volume
                latest_trade = self.alpaca_broker.get_latest_trade(symbol)
                if latest_trade:
                    return int(latest_trade['s'])
            return None
        except Exception as e:
            self.logger.error(f"Error getting current volume: {e}")
            return None
    
    async def generate_trading_signal(self):
        """Generate trading signal using FinRL agent"""
        try:
            if self.finrl_agent and self.market_data is not None:
                # Use recent data for prediction
                recent_data = self.market_data.tail(100)
                
                prediction_result = self.finrl_agent.predict(
                    data=recent_data,
                    config=self.config,
                    use_real_broker=False
                )
                
                if prediction_result['success']:
                    # Generate signal based on prediction
                    current_price = await self.get_current_price(self.config['trading']['symbol'])
                    
                    if current_price:
                        signal = {
                            "action": "HOLD",  # Default action
                            "confidence": 0.5,
                            "price": current_price,
                            "reasoning": "Model prediction"
                        }
                        
                        # Determine action based on prediction
                        if prediction_result['total_return'] > 0.02:  # 2% positive return
                            signal["action"] = "BUY"
                            signal["confidence"] = min(0.9, 0.5 + abs(prediction_result['total_return']))
                        elif prediction_result['total_return'] < -0.02:  # 2% negative return
                            signal["action"] = "SELL"
                            signal["confidence"] = min(0.9, 0.5 + abs(prediction_result['total_return']))
                        
                        return signal
            
            return None
        except Exception as e:
            self.logger.error(f"Error generating trading signal: {e}")
            return None
    
    async def handle_client_message(self, websocket, message):
        """Handle incoming client messages"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "load_config":
                # Load configuration
                config_file = data.get("config_file", "config.yaml")
                self.config = load_config(config_file)
                
                response = {
                    "type": "config_loaded",
                    "success": True,
                    "config": self.config
                }
                await websocket.send(json.dumps(response))
            
            elif message_type == "connect_alpaca":
                # Connect to Alpaca
                api_key = data.get("api_key")
                secret_key = data.get("secret_key")
                
                if api_key and secret_key:
                    self.config['alpaca']['api_key'] = api_key
                    self.config['alpaca']['secret_key'] = secret_key
                    self.config['execution']['broker_api'] = 'alpaca_paper'
                    
                    self.alpaca_broker = AlpacaBroker(self.config)
                    
                    response = {
                        "type": "alpaca_connected",
                        "success": True
                    }
                    await websocket.send(json.dumps(response))
                else:
                    response = {
                        "type": "alpaca_connected",
                        "success": False,
                        "error": "Missing API credentials"
                    }
                    await websocket.send(json.dumps(response))
            
            elif message_type == "start_trading":
                # Start trading
                self.trading_active = True
                
                response = {
                    "type": "trading_started",
                    "success": True
                }
                await websocket.send(json.dumps(response))
                
                # Broadcast to all clients
                await self.broadcast({
                    "type": "trading_status",
                    "active": True,
                    "timestamp": datetime.now().isoformat()
                })
            
            elif message_type == "stop_trading":
                # Stop trading
                self.trading_active = False
                
                response = {
                    "type": "trading_stopped",
                    "success": True
                }
                await websocket.send(json.dumps(response))
                
                # Broadcast to all clients
                await self.broadcast({
                    "type": "trading_status",
                    "active": False,
                    "timestamp": datetime.now().isoformat()
                })
            
            elif message_type == "load_data":
                # Load market data
                if self.config:
                    self.market_data = load_data(self.config)
                    if self.market_data is not None:
                        self.market_data = add_technical_indicators(self.market_data)
                        
                        response = {
                            "type": "data_loaded",
                            "success": True,
                            "data_points": len(self.market_data)
                        }
                    else:
                        response = {
                            "type": "data_loaded",
                            "success": False,
                            "error": "Failed to load data"
                        }
                else:
                    response = {
                        "type": "data_loaded",
                        "success": False,
                        "error": "Configuration not loaded"
                    }
                
                await websocket.send(json.dumps(response))
            
            elif message_type == "train_model":
                # Train FinRL model
                if self.market_data is not None:
                    algorithm = data.get("algorithm", "PPO")
                    learning_rate = data.get("learning_rate", 0.0003)
                    training_steps = data.get("training_steps", 100000)
                    
                    finrl_config = FinRLConfig(
                        algorithm=algorithm,
                        learning_rate=learning_rate,
                        batch_size=64,
                        buffer_size=1000000,
                        learning_starts=100,
                        gamma=0.99,
                        tau=0.005,
                        train_freq=1,
                        gradient_steps=1,
                        verbose=1,
                        tensorboard_log='logs/finrl_tensorboard'
                    )
                    
                    self.finrl_agent = FinRLAgent(finrl_config)
                    
                    # Train in background thread
                    def train_model():
                        try:
                            result = self.finrl_agent.train(
                                data=self.market_data,
                                config=self.config,
                                total_timesteps=training_steps,
                                use_real_broker=False
                            )
                            
                            # Broadcast training completion
                            asyncio.create_task(self.broadcast({
                                "type": "training_completed",
                                "success": result['success'],
                                "result": result
                            }))
                        except Exception as e:
                            asyncio.create_task(self.broadcast({
                                "type": "training_completed",
                                "success": False,
                                "error": str(e)
                            }))
                    
                    training_thread = threading.Thread(target=train_model)
                    training_thread.daemon = True
                    training_thread.start()
                    
                    response = {
                        "type": "training_started",
                        "success": True
                    }
                else:
                    response = {
                        "type": "training_started",
                        "success": False,
                        "error": "Market data not loaded"
                    }
                
                await websocket.send(json.dumps(response))
            
            else:
                # Unknown message type
                response = {
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                }
                await websocket.send(json.dumps(response))
                
        except json.JSONDecodeError:
            response = {
                "type": "error",
                "message": "Invalid JSON message"
            }
            await websocket.send(json.dumps(response))
        except Exception as e:
            response = {
                "type": "error",
                "message": f"Server error: {str(e)}"
            }
            await websocket.send(json.dumps(response))
    
    async def websocket_handler(self, websocket, path):
        """Main WebSocket handler"""
        await self.register(websocket)
        try:
            async for message in websocket:
                await self.handle_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)
    
    async def start_server(self):
        """Start the WebSocket server"""
        # Start background tasks
        asyncio.create_task(self.handle_market_data())
        asyncio.create_task(self.handle_portfolio_updates())
        asyncio.create_task(self.handle_trading_signals())
        
        # Start WebSocket server
        server = await websockets.serve(
            self.websocket_handler,
            self.host,
            self.port
        )
        
        self.logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
        
        # Keep server running
        await server.wait_closed()
    
    def run_server(self):
        """Run the server in a separate thread"""
        def run():
            asyncio.run(self.start_server())
        
        server_thread = threading.Thread(target=run)
        server_thread.daemon = True
        server_thread.start()
        
        return server_thread

def create_websocket_server(host="localhost", port=8765):
    """Create and return a WebSocket server instance"""
    return TradingWebSocketServer(host=host, port=port) 