#!/usr/bin/env python3
"""
UI Launcher for Algorithmic Trading System

Provides multiple UI options:
- Streamlit: Quick prototyping and data science workflows
- Dash: Enterprise-grade interactive dashboards
- Jupyter: Notebook-based interfaces
- WebSocket: Real-time trading interfaces
"""

import argparse
import sys
import os
import subprocess
import webbrowser
import time
import threading
from typing import Optional

def check_dependencies():
    """Check if required UI dependencies are installed"""
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
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def launch_streamlit():
    """Launch Streamlit application"""
    print("🚀 Launching Streamlit UI...")
    
    # Create streamlit app file if it doesn't exist
    streamlit_app_path = "ui/streamlit_app.py"
    if not os.path.exists(streamlit_app_path):
        print(f"❌ Streamlit app not found at {streamlit_app_path}")
        return False
    
    try:
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            streamlit_app_path,
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
        return True
        
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        return False

def launch_dash():
    """Launch Dash application"""
    print("🚀 Launching Dash UI...")
    
    # Create dash app file if it doesn't exist
    dash_app_path = "ui/dash_app.py"
    if not os.path.exists(dash_app_path):
        print(f"❌ Dash app not found at {dash_app_path}")
        return False
    
    try:
        # Launch Dash
        cmd = [
            sys.executable, dash_app_path
        ]
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
        return True
        
    except Exception as e:
        print(f"❌ Error launching Dash: {e}")
        return False

def launch_jupyter():
    """Launch Jupyter interface"""
    print("🚀 Launching Jupyter UI...")
    
    try:
        # Launch Jupyter Lab
        cmd = [
            sys.executable, "-m", "jupyter", "lab",
            "--port", "8888",
            "--ip", "0.0.0.0",
            "--no-browser"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
        return True
        
    except Exception as e:
        print(f"❌ Error launching Jupyter: {e}")
        return False

def launch_websocket_server():
    """Launch WebSocket server"""
    print("🚀 Launching WebSocket Server...")
    
    try:
        from ui.websocket_server import create_websocket_server
        
        server = create_websocket_server(host="0.0.0.0", port=8765)
        server_thread = server.run_server()
        
        print("✅ WebSocket server started on ws://0.0.0.0:8765")
        print("Press Ctrl+C to stop the server")
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping WebSocket server...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error launching WebSocket server: {e}")
        return False

def open_browser(url: str, delay: int = 2):
    """Open browser after delay"""
    def open_url():
        time.sleep(delay)
        try:
            webbrowser.open(url)
            print(f"🌐 Opened browser to: {url}")
        except Exception as e:
            print(f"⚠️ Could not open browser: {e}")
    
    browser_thread = threading.Thread(target=open_url)
    browser_thread.daemon = True
    browser_thread.start()

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="UI Launcher for Algorithmic Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ui_launcher.py streamlit    # Launch Streamlit UI
  python ui_launcher.py dash         # Launch Dash UI
  python ui_launcher.py jupyter      # Launch Jupyter Lab
  python ui_launcher.py websocket    # Launch WebSocket server
  python ui_launcher.py all          # Launch all UIs
        """
    )
    
    parser.add_argument(
        "ui_type",
        choices=["streamlit", "dash", "jupyter", "websocket", "all"],
        help="Type of UI to launch"
    )
    
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open browser"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        help="Custom port number (overrides default)"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("🤖 Algorithmic Trading System - UI Launcher")
    print("=" * 50)
    
    success = False
    
    if args.ui_type == "streamlit":
        success = launch_streamlit()
        if success and not args.no_browser:
            open_browser("http://localhost:8501")
    
    elif args.ui_type == "dash":
        success = launch_dash()
        if success and not args.no_browser:
            open_browser("http://localhost:8050")
    
    elif args.ui_type == "jupyter":
        success = launch_jupyter()
        if success and not args.no_browser:
            open_browser("http://localhost:8888")
    
    elif args.ui_type == "websocket":
        success = launch_websocket_server()
    
    elif args.ui_type == "all":
        print("🚀 Launching all UI interfaces...")
        
        # Launch WebSocket server in background
        websocket_thread = threading.Thread(target=launch_websocket_server)
        websocket_thread.daemon = True
        websocket_thread.start()
        
        # Launch Streamlit
        streamlit_thread = threading.Thread(target=launch_streamlit)
        streamlit_thread.daemon = True
        streamlit_thread.start()
        
        # Launch Dash
        dash_thread = threading.Thread(target=launch_dash)
        dash_thread.daemon = True
        dash_thread.start()
        
        # Launch Jupyter
        jupyter_thread = threading.Thread(target=launch_jupyter)
        jupyter_thread.daemon = True
        jupyter_thread.start()
        
        if not args.no_browser:
            open_browser("http://localhost:8501", 3)  # Streamlit
            open_browser("http://localhost:8050", 5)  # Dash
            open_browser("http://localhost:8888", 7)  # Jupyter
        
        print("✅ All UIs launched!")
        print("📊 Streamlit: http://localhost:8501")
        print("📈 Dash: http://localhost:8050")
        print("📓 Jupyter: http://localhost:8888")
        print("🔌 WebSocket: ws://localhost:8765")
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping all UIs...")
        
        success = True
    
    if success:
        print("✅ UI launched successfully!")
    else:
        print("❌ Failed to launch UI")
        sys.exit(1)

if __name__ == "__main__":
    main() 