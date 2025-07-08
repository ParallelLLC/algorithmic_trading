"""
Streamlit Cloud Deployment Entry Point
This file serves as the main entry point for Streamlit Cloud deployment
"""

import streamlit as st
import sys
import os

# Add the ui directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ui'))

# Import and run the main Streamlit app
from ui.streamlit_app import main

if __name__ == "__main__":
    main() 