# Streamlit Cloud Deployment Guide

## Overview
This guide will help you deploy your algorithmic trading system to Streamlit Cloud.

## Files Created for Deployment

### 1. `streamlit_app.py` (Root Entry Point)
- Main entry point for Streamlit Cloud
- Imports and runs the UI from the `ui/` directory

### 2. `.streamlit/config.toml`
- Streamlit configuration for deployment
- Optimized for cloud deployment

### 3. `packages.txt`
- System dependencies for the deployment environment

### 4. Updated `requirements.txt`
- Fixed dependencies for cloud deployment
- Removed problematic packages

## Deployment Steps

### 1. Push to GitHub
```bash
git add .
git commit -m "feat: prepare for Streamlit Cloud deployment"
git push origin main
```

### 2. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Configure your app:
   - **Repository**: `your-username/algorithmic_trading`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
   - **App URL**: Choose your preferred URL

### 3. Environment Variables (Optional)
If you want to use real API keys, add these in Streamlit Cloud:
- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `ALPACA_BASE_URL`

## Deployment Features

### ✅ Deployment Mode
- The app automatically detects deployment environment
- Falls back to demo mode if modules aren't available
- Shows sample data and mock trading functionality

### ✅ Error Handling
- Graceful handling of missing dependencies
- Informative error messages
- Fallback functionality

### ✅ Performance Optimized
- Minimal dependencies
- Efficient imports
- Cloud-optimized configuration

## Troubleshooting

### Common Issues

1. **Import Errors**
   - The app now handles missing modules gracefully
   - Check the browser console for specific error messages

2. **Dependency Issues**
   - All dependencies are now properly specified in `requirements.txt`
   - System packages are included in `packages.txt`

3. **Configuration Issues**
   - The app uses a default configuration in deployment mode
   - No external files are required

### Debug Mode
To debug deployment issues:
1. Check the Streamlit Cloud logs
2. Look for error messages in the browser console
3. Verify all files are properly committed to GitHub

## Local Testing
Test the deployment version locally:
```bash
streamlit run streamlit_app.py
```

## Features Available in Deployment

### ✅ Working Features
- Configuration loading (demo mode)
- Data visualization with sample data
- Chart generation
- UI navigation
- Mock trading interface

### ⚠️ Limited Features
- Real API connections (requires environment variables)
- Live trading (demo mode only)
- Model training (simulated)

## Next Steps

1. **Deploy to Streamlit Cloud** using the steps above
2. **Test the deployment** to ensure everything works
3. **Add environment variables** for real API access
4. **Monitor the deployment** for any issues

## Support

If you encounter issues:
1. Check the Streamlit Cloud logs
2. Verify all files are committed to GitHub
3. Test locally with `streamlit run streamlit_app.py`
4. Review the error handling in the code

The deployment is now ready! 🚀 