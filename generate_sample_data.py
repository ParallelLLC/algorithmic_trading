import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate more realistic sample data
start_date = datetime(2024, 7, 1, 9, 30)
dates = [start_date + timedelta(minutes=i) for i in range(1000)]  # 1000 data points

# Generate realistic price movements
np.random.seed(42)
base_price = 150.0
prices = []
for i in range(1000):
    if i == 0:
        price = base_price
    else:
        # Add some trend and volatility
        change = np.random.normal(0, 0.5) + (0.001 * i)  # Small upward trend
        price = prices[-1] + change
    prices.append(max(price, 1))  # Ensure price doesn't go negative

# Create OHLCV data
data = []
for i, (date, price) in enumerate(zip(dates, prices)):
    # Generate realistic OHLC from base price
    volatility = 0.02
    high = price * (1 + np.random.uniform(0, volatility))
    low = price * (1 - np.random.uniform(0, volatility))
    open_price = price * (1 + np.random.uniform(-volatility/2, volatility/2))
    close_price = price * (1 + np.random.uniform(-volatility/2, volatility/2))
    volume = int(np.random.uniform(5000, 50000))
    
    data.append({
        'timestamp': date,
        'open': round(open_price, 2),
        'high': round(high, 2),
        'low': round(low, 2),
        'close': round(close_price, 2),
        'volume': volume
    })

df = pd.DataFrame(data)
df.to_csv('data/market_data.csv', index=False)
print(f'Generated {len(df)} realistic data points from {df.timestamp.min()} to {df.timestamp.max()}')
print(f'Price range: ${df.close.min():.2f} - ${df.close.max():.2f}') 