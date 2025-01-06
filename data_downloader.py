import time
import ccxt
import requests
import pandas as pd
from datetime import datetime, timedelta

# Initialize CCXT Binance client without API keys for public data
binance = ccxt.binance({
    'rateLimit': 1200,
    'enableRateLimit': True
})

def fetch_hourly_data(symbol, start_date, end_date):
    """Fetch hourly historical klines for a specific symbol from Binance using CCXT"""
    timeframe = '1h'
    since = int(start_date.timestamp() * 1000)
    all_data = []

    while since < int(end_date.timestamp() * 1000):
        try:
            data = binance.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not data:
                break

            all_data.extend(data)
            since = data[-1][0] + 3600000  # Move to next batch

            time.sleep(binance.rateLimit / 1000)  # Avoid rate limit
        except ccxt.NetworkError as e:
            print(f"Network error for {symbol}, retrying: {e}")
            time.sleep(1)
        except ccxt.BaseError as e:
            print(f"Base error for {symbol}: {e}")
            break

    return [
        {
            "timestamp": int(item[0]),
            "open": float(item[1]),
            "high": float(item[2]),
            "low": float(item[3]),
            "close": float(item[4]),
            "volume": float(item[5]),
        }
        for item in all_data
    ]

def get_top_50_symbols():
    """Fetch top 50 cryptocurrencies by market cap."""
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 50,
        "page": 1,
        "sparkline": False,
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Ensure Binance-compatible symbols (e.g., BTCUSDT)
    return [coin["symbol"].upper() + "USDT" for coin in data]

def main():
    """Main function to download hourly price data."""
    top_50_symbols = get_top_50_symbols()
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=3 * 365)

    for symbol in top_50_symbols:
        print(f"Fetching data for {symbol}...")

        try:
            data = fetch_hourly_data(symbol, start_date, end_date)

            if len(data) < 3 * 365 * 24:  # Skip if less than 3 years of data
                print(f"Skipping {symbol}: insufficient data")
                continue

            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.to_csv(f"{symbol.replace('/', '_')}_hourly.csv", index=False)

            print(f"Saved data for {symbol}")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")

if __name__ == "__main__":
    main()
