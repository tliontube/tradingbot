from binance import AsyncClient

class BinanceClient:
    def __init__(self, api_key, api_secret, demo_mode=True):
        self.client = AsyncClient(api_key, api_secret, testnet=demo_mode)

    async def get_market_data(self, symbol, interval, limit):
        try:
            # Fetch candlestick data for the specified timeframe
            klines = await self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            if limit == 1:
                if isinstance(klines, list) and len(klines) > 0:
                    return klines[-1]  # Return the latest candlestick
                else:
                    raise ValueError(f"Invalid klines data: {klines}")
            else:
                # Process klines to extract both price and volume data
                processed_data = {
                    "closing_prices": [float(k[4]) for k in klines],  # Close price
                    "volumes": [float(k[5]) for k in klines],         # Volume
                    "high_prices": [float(k[2]) for k in klines],     # High price
                    "low_prices": [float(k[3]) for k in klines],      # Low price
                    "open_prices": [float(k[1]) for k in klines],     # Open price
                    "timestamps": [int(k[0]) for k in klines]         # Timestamp
                }
                return processed_data
        except Exception as e:
            print(f"Error fetching market data for {symbol}: {e}")
            return None

    async def get_latest_price(self, symbol):
        try:
            # Fetch the latest price for the symbol
            ticker = await self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker["price"])  # Return the latest price as a float
        except Exception as e:
            print(f"Error fetching latest price for {symbol}: {e}")
            return None

    async def close_connection(self):
        await self.client.close_connection()