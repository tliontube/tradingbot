from binance import AsyncClient

async def test_binance_api():
    api_key = "99a8aacfd74ae18e9da094a001ac280f4c8f69812d8a3ae000ba88ae224d4453"
    api_secret = "5c97d8a81b85dd0272f191c4953f9298da17bb08c6b00946914d5aa0aceb7fbe"

    # Connect to Binance Testnet
    client = await AsyncClient.create(api_key, api_secret, testnet=True)

    try:
        # Fetch account balance
        account_info = await client.get_account()
        print("Full Account Info:", account_info)

        # Find TRX balance
        trx_balance = next((balance for balance in account_info["balances"] if balance["asset"] == "TRX"), None)
        if trx_balance:
            print("TRX Balance:", trx_balance)
        else:
            print("TRX balance not found.")

        # Fetch market data
        klines = await client.get_klines(symbol="BTCUSDT", interval="1h", limit=1)
        print("Market Data:", klines)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.close_connection()

# Run the test
import asyncio
asyncio.run(test_binance_api())