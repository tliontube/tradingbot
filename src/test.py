import asyncio
import yaml
import logging
from binance_client import BinanceClient
from gemini_client import GeminiClient
from strategy_manager import StrategyManager
from telegram_bot import TelegramBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("trade_test")

async def test_all_pairs():
    """
    Test trading on all configured pairs to verify the bot is working correctly
    with 5% balance allocation and dynamic leverage adjustment.
    """
    logger.info("Starting comprehensive trading test on all pairs")
    
    try:
        # Load config
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
            
        # Get trading pairs from config
        trading_pairs_config = config.get("binance", {}).get("trading_pairs", [])
        trading_pairs = [pair["symbol"] for pair in trading_pairs_config if pair.get("enabled", True)]
        
        # If no pairs are configured or enabled, use default pairs
        if not trading_pairs:
            trading_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"]
            logger.info(f"No trading pairs configured or enabled, using defaults: {trading_pairs}")
        else:
            logger.info(f"Trading pairs loaded from config: {trading_pairs}")

        # Initialize clients
        binance_client = BinanceClient(config["binance"]["api_key"], config["binance"]["api_secret"], config["binance"]["demo_mode"])
        gemini_client = GeminiClient(config["gemini"]["api_key"])
        telegram_bot = TelegramBot(config["telegram"]["bot_token"], config["telegram"]["chat_id"])
        
        # Initialize the strategy manager
        strategy_manager = StrategyManager(gemini_client, binance_client, telegram_bot, config)

        # Start the Telegram bot
        await telegram_bot.start()
        
        # Initialize futures settings if using futures
        if binance_client.using_futures:
            # Initialize futures settings for all trading pairs
            for pair in trading_pairs:
                await binance_client.initialize_futures_settings(pair, config)
            
            # Check testnet account balance
            account = await binance_client.client.futures_account()
            assets = account.get('assets', [])
            usdt_asset = next((asset for asset in assets if asset['asset'] == 'USDT'), None)
            
            if usdt_asset:
                available_balance = float(usdt_asset.get('availableBalance', 0))
                wallet_balance = float(usdt_asset.get('walletBalance', 0))
                
                logger.info(f"Account Balance: Available={available_balance} USDT, Wallet={wallet_balance} USDT")
                
                # If balance is low, warn the user
                if wallet_balance < 50:
                    logger.warning("Testnet balance is low, results may be affected")
            
            # Print the current leverage setting
            logger.info(f"Initial Leverage: {binance_client.futures_leverage}x")
        
        # Initialize balance tracking
        await strategy_manager.initialize_balance_tracking()
        
        # Process each trading pair one by one
        for pair in trading_pairs:
            logger.info(f"===== Testing trading on {pair} =====")
            
            # Get market data for this pair
            market_data = await binance_client.get_market_data(pair, interval="1m", limit=200)
            
            if market_data is None:
                logger.error(f"Unable to fetch market data for {pair}")
                continue
                
            # Get trading decision from Gemini
            gemini_decision = gemini_client.get_trading_decision(
                closing_prices=market_data["closing_prices"],
                volumes=market_data["volumes"],
                high_prices=market_data["high_prices"],
                low_prices=market_data["low_prices"],
                open_prices=market_data["open_prices"],
                symbol=pair
            )
            
            if gemini_decision is None:
                logger.error(f"Unable to get trading decision for {pair}")
                continue
                
            # Force a trade decision if gemini doesn't give one (for testing purposes)
            if gemini_decision["decision"] == "waiting":
                logger.info(f"No trade signal from Gemini, forcing a BUY signal for testing purposes")
                
                # Get the latest price
                latest_price = await binance_client.get_latest_price(pair)
                if latest_price is None:
                    logger.error(f"Could not get latest price for {pair}")
                    continue
                    
                # Create a test decision
                gemini_decision = {
                    "decision": "buy",
                    "reason": "TEST - Forced buy for testing purposes",
                    "entry_price": latest_price,
                    "stop_loss": latest_price * 0.98,  # 2% below entry
                    "take_profit": latest_price * 1.03  # 3% above entry
                }
            
            # Try to execute the trade
            logger.info(f"Attempting to execute {gemini_decision['decision']} trade on {pair}")
            
            trade_result = await strategy_manager.execute_trade(pair, gemini_decision)
            
            if trade_result and "orderId" in trade_result:
                logger.info(f"Successfully executed trade on {pair}!")
                logger.info(f"Order ID: {trade_result['orderId']}")
                
                # Check if the trade used dynamic leverage adjustment
                if binance_client.using_futures:
                    logger.info(f"Current leverage for {pair}: {binance_client.futures_leverage}x")
                
                # Wait for a moment to let the trade register
                await asyncio.sleep(3)
                
                # Close the position to clean up
                if binance_client.using_futures:
                    position = await binance_client.get_position_info(pair)
                    position_side = "BUY" if float(position.get("positionAmt", 0)) < 0 else "SELL"
                    close_result = await strategy_manager.close_position(pair, position_side)
                    
                    if close_result:
                        logger.info(f"Successfully closed test position for {pair}")
                    else:
                        logger.error(f"Failed to close test position for {pair}")
            else:
                logger.error(f"Failed to execute trade on {pair}")
            
            # Wait between pairs to avoid rate limits
            logger.info(f"Waiting before testing next pair...")
            await asyncio.sleep(5)
        
        logger.info("===== Testing complete! =====")
        
        # Provide a summary of the results
        balances = await binance_client.get_account_balance()
        if balances and "USDT" in balances:
            final_balance = balances["USDT"]["total"]
            logger.info(f"Final Account Balance: {final_balance} USDT")
        
        # Close connections
        await binance_client.client.close_connection()
        logger.info("Connections closed")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Make sure to close connection
        try:
            if 'binance_client' in locals():
                await binance_client.client.close_connection()
        except:
            pass

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_all_pairs())