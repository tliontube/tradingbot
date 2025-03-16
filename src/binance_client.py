from binance import AsyncClient
from decimal import Decimal
import logging
import time
import asyncio
import math

class BinanceClient:
    def __init__(self, api_key, api_secret, demo_mode=True):
        self.client = AsyncClient(api_key, api_secret, testnet=demo_mode)
        self.demo_mode = demo_mode
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        # Flag to indicate we're using futures
        self.using_futures = True
        # Futures settings
        self.futures_leverage = 10
        self.futures_margin_type = "ISOLATED"
        self.futures_position_side = "BOTH"

    async def initialize_futures_settings(self, symbol, config=None):
        """
        Initialize futures trading settings such as leverage and margin type.
        
        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT')
            config (dict, optional): Configuration with futures settings
        """
        if not self.using_futures:
            return
            
        try:
            # Load configuration if provided
            if config and 'futures' in config.get('binance', {}):
                futures_config = config['binance']['futures']
                self.futures_leverage = futures_config.get('leverage', 10)
                self.futures_margin_type = futures_config.get('margin_type', 'ISOLATED')
                self.futures_position_side = futures_config.get('position_side', 'BOTH')
                
            # Configure position mode (HEDGE or ONE_WAY)
            position_mode = "HEDGE" if self.futures_position_side == "BOTH" else "ONE_WAY"
            try:
                await self.client.futures_change_position_mode(dualSidePosition=position_mode == "HEDGE")
                self.logger.info(f"Set position mode to {position_mode}")
            except Exception as e:
                # If error contains "No need to change position side", it's already set correctly
                if "No need to change position side" not in str(e):
                    self.logger.warning(f"Error setting position mode: {e}")
                
            # Set margin type
            try:
                await self.client.futures_change_margin_type(symbol=symbol, marginType=self.futures_margin_type)
                self.logger.info(f"Set margin type to {self.futures_margin_type}")
            except Exception as e:
                # If already set, API returns an error
                if "No need to change margin type" not in str(e):
                    self.logger.warning(f"Error setting margin type: {e}")
                
            # Set leverage
            try:
                response = await self.client.futures_change_leverage(symbol=symbol, leverage=self.futures_leverage)
                self.logger.info(f"Set leverage to {response['leverage']}x")
            except Exception as e:
                self.logger.warning(f"Error setting leverage: {e}")
            
            # Check if we need to fund the account with test assets
            await self.fund_testnet_account()
                
            self.logger.info(f"Futures settings initialized: leverage={self.futures_leverage}x, margin={self.futures_margin_type}, position_side={self.futures_position_side}")
            
        except Exception as e:
            self.logger.error(f"Error initializing futures settings: {e}")
            
    async def fund_testnet_account(self):
        """
        Fund the testnet account with test USDT if needed.
        This is only available on testnet and will fail on mainnet.
        """
        if not self.demo_mode:
            self.logger.info("Funding is only available on testnet, skipping...")
            return
            
        try:
            # Check current balance
            account = await self.client.futures_account()
            assets = account.get('assets', [])
            usdt_asset = next((asset for asset in assets if asset['asset'] == 'USDT'), None)
            
            if usdt_asset:
                available_balance = float(usdt_asset.get('availableBalance', 0))
                wallet_balance = float(usdt_asset.get('walletBalance', 0))
                self.logger.info(f"Current USDT balance: Available={available_balance}, Wallet={wallet_balance}")
                
                # If balance is low, try to fund the account
                if wallet_balance < 100:
                    self.logger.info("USDT balance is low, attempting to fund testnet account...")
                    
                    # Try all available methods to fund the account
                    # Different testnet versions have different methods
                    funding_methods = [
                        self._try_account_transfer,
                        self._try_direct_asset_transfer, 
                        self._try_get_assets_api,
                        self._try_test_order
                    ]
                    
                    for method in funding_methods:
                        try:
                            result = await method()
                            if result:
                                self.logger.info(f"Successfully funded testnet account using {method.__name__}")
                                # Check if balance increased
                                await asyncio.sleep(2)  # Wait for server to update
                                new_account = await self.client.futures_account()
                                for asset in new_account.get('assets', []):
                                    if asset['asset'] == 'USDT':
                                        new_balance = float(asset.get('walletBalance', 0))
                                        if new_balance > wallet_balance:
                                            self.logger.info(f"Balance increased from {wallet_balance} to {new_balance}")
                                            return True
                                        break
                        except Exception as e:
                            self.logger.warning(f"Failed funding method {method.__name__}: {e}")
                            continue
                    
                    self.logger.info("All funding methods tried. Please manually fund your testnet account with test USDT at https://testnet.binancefuture.com")
            else:
                self.logger.warning("USDT asset not found in account")
                
        except Exception as e:
            self.logger.error(f"Error funding testnet account: {e}")
            self.logger.info("Please manually fund your testnet account with test USDT at https://testnet.binancefuture.com")
        
        return False

    async def _try_account_transfer(self):
        """Try to transfer funds from spot to futures wallet"""
        try:
            # Standard transfer between spot and futures
            result = await self.client.futures_account_transfer(
                asset='USDT',
                amount='10000',
                type=1  # 1: Spot to USDⓈ-Futures, 2: USDⓈ-Futures to Spot
            )
            self.logger.info(f"Transfer result: {result}")
            return True
        except Exception as e:
            self.logger.warning(f"Could not transfer funds: {e}")
            return False

    async def _try_direct_asset_transfer(self):
        """Try direct asset transfer API"""
        try:
            # Direct asset transfer API (varies by testnet)
            result = await self.client._request_futures_api(
                'post', 
                'asset/transfer', 
                signed=True, 
                data={
                    'asset': 'USDT',
                    'amount': '10000',
                    'type': 1
                }
            )
            self.logger.info(f"Direct asset transfer result: {result}")
            return True
        except Exception as e:
            self.logger.warning(f"Could not perform direct asset transfer: {e}")
            return False

    async def _try_get_assets_api(self):
        """Try get assets API (some testnets provide this)"""
        try:
            # Try "get assets" API if available
            result = await self.client._request_futures_api(
                'post',
                'asset/get-assets',
                signed=True,
                data={}
            )
            self.logger.info(f"Get assets result: {result}")
            return True
        except Exception as e:
            self.logger.warning(f"Could not get test assets: {e}")
            return False
            
    async def _try_test_order(self):
        """Try placing a test order to trigger system funding"""
        try:
            # Some testnets fund account when placing test orders
            result = await self.client._request_futures_api(
                'post',
                'order/test',
                signed=True,
                data={
                    'symbol': 'BTCUSDT',
                    'side': 'BUY',
                    'type': 'MARKET',
                    'quantity': 0.001
                }
            )
            self.logger.info(f"Test order result: {result}")
            return True
        except Exception as e:
            self.logger.warning(f"Could not place test order: {e}")
            return False

    async def get_market_data(self, symbol, interval, limit):
        try:
            # Fetch candlestick data using futures endpoint
            if self.using_futures:
                klines = await self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)
            else:
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
            self.logger.error(f"Error fetching market data for {symbol}: {e}")
            return None

    async def get_latest_price(self, symbol):
        try:
            # Fetch the latest price using futures endpoint
            if self.using_futures:
                ticker = await self.client.futures_symbol_ticker(symbol=symbol)
                return float(ticker["price"])  # Return the latest price as a float
            else:
                ticker = await self.client.get_symbol_ticker(symbol=symbol)
                return float(ticker["price"])
        except Exception as e:
            self.logger.error(f"Error fetching latest price for {symbol}: {e}")
            return None

    async def get_symbol_info(self, symbol):
        """
        Get detailed information for a symbol including precision requirements.
        
        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            dict: Symbol information including precision requirements
        """
        try:
            if self.using_futures:
                # For futures, get exchange info and find the specific symbol
                exchange_info = await self.client.futures_exchange_info()
                for symbol_info in exchange_info["symbols"]:
                    if symbol_info["symbol"] == symbol:
                        # Adapt futures info to match the format used in spot
                        return {
                            "symbol": symbol_info["symbol"],
                            "status": symbol_info["status"],
                            "baseAsset": symbol_info["baseAsset"],
                            "quoteAsset": symbol_info["quoteAsset"],
                            "quantityPrecision": symbol_info["quantityPrecision"],
                            "pricePrecision": symbol_info["pricePrecision"]
                        }
                return None
            else:
                # For spot trading
                exchange_info = await self.client.get_exchange_info()
                for symbol_info in exchange_info["symbols"]:
                    if symbol_info["symbol"] == symbol:
                        return symbol_info
                return None
        except Exception as e:
            self.logger.error(f"Error fetching symbol info for {symbol}: {e}")
            return None

    async def get_account_balance(self):
        """
        Get account balance information.
        
        Returns:
            dict: Account balance for all assets
        """
        try:
            if self.using_futures:
                # Get futures account balance
                account = await self.client.futures_account()
                balances = {}
                for asset in account["assets"]:
                    balances[asset["asset"]] = {
                        "free": float(asset["availableBalance"]),
                        "locked": float(asset["initialMargin"]),
                        "total": float(asset["walletBalance"])
                    }
                return balances
            else:
                # Get spot account balance
                account = await self.client.get_account()
                balances = {}
                for balance in account["balances"]:
                    asset = balance["asset"]
                    free = float(balance["free"])
                    locked = float(balance["locked"])
                    if free > 0 or locked > 0:
                        balances[asset] = {
                            "free": free,
                            "locked": locked,
                            "total": free + locked
                        }
                return balances
        except Exception as e:
            self.logger.error(f"Error fetching account balance: {e}")
            return None

    async def place_order(self, symbol, side, order_type, quantity, price=None, time_in_force="GTC", reduce_only=False, positionSide=None, stop_price=None):
        """
        Place an order on Binance.
        
        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT')
            side (str): Order side (BUY or SELL)
            order_type (str): Order type (LIMIT, MARKET, STOP_MARKET, TAKE_PROFIT_MARKET, etc.)
            quantity (float): Order quantity
            price (float, optional): Order price for limit orders
            time_in_force (str, optional): Time in force (GTC, IOC, FOK)
            reduce_only (bool, optional): Whether this order is to reduce position only (futures only)
            positionSide (str, optional): Position side for futures in hedged mode (LONG or SHORT)
            stop_price (float, optional): Stop price for stop orders
            
        Returns:
            dict: Order information
        """
        try:
            # Validate quantity is greater than zero
            if quantity <= 0:
                error_msg = f"Cannot place order with quantity <= 0: {quantity}"
                self.logger.error(error_msg)
                return {"status": "ERROR", "msg": error_msg}
                
            # For market orders, get the current price to calculate notional value
            current_price = price
            if order_type == "MARKET" or current_price is None:
                current_price = await self.get_latest_price(symbol)
                if not current_price:
                    self.logger.error("Could not get latest price to calculate notional value")
                    return {"status": "ERROR", "msg": "Could not get latest price"}
            
            # Calculate notional value (quantity * price)
            notional_value = quantity * current_price
            
            # Check minimum notional for futures
            if self.using_futures:
                min_notional = 100.1  # Binance Futures requires min 100.0 USDT
                if notional_value < min_notional:
                    self.logger.warning(f"Order notional value ({notional_value:.2f} USDT) is below minimum ({min_notional} USDT). Adjusting quantity.")
                    # Adjust quantity to meet minimum notional
                    new_quantity = min_notional / current_price
                    # Round to appropriate precision
                    symbol_info = await self.get_symbol_info(symbol)
                    if symbol_info and 'quantityPrecision' in symbol_info:
                        precision = symbol_info['quantityPrecision']
                        # Round down to precision
                        factor = 10 ** precision
                        new_quantity = math.floor(new_quantity * factor) / factor
                    else:
                        # Default precision of 3
                        new_quantity = math.floor(new_quantity * 1000) / 1000
                    
                    quantity = new_quantity
                    notional_value = quantity * current_price
                    self.logger.info(f"Adjusted quantity to {quantity} to meet minimum notional requirement. New notional: {notional_value:.2f} USDT")
            
            # Handle case where order_type was passed with lowercase
            order_type = order_type.upper()
            
            # Map common format errors
            if order_type == "STOP_MARKET":
                order_type = "STOP_MARKET"  # normalize format
            elif order_type == "TAKE_PROFIT_MARKET":
                order_type = "TAKE_PROFIT_MARKET"  # normalize format
            
            self.logger.info(f"Placing {side} {order_type} order for {symbol}: quantity={quantity}, price={current_price}, notional={notional_value:.2f} USDT")
            
            if self.using_futures:
                params = {
                    "symbol": symbol,
                    "side": side.upper(),  # Ensure side is uppercase
                    "type": order_type,
                    "quantity": quantity
                }
                
                # Only include reduceOnly parameter if it's explicitly set to True
                if reduce_only:
                    params["reduceOnly"] = True
                
                # If in hedged mode (position_side = "BOTH"), specify the position side
                if self.futures_position_side == "BOTH":
                    # Use provided positionSide if available
                    if positionSide:
                        if positionSide in ["LONG", "SHORT"]:
                            params["positionSide"] = positionSide
                            self.logger.info(f"Using provided position side: {positionSide}")
                        else:
                            self.logger.warning(f"Invalid positionSide: {positionSide}. Must be LONG or SHORT.")
                    elif not reduce_only:
                        # For new positions (not reduce only), set the position side based on the order side
                        auto_position_side = "LONG" if side.upper() == "BUY" else "SHORT"
                        params["positionSide"] = auto_position_side
                        self.logger.info(f"Setting position side to {auto_position_side} for {side} order")
                
                if order_type == "LIMIT":
                    params["price"] = price
                    params["timeInForce"] = time_in_force
                
                # For stop and take profit orders, add stopPrice
                if order_type in ["STOP", "STOP_MARKET", "TAKE_PROFIT", "TAKE_PROFIT_MARKET", "TRAILING_STOP_MARKET"]:
                    if stop_price is None:
                        self.logger.error(f"Stop price is required for {order_type} orders")
                        return {"status": "ERROR", "msg": "Stop price is required"}
                    params["stopPrice"] = stop_price
                
                # Place futures order
                try:
                    order = await self.client.futures_create_order(**params)
                    self.logger.info(f"Futures order placed: {order}")
                    return order
                except Exception as e:
                    error_msg = str(e)
                    if "APIError(code=-4164)" in error_msg:
                        # This specific error means the notional value is too small
                        actual_notional = quantity * current_price
                        self.logger.error(f"Order placement failed: Notional too small. Required: 100.0, Actual: {actual_notional:.8f}")
                        
                        # Try to place with absolute minimum
                        min_needed_qty = 100.1 / current_price
                        symbol_info = await self.get_symbol_info(symbol)
                        if symbol_info and 'quantityPrecision' in symbol_info:
                            precision = symbol_info['quantityPrecision']
                            factor = 10 ** precision
                            min_needed_qty = math.ceil(min_needed_qty * factor) / factor
                        
                        self.logger.info(f"Attempting with minimum quantity: {min_needed_qty}")
                        params["quantity"] = min_needed_qty
                        
                        try:
                            order = await self.client.futures_create_order(**params)
                            self.logger.info(f"Order placed with minimum quantity: {order}")
                            return order
                        except Exception as retry_error:
                            self.logger.error(f"Failed on retry with minimum quantity: {retry_error}")
                            
                            # If the error is related to position side, try to fix it
                            if "APIError(code=-4061)" in str(retry_error):
                                self.logger.warning("Position side error detected, trying to fix...")
                                
                                # Adjust position side if needed
                                if self.futures_position_side == "BOTH":
                                    if "positionSide" in params:
                                        # Try the opposite position side as last resort
                                        opposite_position_side = "SHORT" if params["positionSide"] == "LONG" else "LONG"
                                        params["positionSide"] = opposite_position_side
                                        self.logger.info(f"Retrying with opposite position side: {opposite_position_side}")
                                        
                                        try:
                                            order = await self.client.futures_create_order(**params)
                                            self.logger.info(f"Order placed with adjusted position side: {order}")
                                            return order
                                        except Exception as final_error:
                                            self.logger.error(f"Final attempt failed: {final_error}")
                            
                            return None
                    else:
                        raise e
            else:
                params = {
                    "symbol": symbol,
                    "side": side,
                    "type": order_type,
                    "quantity": quantity
                }
                
                if order_type == "LIMIT":
                    params["price"] = price
                    params["timeInForce"] = time_in_force
                
                # For stop orders in spot
                if "STOP" in order_type:
                    if stop_price is None:
                        self.logger.error(f"Stop price is required for {order_type} orders")
                        return {"status": "ERROR", "msg": "Stop price is required"}
                    params["stopPrice"] = stop_price
                
                # Place spot order
                order = await self.client.create_order(**params)
                self.logger.info(f"Spot order placed: {order}")
                return order
        except Exception as e:
            # Check for specific error codes
            error_msg = str(e)
            if "APIError(code=-4003)" in error_msg:
                self.logger.error(f"Error placing order: Quantity less than or equal to zero. Provided quantity: {quantity}")
                # Check the account balance to provide more context
                try:
                    account = await self.client.futures_account()
                    usdt_balance = None
                    for asset in account.get('assets', []):
                        if asset['asset'] == 'USDT':
                            usdt_balance = float(asset.get('availableBalance', 0))
                            break
                            
                    self.logger.info(f"Current USDT balance: {usdt_balance}")
                    
                    if usdt_balance is not None and usdt_balance < 10:
                        self.logger.warning(f"Low USDT balance detected: {usdt_balance}. Need to fund testnet account.")
                        await self.fund_testnet_account()
                except Exception as balance_e:
                    self.logger.error(f"Error checking balance: {balance_e}")
            elif "APIError(code=-2019)" in error_msg:
                self.logger.error(f"Error placing order: Margin is insufficient. Need to fund account or reduce position size.")
            elif "APIError(code=-4164)" in error_msg:
                actual_notional = quantity * current_price
                self.logger.error(f"Error placing order: Order's notional must be no smaller than 100.0. Current notional: quantity * price = {actual_notional}")
                if current_price and current_price > 0:
                    min_quantity = 100.1 / current_price
                    self.logger.info(f"To meet minimum notional of 100.0 USDT, quantity should be at least {min_quantity:.8f} {symbol.replace('USDT', '')}")
            else:
                self.logger.error(f"Error placing order: {e}")
                
            return None

    async def cancel_order(self, symbol, order_id):
        """
        Cancel an existing order.
        
        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT')
            order_id (str): Order ID to cancel
            
        Returns:
            dict: Cancellation result
        """
        try:
            if self.using_futures:
                result = await self.client.futures_cancel_order(symbol=symbol, orderId=order_id)
            else:
                result = await self.client.cancel_order(symbol=symbol, orderId=order_id)
            return result
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return None

    async def get_open_orders(self, symbol=None):
        """
        Get all open orders for a symbol.
        
        Args:
            symbol (str, optional): The trading pair symbol (e.g., 'BTCUSDT'). If None, get all open orders.
            
        Returns:
            list: List of open orders
        """
        try:
            if self.using_futures:
                if symbol:
                    result = await self.client.futures_get_open_orders(symbol=symbol)
                else:
                    result = await self.client.futures_get_open_orders()
            else:
                if symbol:
                    result = await self.client.get_open_orders(symbol=symbol)
                else:
                    result = await self.client.get_open_orders()
            return result
        except Exception as e:
            self.logger.error(f"Error getting open orders: {e}")
            return []

    async def get_position_info(self, symbol):
        """
        Get position information for a futures symbol.
        
        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            dict: Position information
        """
        try:
            if not self.using_futures:
                self.logger.error("Position info is only available for futures")
                return None
                
            positions = await self.client.futures_position_information(symbol=symbol)
            for position in positions:
                if position["symbol"] == symbol:
                    return position
            return None
        except Exception as e:
            self.logger.error(f"Error getting position info for {symbol}: {e}")
            return None

    async def close_connection(self):
        """Close the client session."""
        try:
            await self.client.close_connection()
        except Exception as e:
            self.logger.error(f"Error closing connection: {e}")

    async def cancel_all_orders(self, symbol):
        """
        Cancel all open orders for a symbol.
        
        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            dict: Cancellation result
        """
        try:
            if self.using_futures:
                result = await self.client.futures_cancel_all_open_orders(symbol=symbol)
            else:
                # For spot, we need to cancel orders one by one
                open_orders = await self.get_open_orders(symbol)
                results = []
                for order in open_orders:
                    cancel_result = await self.cancel_order(symbol, order["orderId"])
                    results.append(cancel_result)
                return results
            return result
        except Exception as e:
            self.logger.error(f"Error cancelling all orders: {e}")
            return None

    async def get_price_precision(self, symbol):
        """
        Get the price precision for a symbol.
        
        Args:
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            int: Price precision
        """
        try:
            symbol_info = await self.get_symbol_info(symbol)
            if symbol_info and 'pricePrecision' in symbol_info:
                return symbol_info['pricePrecision']
            else:
                # Default to 2 decimal places if not found
                return 2
        except Exception as e:
            self.logger.error(f"Error getting price precision: {e}")
            return 2  # Default to 2 decimal places