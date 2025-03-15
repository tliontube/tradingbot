import aiohttp

class TelegramBot:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
       # self.binance_client = binance_client
       # self.strategy_manager = strategy_manager

    async def send_message(self, message):
        # Send a message to Telegram
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"  # Use Markdown formatting for better readability
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    print(f"Failed to send Telegram message: {await response.text()}")

    async def start(self):
        # Start the bot
        print("Telegram bot started.")

    async def stop(self):
        # Stop the bot
        print("Telegram bot stopped.")