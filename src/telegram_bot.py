import aiohttp
import logging

class TelegramBot:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.session = aiohttp.ClientSession()
        self.logger = logging.getLogger(__name__)

    async def start(self):
        """Start the Telegram bot session."""
        self.logger.info("Telegram bot started.")

    async def send_message(self, message):
        """Send a message to the configured chat."""
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": self._escape_markdown(message),
            "parse_mode": "MarkdownV2"
        }
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    self.logger.error(f"Failed to send message: {await response.text()}")
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")

    def _escape_markdown(self, text):
        """Escape special characters for Telegram Markdown."""
        escape_chars = r"_*[]()~`>#+-=|{}.!"
        return ''.join(f"\\{char}" if char in escape_chars else char for char in text)

    async def send_photo(self, photo_file):
        """Send a photo to the configured chat."""
        url = f"{self.base_url}/sendPhoto"
        form = aiohttp.FormData()
        form.add_field('chat_id', str(self.chat_id))
        form.add_field('photo', photo_file)
        
        try:
            async with self.session.post(url, data=form) as response:
                if response.status != 200:
                    self.logger.error(f"Failed to send photo: {await response.text()}")
        except Exception as e:
            self.logger.error(f"Error sending photo: {e}")

    async def stop(self):
        """Close the Telegram bot session."""
        await self.session.close()
        self.logger.info("Telegram bot stopped.")
