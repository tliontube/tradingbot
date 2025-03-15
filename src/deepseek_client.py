import requests
import yaml

class DeepSeekClient:
    def __init__(self, api_key, endpoint):
        self.api_key = api_key
        self.endpoint = endpoint

    def get_trading_decision(self, market_data):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(self.endpoint, json=market_data, headers=headers)
        if response.status_code == 200:
            print("DeepSeek API Response:", response.json())  # Log the response
            return response.json()
        else:
            raise Exception(f"DeepSeek API error: {response.status_code} - {response.text}")