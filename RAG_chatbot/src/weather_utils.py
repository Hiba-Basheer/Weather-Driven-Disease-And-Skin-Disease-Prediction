"""
Weather enrichment module for RAG chatbot.
Extracts city from user input and fetches current weather using OpenWeatherMap.
"""

import os
import re
import requests
import logging
from dotenv import load_dotenv

# Load environment variables
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))  
API_KEY = os.getenv("OPENWEATHER_API_KEY")

print("Loaded API key:", API_KEY)  

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_city(text: str) -> str:
    """
    Extracts city name from user input using regex.

    Parameters:
        text (str): Raw user input or structured string.

    Returns:
        str: City name if found, else None.
    """
    match = re.search(r"\b(?:in|from|at|living in|live in)\s+([A-Za-z\s]+)", text.lower())
    if match:
        city = match.group(1).strip()
        logging.info(f"City extracted: {city}")
        return city
    logging.warning("City not found in input.")
    return None

def fetch_weather(city: str) -> dict:
    """
    Fetches current weather data for a given city using OpenWeatherMap API.

    Parameters:
        city (str): City name.

    Returns:
        dict: Weather details (temperature, humidity, condition).
    """
    if not API_KEY:
        logging.error("OPENWEATHER_API_KEY not found in .env file.")
        return {}

    try:
        params = {"q": city, "appid": API_KEY, "units": "metric"}
        response = requests.get("https://api.openweathermap.org/data/2.5/weather", params=params)
        data = response.json()

        logging.info(f"Weather API response for {city}: {data}")

        if response.status_code != 200:
            logging.error(f"Weather API failed with status {response.status_code}: {data}")
            return {}

        if "main" not in data or "weather" not in data:
            logging.warning(f"Incomplete weather data for city '{city}': {data}")
            return {}

        weather = {
            "temperature": f"{data['main']['temp']}°C",
            "humidity": f"{data['main']['humidity']}%",
            "condition": data['weather'][0]['description'],
            "wind_speed": f"{data['wind']['speed']} m/s"
        }
        logging.info(f"Weather fetched for {city}: {weather}")
        return weather
    except Exception as e:
        logging.error(f"Failed to fetch weather for {city}: {e}")
        return {}

# test
if __name__ == "__main__":
    test_city = "Kochi"
    weather = fetch_weather(test_city)
    print("Weather:", weather)