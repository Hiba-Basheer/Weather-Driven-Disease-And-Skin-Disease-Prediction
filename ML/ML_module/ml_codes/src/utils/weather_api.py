import os
import requests
import logging
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
load_dotenv(dotenv_path)

def get_weather_data(city: str, api_key: str = None) -> dict:
    """
    Fetch current weather data for a given city using OpenWeatherMap API.

    Parameters:
        city (str): Name of the city.
        api_key (str, optional): API key for OpenWeatherMap. If None, it is 
                                 read from the environment variable 'OPENWEATHERMAP_API_KEY'.

    Returns:
        dict: A dictionary containing temperature (°C), humidity (%), and wind speed (m/s).
    """
    if api_key is None:
        api_key = os.getenv("OPENWEATHERMAP_API_KEY") 

    if not api_key:
        logger.error("OPENWEATHERMAP_API_KEY not found.")
        raise ValueError("API key not found. Ensure OPENWEATHERMAP_API_KEY is set.")

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status() 
        data = response.json()

        # Check for successful response code from OWM
        if data.get("cod") != 200:
             raise Exception(f"OpenWeatherMap API Error for {city}: {data.get('message', 'Unknown error')}")

        return {
            "temperature": float(data['main']['temp']),
            "humidity": float(data['main']['humidity']),   # 0-100%
            "wind_speed": float(data['wind']['speed'])     # m/s
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {city}: {e}")
        raise Exception(f"Failed to connect to OpenWeatherMap API: {e}")
    except KeyError as e:
        logger.error(f"Failed to parse weather data for {city}. Missing key: {e}")
        raise Exception(f"Failed to parse weather data: response missing key {e}")
