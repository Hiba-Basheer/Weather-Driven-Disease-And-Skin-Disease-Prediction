import os
import requests
from dotenv import load_dotenv

# Load environment variables from the .env file
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
load_dotenv(dotenv_path)

def get_weather_data(city, api_key=None):
    """
    Fetch current weather data for a given city using WeatherAPI.

    Parameters:
        city (str): Name of the city.
        api_key (str, optional): API key for WeatherAPI. If not provided, it is read from the environment.

    Returns:
        dict: A dictionary containing temperature (°C), humidity (fraction), and wind speed (m/s).
    """
    if api_key is None:
        api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        raise ValueError("API key not found.")

    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}&aqi=no"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch weather data: {response.json()}")

    data = response.json()
    return {
        "temperature": data["current"]["temp_c"],
        "humidity": data["current"]["humidity"] / 100,
        "wind_speed": data["current"]["wind_kph"] / 3.6  # Convert km/h to m/s
    }
