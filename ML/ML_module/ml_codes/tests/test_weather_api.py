import os
import sys
from dotenv import load_dotenv
import pytest

# Locate and load the .env file
dotenv_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".env")
)
load_dotenv(dotenv_path)

# Add project root to sys.path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

from ml_codes.src.utils.weather_api import get_weather_data


def test_weather_api():
    """Test weather data returned for a given city."""
    weather = get_weather_data("London")

    assert "temperature" in weather
    assert "humidity" in weather
    assert "wind_speed" in weather
    assert isinstance(weather["temperature"], float)
    assert 0 <= weather["humidity"] <= 1
    assert weather["wind_speed"] >= 0
