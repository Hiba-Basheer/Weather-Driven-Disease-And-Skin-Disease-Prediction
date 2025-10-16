import os
import sys
from dotenv import load_dotenv
import pytest
from pathlib import Path 

# PATH CONFIGURATION
CURRENT_FILE_DIR = Path(os.path.dirname(__file__)).resolve()

# 2. Load .env file
dotenv_path = CURRENT_FILE_DIR.parent.parent.parent / ".env"

if dotenv_path.exists():
    load_dotenv(dotenv_path)
else:
    print(f"WARNING: .env file not found at expected path: {dotenv_path}. Tests relying on API key may be skipped.")

# 3. Add path for imports
PROJECT_ROOT_FOR_IMPORTS = CURRENT_FILE_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT_FOR_IMPORTS))

# IMPORT UTILITY
try:
    from ml_codes.src.utils.weather_api import get_weather_data
except ImportError as e:
    pytest.skip(f"FATAL: Cannot import weather_api. Check that the project root path is correct. Error: {e}")

# TEST FUNCTION
def test_weather_api():
    """Test weather data returned for a given city."""
    if not os.getenv("OPENWEATHERMAP_API_KEY"):
        pytest.skip("OPENWEATHERMAP_API_KEY environment variable is missing.")

    # Execute the API call
    weather = get_weather_data("London")

    # Assertions
    assert "temperature" in weather
    assert "humidity" in weather
    assert "wind_speed" in weather
    assert isinstance(weather["temperature"], float)
    
    assert 0 <= weather["humidity"] <= 100 
    
    assert weather["wind_speed"] >= 0