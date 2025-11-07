"""
weather_fetcher.py
Asynchronous weather data retriever for ML and DL services using OpenWeatherMap API.
"""

import logging
import os

import httpx
from dotenv import load_dotenv

# Environment setup
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("WeatherFetcher")


# Weather fetch function
async def fetch_and_log_weather_data(
    city: str, api_key: str, note: str, service_type: str
) -> dict:
    """
    Fetches weather data asynchronously from OpenWeatherMap API using httpx.

    Args:
        city (str): Name of the city to fetch weather for.
        api_key (str): OpenWeatherMap API key.
        note (str): Context note for logging.
        service_type (str): Type of service making the request (e.g., 'ML' or 'DL').

    Returns:
        dict: Processed weather data or an error message.
    """
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}

    # Validate API key
    if not api_key:
        logger.error(f"{service_type} - Missing OpenWeatherMap API key.")
        return {"error": "Missing OpenWeatherMap API key."}

    logger.info(f"{service_type} - Fetching weather data for '{city}'. Note: '{note}'")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract and validate data fields safely
            main_data = data.get("main", {})
            wind_data = data.get("wind", {})

            processed_data = {
                "city_name": data.get("name", city),
                "temp": main_data.get("temp", 0.0),
                "humidity": main_data.get("humidity", 0),
                "wind_speed": wind_data.get("speed", 0.0),
            }

            logger.info(f"{service_type} - Weather data for {city}: {processed_data}")
            return processed_data

    except httpx.HTTPStatusError as e:
        try:
            error_detail = e.response.json().get("message", str(e))
        except Exception:
            error_detail = str(e)
        logger.error(
            f"{service_type} - HTTP error fetching weather for {city}: {error_detail}"
        )
        return {
            "error": f"Failed to fetch weather data for {city}. Detail: {error_detail}"
        }

    except httpx.RequestError as e:
        logger.error(f"{service_type} - Network error fetching weather for {city}: {e}")
        return {"error": f"Network error while fetching weather: {e}"}

    except Exception as e:
        logger.exception(
            f"{service_type} - Unexpected error while fetching weather for {city}: {e}"
        )
        return {"error": f"Unexpected error: {e}"}


# Standalone test
if __name__ == "__main__":
    import asyncio

    async def main():
        api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        city = "Kozhikode"
        note = "Testing asynchronous weather retrieval"
        service_type = "RAGent-ML"

        result = await fetch_and_log_weather_data(city, api_key, note, service_type)
        print("\nFinal Result:\n", result)

    asyncio.run(main())
