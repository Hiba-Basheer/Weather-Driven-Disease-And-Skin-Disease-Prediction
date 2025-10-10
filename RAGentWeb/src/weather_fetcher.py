import os
import httpx
import logging
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("WeatherFetcher")

async def fetch_and_log_weather_data(city: str, api_key: str, note: str, service_type: str) -> dict:
    """
    Fetches weather data asynchronously from OpenWeatherMap API using httpx.

    Args:
        city: The city name.
        api_key: The OpenWeatherMap API key.
        note: The user-provided context note.
        service_type: Type of service calling (ML/DL) for logging.

    Returns:
        A dictionary of processed weather data or an error message.
    """
    
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"  
    }

    # Validate API key
    if not api_key:
        logger.error(f"{service_type} - Weather API Key is missing.")
        return {"error": "Weather API Key is missing. Check .env file."}

    logger.info(f"{service_type} - Fetching weather for '{city}'. Note: '{note}'")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract key weather data
            processed_data = {
                "city_name": data.get("name", city),
                "temp": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"]
            }

            logger.info(f"{service_type} - Weather data retrieved for {city}: {processed_data}")
            return processed_data

    except httpx.HTTPStatusError as e:
        error_detail = e.response.json().get('message', str(e))
        logger.error(f"{service_type} - Failed to fetch weather for {city}. HTTP Error: {error_detail}")
        return {"error": f"Failed to retrieve weather data for {city}. Detail: {error_detail}"}

    except Exception as e:
        logger.exception(f"{service_type} - Unknown error fetching weather for {city}: {e}")
        return {"error": f"Unknown error fetching weather: {e}"}



# if __name__ == "__main__":
#     async def main():
#         api_key = os.getenv("OPENWEATHERMAP_API_KEY")
#         city = "Kozhikode"
#         note = "Testing async weather fetch"
#         service_type = "RAGent-ML"
        
#         result = await fetch_and_log_weather_data(city, api_key, note, service_type)
#         print("\nFinal Result:\n", result)

#     asyncio.run(main())


