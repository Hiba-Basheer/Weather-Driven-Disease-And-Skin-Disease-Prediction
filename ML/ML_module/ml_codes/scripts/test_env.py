import os
from dotenv import load_dotenv

dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))
load_dotenv(dotenv_path)

api_key = os.getenv("OPENWEATHER_API_KEY")
print(api_key)