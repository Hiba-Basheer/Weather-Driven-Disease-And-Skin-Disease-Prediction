import os
from dotenv import load_dotenv

load_dotenv()  # ensures it loads the new .env key
print(os.getenv("OPENAI_API_KEY"))  # check it prints your new key
