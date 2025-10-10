import os
from dotenv import load_dotenv

def load_environment():
    env_file = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    load_dotenv(os.path.abspath(env_file))
