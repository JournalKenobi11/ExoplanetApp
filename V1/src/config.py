from pathlib import Path
from dotenv import load_dotenv
import os


PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(
    PROJECT_ROOT / ".env"
)


MODEL_PATH = (
    PROJECT_ROOT
    / os.getenv(
        "MODEL_PATH"
    )
)

CACHE_DB = (
    PROJECT_ROOT
    / os.getenv(
        "CACHE_DB"
    )
)

THRESHOLD = float(
    os.getenv(
        "THRESHOLD",
        0.50
    )
)