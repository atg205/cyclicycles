"""Configuration file for path management."""
from pathlib import Path
import os

# Get the project root directory (where data/ folder is)
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Define important directories
DATA_DIR = PROJECT_ROOT / 'data'
RESULT_DIR = DATA_DIR / 'results'
INSTANCE_DIR = DATA_DIR / 'instances'

def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists and return its Path object."""
    path.mkdir(parents=True, exist_ok=True)
    return path