"""Configuration file for path management."""
from pathlib import Path
import os

# Get the project root directory (where data/ folder is)
PROJECT_ROOT = Path(__file__).parent.parent

# Define important directories
DATA_DIR = PROJECT_ROOT / 'data' / 'instances'
RESULT_DIR = PROJECT_ROOT / 'data' / 'results'

def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists and return its Path object."""
    path.mkdir(parents=True, exist_ok=True)
    return path
