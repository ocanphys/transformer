from pathlib import Path

# Since config.py is at the root, its parent IS the root directory
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_VOLUME = PROJECT_ROOT / "volume"
