import argparse
from pathlib import Path

def existing_file(path_str: str) -> Path:
    """Validate that path exists and is a file, then return Path object."""
    path = Path(path_str)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Not a file: {path}")
    return path