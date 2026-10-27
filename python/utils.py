import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(*parts: str) -> str:
    """Return an absolute path rooted at the repository directory."""
    return str(PROJECT_ROOT.joinpath(*parts))


def load_config(filepath: str):
    """Loads configuration from a JSON file.
    
    Args:
        filepath (str): Path to config file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{filepath}' not found. Please create it.")
    except json.JSONDecodeError:
        raise ValueError(f"Could not decode JSON from '{filepath}'. Check file format.")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while loading config from '{filepath}': {e}")
