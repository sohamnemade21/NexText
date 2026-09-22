from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parents[3]


CONFIG_FILE_PATH = PROJECT_ROOT / "config" / "config.yaml"
PARAMS_FILE_PATH = PROJECT_ROOT / "params.yaml"