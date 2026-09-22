import os
from pathlib import Path

import yaml
from box.exceptions import BoxValueError
from box import ConfigBox

from NexText.logging import logger


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Read a YAML file and return its contents as a ConfigBox.
    """

    try:
        with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:

            content = yaml.safe_load(yaml_file)

            logger.info(
                f"YAML file: {path_to_yaml} loaded successfully."
            )

            return ConfigBox(content)

    except BoxValueError:
        raise BoxValueError(
            "YAML file is empty."
        )

    except Exception as e:
        raise e


def create_directories(
    path_to_directories: list,
    verbose=True
):
    """
    Create directories from a list of paths.
    """

    for path in path_to_directories:

        os.makedirs(path, exist_ok=True)

        if verbose:
            logger.info(
                f"Created directory at: {path}"
            )


def get_size(path: str) -> str:
    """
    Get file size in MB.
    """

    size_in_mb = round(
        os.path.getsize(path) / (1024 * 1024),
        2
    )

    return f"~ {size_in_mb} MB"