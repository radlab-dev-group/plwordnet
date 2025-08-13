import json

from pathlib import Path
from dataclasses import dataclass


@dataclass
class MilvusConfig:
    """
    Configuration class for Milvus connection parameters.
    """

    host: str = "localhost"
    port: str = "19530"
    user: str = ""
    password: str = ""
    db_name: str = "default"

    @classmethod
    def from_json_file(cls, config_path: str) -> "MilvusConfig":
        """
        Load Milvus configuration from a JSON file.

        Args:
            config_path: Path to the JSON configuration file

        Returns:
            MilvusConfig: Configuration object

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            json.JSONDecodeError: If the JSON is invalid
            KeyError: If required configuration keys are missing
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"File not found: {config_path}")

        with open(config_file, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        if "host" not in config_data:
            raise KeyError("Missing required configuration key: host")

        return cls(
            host=config_data.get("host", "localhost"),
            port=config_data.get("port", "19530"),
            user=config_data.get("user", ""),
            password=config_data.get("password", ""),
            db_name=config_data.get("db_name", "default"),
        )
