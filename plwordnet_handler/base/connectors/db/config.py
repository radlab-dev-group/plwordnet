import json

from pathlib import Path
from dataclasses import dataclass


@dataclass
class DbSQLConfig:
    """
    Configuration class for MySQL database connection.
    """

    host: str
    port: int
    user: str
    password: str
    database: str

    @classmethod
    def from_json_file(cls, config_path: str) -> "DbSQLConfig":
        """
        Load Db SQL configuration from a JSON file.

        Args:
            config_path: Path to the JSON configuration file

        Returns:
            DbSQLConfig: Configuration object

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            json.JSONDecodeError: If the JSON is invalid
            KeyError: If required configuration keys are missing
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        required_keys = {"host", "port", "user", "password", "database"}
        missing_keys = required_keys - set(config_data.keys())
        if missing_keys:
            raise KeyError(f"Missing required configuration keys: {missing_keys}")

        return cls(
            host=config_data["host"],
            port=config_data["port"],
            user=config_data["user"],
            password=config_data["password"],
            database=config_data["database"],
        )

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary format.

        Returns:
            dict: Configuration as dictionary
        """
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "database": self.database,
        }
