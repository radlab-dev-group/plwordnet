from typing import Optional

from plwordnet_handler.base.connectors.milvus.config import MilvusConfig
from plwordnet_handler.base.connectors.milvus.core.base import _MilvusBaseConnector


class MilvusConnector(_MilvusBaseConnector):
    def __init__(self, logger_name: Optional[str] = None, **kwargs):
        """
        Initialize the Milvus connector with optional logger configuration.

        Args:
            logger_name: Optional name for the logger instance. If None,
            uses the module name as the default logger name
            **kwargs: Additional keyword arguments passed
            to the parent class constructor
        """

        if logger_name is None:
            logger_name = __name__

        super().__init__(logger_name=logger_name, **kwargs)

    @classmethod
    def from_config_file(cls, config_path: str) -> "MilvusConnector":
        """
        Create a MilvusWordNetBaseHandler instance from a configuration file.

        Args:
            config_path: Path to the JSON configuration file

        Returns:
            MilvusWordNetSchemaHandler: Handler instance with loaded configuration
        """
        config = MilvusConfig.from_json_file(config_path)
        return cls(config=config)
