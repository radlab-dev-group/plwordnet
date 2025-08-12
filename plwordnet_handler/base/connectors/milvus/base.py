import logging

from abc import ABC
from typing import Optional
from pymilvus import connections, MilvusException

from plwordnet_handler.base.connectors.milvus.config import MilvusConfig
from plwordnet_handler.utils.logger import prepare_logger


class MilvusWordNetBaseHandler(ABC):
    """
    Handler for managing Milvus collections for WordNet embeddings.
    Supports both synset and lexical unit embeddings with metadata.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        user: str = "",
        password: str = "",
        db_name: str = "default",
        config: Optional[MilvusConfig] = None,
        log_level: str = "INFO",
        logger_name: Optional[str] = None,
    ):
        """
        Initialize Milvus connection handler.

        Args:
            host: Milvus server host
            port: Milvus server port
            user: Username for authentication
            password: Password for authentication
            db_name: Database name
            config: MilvusConfig object (takes precedence over individual parameters)
            log_level: Logging level (Default is INFO)
        """
        if config:
            self.host = config.host
            self.port = config.port
            self.user = config.user
            self.password = config.password
            self.db_name = config.db_name
        else:
            self.host = host
            self.port = port
            self.user = user
            self.password = password
            self.db_name = db_name

        self.connection_alias = "wordnet_connection"

        self.log_level = log_level
        self.logger = prepare_logger(
            logger_name=logger_name or self.__name__, log_level=log_level
        )

    @classmethod
    def from_config_file(cls, config_path: str) -> "MilvusWordNetBaseHandler":
        """
        Create a MilvusWordNetBaseHandler instance from a configuration file.

        Args:
            config_path: Path to the JSON configuration file

        Returns:
            MilvusWordNetSchemaHandler: Handler instance with loaded configuration
        """
        config = MilvusConfig.from_json_file(config_path)
        return cls(config=config)

    def connect(self) -> bool:
        """
        Establish connection to Milvus server.

        Returns:
            bool: True if the connection is successful, False otherwise
        """
        try:
            connections.connect(
                alias=self.connection_alias,
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db_name=self.db_name,
            )
            self.logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            return True
        except MilvusException as e:
            self.logger.error(f"Failed to connect to Milvus: {e}")
            return False

    def disconnect(self):
        """Disconnect from Milvus server."""
        try:
            connections.disconnect(self.connection_alias)
            self.logger.info("Disconnected from Milvus")
        except MilvusException as e:
            self.logger.error(f"Error disconnecting from Milvus: {e}")
