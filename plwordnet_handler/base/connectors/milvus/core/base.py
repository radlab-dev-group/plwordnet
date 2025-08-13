from abc import ABC, abstractmethod
from pymilvus import (
    db,
    connections,
    Collection,
    utility,
    MilvusException,
)
from typing import Dict, Any, Optional

from plwordnet_handler.utils.logger import prepare_logger
from plwordnet_handler.base.connectors.milvus.config import MilvusConfig


class _MilvusBaseConnector(ABC):
    conn_name = "wordnet_connection"
    name_default = "wordnet_connection_default"

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

        self.log_level = log_level
        self.logger = prepare_logger(
            logger_name=logger_name or self.__name__, log_level=log_level
        )

    def connect(self) -> bool:
        """
        Establish connection to Milvus server.

        Returns:
            bool: True if the connection is successful, False otherwise
        """
        try:
            connections.connect(
                alias=self.conn_name,
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
        """
        Disconnect from Milvus server.
        """
        try:
            connections.disconnect(self.conn_name)
            self.logger.info("Disconnected from Milvus")
        except MilvusException as e:
            self.logger.error(f"Error disconnecting from Milvus: {e}")


class _MilvusBaseInitializer(_MilvusBaseConnector, ABC):
    def connect_to_default(self) -> bool:
        """
        Connect to a default database to create a custom database.

        Returns:
            bool: True if the connection is successful, False otherwise
        """
        try:
            connections.connect(
                alias=self.name_default,
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db_name="default",  # Always connect to default first
            )
            self.logger.info(
                f"Connected to default Milvus database at {self.host}:{self.port}"
            )
            return True
        except MilvusException as e:
            self.logger.error(f"Failed to connect to default Milvus database: {e}")
            return False

    def create_database_if_does_not_exist(self) -> bool:
        """
        Create the database if it doesn't exist.

        Returns:
            bool: True if database creation is successful or already exists
        """
        try:
            # First connect to default database
            if not self.connect_to_default():
                return False

            existing_dbs = db.list_database(using=self.name_default)
            self.logger.debug(f"Existing databases: {existing_dbs}")
            if self.db_name in existing_dbs:
                self.logger.debug(f"Database '{self.db_name}' already exists")
                connections.disconnect(self.name_default)
                return True

            db.create_database(db_name=self.db_name, using=self.name_default)
            connections.disconnect(self.name_default)
            self.logger.info(f"Created database: {self.db_name}")
            return True
        except MilvusException as e:
            self.logger.error(f"Failed to create database: {e}")
            return False

    def initialize(self) -> bool:
        """
        Initialize the complete Milvus setup:
         - connect,
         - create collections,
         - indexes.

        Returns:
            bool: True if initialization is successful
        """
        if not self.create_database_if_does_not_exist():
            self.logger.error(
                f"Problem with database creation: '{self.db_name}' "
                f"at {self.host}:{self.port}"
            )
            return False

        if not self.connect():
            return False

        if not self.create_collections():
            return False

        if not self.create_indexes():
            return False

        self.logger.info("Milvus WordNet handler initialized successfully")
        return True

    @abstractmethod
    def create_indexes(self, index_type: str = "HNSW") -> bool:
        """
        Create indexes on vector fields for efficient similarity search.

        Args:
            index_type: Type of index to create for vector fields.
                       Defaults to "HNSW"

        Returns:
            bool: True if index creation is successful, False otherwise

        Note:
            Must be implemented by concrete subclasses
        """
        raise NotImplemented

    @abstractmethod
    def create_collections(self) -> bool:
        """
        Create all necessary collections with appropriate schemas.

        Returns:
            bool: True if collection creation is successful, False otherwise

        Note:
            Must be implemented by concrete subclasses
        """

        raise NotImplemented


class MilvusWordNetBaseHandler(_MilvusBaseInitializer, ABC):
    """
    Handler for managing Milvus collections for WordNet embeddings.
    Supports both synset and lexical unit embeddings with metadata.
    """

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

    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name: Name of the collection to check

        Returns:
            bool: True if a collection exists, False otherwise
        """
        try:
            return utility.has_collection(collection_name, using=self.conn_name)
        except MilvusException as e:
            self.logger.error(f"Error checking collection existence: {e}")
            return False

    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Dict containing collection info or None if error occurred
        """
        try:
            if not self.collection_exists(collection_name):
                return None

            collection = Collection(collection_name, using=self.conn_name)

            return {
                "name": collection_name,
                "schema": collection.schema,
                "num_entities": collection.num_entities,
                "is_empty": collection.is_empty,
                "indexes": collection.indexes,
            }
        except MilvusException as e:
            self.logger.error(f"Error getting collection info: {e}")
            return None
