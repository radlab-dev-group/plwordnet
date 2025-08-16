from abc import ABC
from pymilvus import (
    connections,
    Collection,
    utility,
    MilvusException,
)
from typing import Dict, Any, Optional

from plwordnet_handler.utils.logger import prepare_logger
from plwordnet_handler.base.connectors.milvus.config import MilvusConfig


class MilvusBaseConnector(ABC):
    conn_name = "wordnet_connection"

    lu_collection_name = "lu_embeddings"
    synset_collection_name = "synset_embeddings"
    lu_examples_collection_name = "lu_examples_embeddings"

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

        self.lu_collection = None
        self.synset_collection = None
        self.lu_examples_collection = None

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

    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of collections.

        Returns:
            Dict containing status of collections
        """
        return {
            "synset_collection": self.get_collection_info(
                self.synset_collection_name
            ),
            "lu_collection": self.get_collection_info(self.lu_collection_name),
            "lu_examples_collection": self.get_collection_info(
                self.lu_examples_collection_name
            ),
            "connection": self.conn_name,
        }

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

    def _get_lu_collection(self):
        """
        Get or create a lazy-initialized lexical unit collection instance.

        Returns the cached lexical unit collection if it exists, otherwise creates
        a new Collection object connected to the lexical unit collection using
        the configured connection name and collection name.

        Returns:
            Collection: Milvus collection instance for lexical units
        """

        if self.lu_collection is None:
            self.lu_collection = Collection(
                self.lu_collection_name, using=self.conn_name
            )
        return self.lu_collection

    def _get_synset_collection(self):
        """
        Get or create a lazy-initialized synset collection instance.

        Returns the cached synset collection if it exists, otherwise creates
        a new Collection object connected to the synset collection using
        the configured connection name and collection name.

        Returns:
            Collection: Milvus collection instance for synsets
        """

        if self.synset_collection is None:
            self.synset_collection = Collection(
                self.synset_collection_name, using=self.conn_name
            )
        return self.synset_collection

    def _get_lu_examples_collection(self):
        """
        Get or create a lazy-initialized lexical unit examples collection instance.

        Returns the cached lexical unit examples collection if it exists, otherwise
        creates a new Collection object connected to the lexical units examples
        collection using the configured connection name and collection name.

        Returns:
            Collection: Milvus collection instance for lexical unit examples
        """

        if self.lu_examples_collection is None:
            self.lu_examples_collection = Collection(
                self.lu_examples_collection_name, using=self.conn_name
            )
        return self.lu_examples_collection
