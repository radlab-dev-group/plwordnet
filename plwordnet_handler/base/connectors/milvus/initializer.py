from abc import ABC, abstractmethod
from pymilvus import (
    db,
    connections,
    Collection,
    utility,
    MilvusException,
    CollectionSchema,
)
from typing import Optional

from pymysql import NotSupportedError

from plwordnet_handler.base.connectors.milvus.core.config import MilvusConfig
from plwordnet_handler.base.connectors.milvus.core.base_connector import (
    MilvusBaseConnector,
)
from plwordnet_handler.base.connectors.milvus.core.schema import (
    EmbeddingIndexType,
    PlwordnetMilvusSchema,
)


class _MilvusWordNetBaseInitializer(MilvusBaseConnector, ABC):
    name_default = "wordnet_connection_default"

    @abstractmethod
    def create_indexes(self, index_type: str = "IVF_FLAT") -> bool:
        """
        Create indexes on vector fields for efficient similarity search.

        Args:
            index_type: Type of index to create for vector fields.
                       Defaults to "IVF_FLAT"

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

    def _connect_to_default(self) -> bool:
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


class MilvusWordNetSchemaInitializer(_MilvusWordNetBaseInitializer):
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        user: str = "",
        password: str = "",
        db_name: str = "default",
        config: Optional[MilvusConfig] = None,
        syn_vector_dim: int = 1152,
        lu_vector_dim: int = 1152,
        lu_examples_vector_dim: int = 1152,
        log_level: str = "INFO",
    ):
        """
        Initialize the Milvus embeddings connector with database
        and schema configuration.

        Args:
            host: Milvus server hostname. Defaults to "localhost"
            port: Milvus server port. Defaults to 19530
            user: Username for authentication. Defaults to empty string
            password: Password for authentication. Defaults to empty string
            db_name: Name of the Milvus database. Defaults to "default"
            config: Optional MilvusConfig object for advanced configuration
            syn_vector_dim: Dimension size for synset embedding vectors.
            Defaults to 1152
            lu_vector_dim: Dimension size for "lexical unit embedding"
            vectors. Defaults to 1152
            lu_examples_vector_dim: Dimension size for a vector of examples
            belonging to lexical units. Defaults to 1152
            log_level: Logging level for the connector. Defaults to "INFO"
        """

        super().__init__(
            host=host,
            port=port,
            user=user,
            password=password,
            db_name=db_name,
            config=config,
            log_level=log_level,
            logger_name=__name__,
        )

        self.synset_schema = None
        self.synset_vector_dim = syn_vector_dim

        self.lu_schema = None
        self.lu_vector_dim = lu_vector_dim

        self.lu_examples_schema = None
        self.lu_example_vector_dim = lu_examples_vector_dim

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

    def create_database_if_does_not_exist(self) -> bool:
        """
        Create the database if it doesn't exist.

        Returns:
            bool: True if database creation is successful or already exists
        """
        try:
            # First connect to default database
            if not self._connect_to_default():
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

    def create_collections(self) -> bool:
        """
        Create both synset and lexical unit collections.

        Returns:
            bool: True if both collections are created successfully
        """
        try:
            self.__create_synset_collection()
            self.__create_lexical_unit_collection()
            self.__create_lexical_unit_examples_collection()
            return True
        except MilvusException as e:
            self.logger.error(f"Failed to create collections: {e}")
            return False

    def create_indexes(self, index_type: str = "IVF_FLAT") -> bool:
        """
        Create indexes on vector fields for both collections.

        Args:
            index_type: Type of index to create (HNSW, IVF_FLAT, etc.)
            Defaults to "IVF_FLAT"

        Returns:
            bool: True if indexes are created successfully
        """
        try:
            if index_type == "HNSW":
                index_params = EmbeddingIndexType.HNSW.copy()
            elif index_type == "IVF_FLAT":
                index_params = EmbeddingIndexType.IVF_FLAT.copy()
            else:
                raise NotSupportedError("Not supported index type")

            collections = [
                self.synset_collection_name,
                self.lu_collection_name,
                self.lu_examples_collection_name,
            ]

            for collection_name in collections:
                collection = Collection(collection_name, using=self.conn_name)
                collection.create_index(
                    field_name="embedding", index_params=index_params
                )
                collection.load()

            self.logger.info(f"Created {index_type} indexes on collections")
            return True
        except MilvusException as e:
            self.logger.error(f"Failed to create indexes: {e}")
            return False

    def __create_synset_schema(self) -> CollectionSchema:
        """
        Create a schema for synset embeddings collection.

        Returns:
            CollectionSchema: Schema for synset embeddings
        """
        self.synset_schema = PlwordnetMilvusSchema.Synset.create(
            emb_size=self.synset_vector_dim
        )
        return self.synset_schema

    def __create_lu_schema(self) -> CollectionSchema:
        """
        Create a schema for lexical unit embeddings' collection.

        Returns:
            CollectionSchema: Schema for lexical unit embeddings
        """
        self.lu_schema = PlwordnetMilvusSchema.LU.create(emb_size=self.lu_vector_dim)
        return self.lu_schema

    def __create_lu_examples_schema(self) -> CollectionSchema:
        """
        Create a schema for lexical unit examples embeddings' collection.

        Returns:
            CollectionSchema: Schema for lexical unit examples embeddings
        """
        self.lu_examples_schema = PlwordnetMilvusSchema.LUExample.create(
            emb_size=self.lu_example_vector_dim
        )
        return self.lu_examples_schema

    def __create_synset_collection(self):
        """
        Create a Milvus collection for synset embeddings if it doesn't exist.

        Checks if the synset collection already exists and creates it with the
        appropriate schema if not found. Logs the creation status.
        """

        if not utility.has_collection(
            self.synset_collection_name, using=self.conn_name
        ):
            synset_schema = self.__create_synset_schema()
            Collection(
                name=self.synset_collection_name,
                schema=synset_schema,
                using=self.conn_name,
            )
            self.logger.info(f"Created collection: {self.synset_collection_name}")
        else:
            self.logger.info(
                f"Collection {self.synset_collection_name} already exists"
            )

    def __create_lexical_unit_collection(self):
        """
        Create a Milvus collection for lexical unit embeddings if it doesn't exist.

        Checks if the lexical unit collection already exists and creates it with the
        appropriate schema if not found. Logs the creation status.
        """

        if not self.collection_exists(self.lu_collection_name):
            lu_schema = self.__create_lu_schema()
            Collection(
                name=self.lu_collection_name,
                schema=lu_schema,
                using=self.conn_name,
            )
            self.logger.info(f"Created collection: {self.lu_collection_name}")
        else:
            self.logger.info(f"Collection {self.lu_collection_name} already exists")

    def __create_lexical_unit_examples_collection(self):
        """
        Create a Milvus collection for lexical unit examples
        embeddings if it doesn't exist.

        Checks if the lexical unit examples collection already exists and creates
        it with the appropriate schema if not found. Logs the creation status.
        """

        if not utility.has_collection(
            self.lu_examples_collection_name, using=self.conn_name
        ):
            lu_examples_schema = self.__create_lu_examples_schema()
            Collection(
                name=self.lu_examples_collection_name,
                schema=lu_examples_schema,
                using=self.conn_name,
            )
            self.logger.info(
                f"Created collection: {self.lu_examples_collection_name}"
            )
        else:
            self.logger.info(
                f"Collection {self.lu_examples_collection_name} already exists"
            )
