from pymilvus import (
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    MilvusException,
)
from typing import Dict, Any, Optional

from plwordnet_handler.base.connectors.milvus.config import MilvusConfig
from plwordnet_handler.base.connectors.milvus.base import MilvusWordNetBaseHandler


class _SchemaDef:
    class LU:
        @classmethod
        def create_schema(cls, emb_size: int) -> CollectionSchema:
            """
            Create a schema for lexical unit embeddings' collection.

            Returns:
                CollectionSchema: Schema for lexical unit embeddings
            """
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=False,
                    description="Lexical Unit ID from WordNet",
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=emb_size,
                    description="Lexical unit embedding vector",
                ),
                FieldSchema(
                    name="lemma",
                    dtype=DataType.VARCHAR,
                    max_length=200,
                    description="Lemma of the lexical unit",
                ),
                FieldSchema(
                    name="pos", dtype=DataType.INT32, description="Part of speech"
                ),
                FieldSchema(
                    name="domain",
                    dtype=DataType.INT32,
                    description="Domain information",
                ),
                FieldSchema(
                    name="variant",
                    dtype=DataType.INT32,
                    description="Variant information",
                ),
            ]

            schema = CollectionSchema(
                fields=fields,
                description="Lexical unit embeddings with metadata",
                enable_dynamic_field=True,
            )

            return schema

    class Synset:
        @classmethod
        def create_schema(cls, emb_size: int) -> CollectionSchema:
            """
            Create a schema for synset embeddings collection.

            Returns:
                CollectionSchema: Schema for synset embeddings
            """
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=False,
                    description="Synset ID from WordNet",
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=emb_size,
                    description="Synset embedding vector",
                ),
                FieldSchema(
                    name="unitsstr",
                    dtype=DataType.VARCHAR,
                    max_length=500,
                    description="String representation of units in synset",
                ),
            ]

            schema = CollectionSchema(
                fields=fields,
                description="Synset embeddings with metadata",
                enable_dynamic_field=True,
            )

            return schema


class MilvusWordNetSchemaHandler(MilvusWordNetBaseHandler):
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
        syn_vector_dim: int = 1024,
        lu_vector_dim: int = 1024,
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
            Defaults to 1024
            lu_vector_dim: Dimension size for "lexical unit embedding" vectors.
            Defaults to 1024
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
        self.synset_collection_name = "synset_embeddings"
        self.synset_vector_dim = syn_vector_dim

        self.lu_schema = None
        self.lu_collection_name = "lu_embeddings"
        self.lu_vector_dim = lu_vector_dim

    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name: Name of the collection to check

        Returns:
            bool: True if a collection exists, False otherwise
        """
        try:
            return utility.has_collection(
                collection_name, using=self.connection_alias
            )
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

            collection = Collection(collection_name, using=self.connection_alias)

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

    def initialize(self) -> bool:
        """
        Initialize the complete Milvus setup:
         - connect,
         - create collections,
         - indexes.

        Returns:
            bool: True if initialization is successful
        """
        if not self.connect():
            return False

        if not self.__create_collections():
            return False

        if not self.__create_indexes():
            return False

        self.logger.info("Milvus WordNet handler initialized successfully")
        return True

    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of both collections.

        Returns:
            Dict containing status of both collections
        """
        return {
            "synset_collection": self.get_collection_info(
                self.synset_collection_name
            ),
            "lu_collection": self.get_collection_info(self.lu_collection_name),
            "connection": self.connection_alias,
        }

    def __create_collections(self) -> bool:
        """
        Create both synset and lexical unit collections.

        Returns:
            bool: True if both collections are created successfully
        """
        try:
            self.__create_synset_collection()
            self.__create_lexical_unit_collection()
            return True
        except MilvusException as e:
            self.logger.error(f"Failed to create collections: {e}")
            return False

    def __create_indexes(self, index_type: str = "HNSW") -> bool:
        """
        Create indexes on vector fields for both collections.

        Args:
            index_type: Type of index to create (HNSW, IVF_FLAT, etc.)

        Returns:
            bool: True if indexes are created successfully
        """
        try:
            index_params = {
                "metric_type": "L2",
                "index_type": index_type,
                "params": (
                    {"M": 8, "efConstruction": 64} if index_type == "HNSW" else {}
                ),
            }

            # Create index for synset collection
            synset_collection = Collection(
                self.synset_collection_name, using=self.connection_alias
            )
            synset_collection.create_index(
                field_name="embedding", index_params=index_params
            )

            # Create an index for LU collection
            lu_collection = Collection(
                self.lu_collection_name, using=self.connection_alias
            )
            lu_collection.create_index(
                field_name="embedding", index_params=index_params
            )

            self.logger.info(f"Created {index_type} indexes on both collections")
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
        self.synset_schema = _SchemaDef.Synset.create_schema(
            emb_size=self.synset_vector_dim
        )
        return self.synset_schema

    def __create_lu_schema(self) -> CollectionSchema:
        """
        Create a schema for lexical unit embeddings' collection.

        Returns:
            CollectionSchema: Schema for lexical unit embeddings
        """
        self.lu_schema = _SchemaDef.LU.create_schema(emb_size=self.lu_vector_dim)
        return self.lu_schema

    def __create_synset_collection(self):
        """
        Create a Milvus collection for synset embeddings if it doesn't exist.

        Checks if the synset collection already exists and creates it with the
        appropriate schema if not found. Logs the creation status.
        """

        if not utility.has_collection(
            self.synset_collection_name, using=self.connection_alias
        ):
            synset_schema = self.__create_synset_schema()
            Collection(
                name=self.synset_collection_name,
                schema=synset_schema,
                using=self.connection_alias,
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

        if not utility.has_collection(
            self.lu_collection_name, using=self.connection_alias
        ):
            lu_schema = self.__create_lu_schema()
            Collection(
                name=self.lu_collection_name,
                schema=lu_schema,
                using=self.connection_alias,
            )
            self.logger.info(f"Created collection: {self.lu_collection_name}")
        else:
            self.logger.info(f"Collection {self.lu_collection_name} already exists")
