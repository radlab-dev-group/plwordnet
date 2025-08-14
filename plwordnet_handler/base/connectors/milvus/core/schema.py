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
from plwordnet_handler.base.connectors.milvus.core.base import (
    MilvusWordNetBaseHandler,
)

MAX_TEXT_LEN = 6000


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
                    auto_id=True,
                    description="Lexical Unit Embedding ID",
                ),
                FieldSchema(
                    name="lu_id",
                    dtype=DataType.INT64,
                    description="Lexical Unit ID from Słowosieć",
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
                    max_length=510,
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
                FieldSchema(
                    name="model_name",
                    dtype=DataType.VARCHAR,
                    max_length=512,
                    description="Name of model used to generate embeddings",
                ),
            ]

            schema = CollectionSchema(
                fields=fields,
                description="Lexical unit embeddings with metadata",
                enable_dynamic_field=True,
            )

            return schema

    class LUExample:
        @classmethod
        def create_schema(cls, emb_size: int) -> CollectionSchema:
            """
            Create a schema for lexical unit examples embeddings' collection.

            Returns:
                CollectionSchema: Schema for lexical unit examples embeddings
            """
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=True,
                    description="Lexical Unit Example Embedding ID",
                ),
                FieldSchema(
                    name="lu_id",
                    dtype=DataType.INT64,
                    description="Lexical Unit ID from Słowosieć",
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=emb_size,
                    description="Lexical unit example embedding vector",
                ),
                FieldSchema(
                    name="example",
                    dtype=DataType.VARCHAR,
                    max_length=MAX_TEXT_LEN,
                    description="lexical unit example (definition, sentiment, etc.)",
                ),
                FieldSchema(
                    name="model_name",
                    dtype=DataType.VARCHAR,
                    max_length=512,
                    description="Name of model used to generate embeddings",
                ),
            ]

            schema = CollectionSchema(
                fields=fields,
                description="Lexical unit examples embeddings with metadata",
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
                    auto_id=True,
                    description="Synset Embedding ID",
                ),
                FieldSchema(
                    name="syn_id",
                    dtype=DataType.INT64,
                    description="Synset ID from Słowosieć",
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
                    max_length=1024,
                    description="String representation of units in synset",
                ),
                FieldSchema(
                    name="model_name",
                    dtype=DataType.VARCHAR,
                    max_length=512,
                    description="Name of model used to generate embeddings",
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
        self.synset_collection_name = "synset_embeddings"
        self.synset_vector_dim = syn_vector_dim

        self.lu_schema = None
        self.lu_collection_name = "lu_embeddings"
        self.lu_vector_dim = lu_vector_dim

        self.lu_examples_schema = None
        self.lu_examples_collection_name = "lu_examples_embeddings"
        self.lu_example_vector_dim = lu_examples_vector_dim

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
            "lu_examples_collection": self.get_collection_info(
                self.lu_examples_collection_name
            ),
            "connection": self.conn_name,
        }

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
            index_params = {
                "metric_type": "L2",
                "index_type": index_type,
                "params": (
                    {"M": 8, "efConstruction": 64}
                    if index_type == "HNSW"
                    else {"nlist": 1536}
                ),
            }

            collections = [
                (self.synset_collection_name, "synset"),
                (self.lu_collection_name, "LU"),
                (self.lu_examples_collection_name, "LU examples"),
            ]

            for collection_name, desc in collections:
                collection = Collection(collection_name, using=self.conn_name)
                collection.create_index(
                    field_name="embedding", index_params=index_params
                )
                collection.load()
            #
            # # Create index for synset collection
            # synset_collection = Collection(
            #     self.synset_collection_name, using=self.conn_name
            # )
            # synset_collection.create_index(
            #     field_name="embedding", index_params=index_params
            # )
            #
            # # Create an index for LU collection
            # lu_collection = Collection(self.lu_collection_name, using=self.conn_name)
            # lu_collection.create_index(
            #     field_name="embedding", index_params=index_params
            # )
            #
            # # Create an index for LU examples collection
            # lu_examples_collection = Collection(
            #     self.lu_examples_collection_name, using=self.conn_name
            # )
            # lu_examples_collection.create_index(
            #     field_name="embedding", index_params=index_params
            # )

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

    def __create_lu_examples_schema(self) -> CollectionSchema:
        """
        Create a schema for lexical unit examples embeddings' collection.

        Returns:
            CollectionSchema: Schema for lexical unit examples embeddings
        """
        self.lu_examples_schema = _SchemaDef.LUExample.create_schema(
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

        if not utility.has_collection(self.lu_collection_name, using=self.conn_name):
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
