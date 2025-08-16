from pymilvus import (
    CollectionSchema,
    FieldSchema,
    DataType,
)

MAX_TEXT_LEN = 6000


class EmbeddingIndexType:
    IVF_FLAT = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1536},
    }

    HNSW = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 64},
    }


class PlwordnetMilvusSchema:
    class LU:
        @classmethod
        def create(cls, emb_size: int) -> CollectionSchema:
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
        def create(cls, emb_size: int) -> CollectionSchema:
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
        def create(cls, emb_size: int) -> CollectionSchema:
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
