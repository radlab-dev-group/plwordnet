import numpy as np

from pymilvus import MilvusException
from typing import Dict, Any, List, Union

from plwordnet_handler.base.connectors.milvus.core.schema import MAX_TEXT_LEN
from plwordnet_handler.base.connectors.milvus.core.config import MilvusConfig
from plwordnet_handler.base.connectors.milvus.core.base_connector import (
    MilvusBaseConnector,
)


class MilvusWordNetInsertHandler(MilvusBaseConnector):
    """
    Handler for inserting data into WordNet Milvus collections.
    Extends MilvusBaseConnector with insert capabilities.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the insert handler with same parameters as schema handler.
        """
        super().__init__(*args, **kwargs)

    def insert_synset_embeddings(
        self, data: List[Dict[str, Any]], batch_size: int = 1000
    ) -> bool:
        """
        Insert synset embeddings into the synset collection.

        Args:
            data: List of dictionaries containing synset data with keys:
                  - id: Synset ID (int)
                  - embedding: Embedding vector (list of floats)
                  - unitsstr: String representation of units (str)
                  - model_name: Name of the model used (str)
            batch_size: Number of records to insert in each batch

        Returns:
            bool: True if insertion is successful, False otherwise
        """
        try:
            collection = self._get_synset_collection()
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                entities = [
                    [item["id"] for item in batch],
                    [item["embedding"] for item in batch],
                    [item["unitsstr"] for item in batch],
                    [item["model_name"] for item in batch],
                    [item["type"] for item in batch],
                    [item["strategy"] for item in batch],
                ]
                collection.insert(entities)
            return True

        except MilvusException as e:
            self.logger.error(f"Failed to insert synset embeddings: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during synset insertion: {e}")
            return False

    def insert_lu_embeddings(
        self, data: List[Dict[str, Any]], batch_size: int = 1000
    ) -> bool:
        """
        Insert lexical unit embeddings into the LU collection.

        Args:
            data: List of dictionaries containing LU data with keys:
                  - id: Lexical Unit ID (int)
                  - embedding: Embedding vector (list of floats)
                  - lemma: Lemma of the lexical unit (str)
                  - pos: Part of speech (int)
                  - domain: Domain information (int)
                  - variant: Variant information (int)
                  - model_name: Name of the model used (str)
            batch_size: Number of records to insert in each batch

        Returns:
            bool: True if insertion is successful, False otherwise
        """
        try:
            collection = self._get_lu_collection()
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                entities = [
                    [item["id"] for item in batch],
                    [item["embedding"] for item in batch],
                    [item["lemma"] for item in batch],
                    [item["pos"] for item in batch],
                    [item["domain"] for item in batch],
                    [item["variant"] for item in batch],
                    [item["model_name"] for item in batch],
                    [item["type"] for item in batch],
                    [item["strategy"] for item in batch],
                ]
                collection.insert(entities)
            return True

        except MilvusException as e:
            self.logger.error(f"Failed to insert LU embeddings: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during LU insertion: {e}")
            return False

    def insert_lu_examples_embeddings(
        self, data: List[Dict[str, Any]], batch_size: int = 1000
    ) -> bool:
        """
        Insert lexical unit examples embeddings into the LU examples collection.

        Args:
            data: List of dictionaries containing LU examples data with keys:
                  - id: Lexical Unit ID (int)
                  - embedding: Embedding vector (list of floats)
                  - example: Example text (str)
                  - model_name: Name of the model used (str)
            batch_size: Number of records to insert in each batch

        Returns:
            bool: True if insertion is successful, False otherwise
        """
        try:
            collection = self._get_lu_examples_collection()
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                entities = [
                    [item["id"] for item in batch],
                    [item["embedding"] for item in batch],
                    [item["example"][:MAX_TEXT_LEN] for item in batch],
                    [item["model_name"] for item in batch],
                    [item["type"] for item in batch],
                    [item["strategy"] for item in batch],
                ]
                collection.insert(entities)
            return True

        except MilvusException as e:
            self.logger.error(f"Failed to insert LU examples embeddings: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during LU examples insertion: {e}")
            return False

    def insert_single_synset(
        self,
        synset_id: int,
        embedding: Union[List[float], np.ndarray],
        unitsstr: str,
        model_name: str,
    ) -> bool:
        """
        Insert a single synset embedding.

        Args:
            synset_id: Synset ID
            embedding: Embedding vector
            unitsstr: String representation of units
            model_name: Name of the model used

        Returns:
            bool: True if insertion is successful, False otherwise
        """
        data = [
            {
                "id": synset_id,
                "embedding": (
                    embedding.tolist()
                    if isinstance(embedding, np.ndarray)
                    else embedding
                ),
                "unitsstr": unitsstr,
                "model_name": model_name,
            }
        ]
        return self.insert_synset_embeddings(data, batch_size=1)

    def insert_single_lu(
        self,
        lu_id: int,
        embedding: Union[List[float], np.ndarray],
        lemma: str,
        pos: int,
        domain: int,
        variant: int,
        model_name: str,
    ) -> bool:
        """
        Insert a single lexical unit embedding.

        Args:
            lu_id: Lexical unit ID
            embedding: Embedding vector
            lemma: Lemma of the lexical unit
            pos: Part of speech
            domain: Domain information
            variant: Variant information
            model_name: Name of the model used

        Returns:
            bool: True if insertion is successful, False otherwise
        """
        data = [
            {
                "id": lu_id,
                "embedding": (
                    embedding.tolist()
                    if isinstance(embedding, np.ndarray)
                    else embedding
                ),
                "lemma": lemma,
                "pos": pos,
                "domain": domain,
                "variant": variant,
                "model_name": model_name,
            }
        ]
        return self.insert_lu_embeddings(data, batch_size=1)

    def insert_single_lu_example(
        self,
        lu_id: int,
        embedding: Union[List[float], np.ndarray],
        example: str,
        model_name: str,
    ) -> bool:
        """
        Insert a single lexical unit example embedding.

        Args:
            lu_id: Lexical unit ID
            embedding: Embedding vector
            example: Example text
            model_name: Name of the model used

        Returns:
            bool: True if insertion is successful, False otherwise
        """
        data = [
            {
                "id": lu_id,
                "embedding": (
                    embedding.tolist()
                    if isinstance(embedding, np.ndarray)
                    else embedding
                ),
                "example": example,
                "model_name": model_name,
            }
        ]
        return self.insert_lu_examples_embeddings(data, batch_size=1)

    @classmethod
    def from_config_file(cls, config_path: str) -> "MilvusWordNetInsertHandler":
        """
        Create a MilvusWordNetBaseHandler instance from a configuration file.

        Args:
            config_path: Path to the JSON configuration file

        Returns:
            MilvusWordNetSchemaHandler: Handler instance with loaded configuration
        """
        config = MilvusConfig.from_json_file(config_path)
        return cls(config=config)

    def get_all_lu_examples(self, limit: int = 0) -> List[Dict[str, Any]]:
        """
        Retrieve **all** lexical unit examples from the LU examples collection.

        Args:
            limit: Optional maximum number of examples to return.
                   ``0`` (default) means no explicit limit – all records are fetched.
                   Use with caution on very large collections.

        Returns:
            List of dictionaries, each containing:
                - id: Lexical Unit ID
                - embedding: Embedding vector (list of floats)
                - example: Example text (truncated to ``MAX_TEXT_LEN``)
                - model_name: Name of the model used
                - type: (optional) type field stored in the collection
                - strategy: (optional) strategy field stored in the collection
        """
        try:
            collection = self._get_lu_examples_collection()
            results = collection.query(
                expr="",
                output_fields=[
                    "id",
                    "embedding",
                    "example",
                    "model_name",
                    "type",
                    "strategy",
                ],
                limit=limit if limit > 0 else None,
            )
            return [
                {
                    "id": r.get("id"),
                    "embedding": r.get("embedding"),
                    "example": r.get("example"),
                    "model_name": r.get("model_name"),
                    "type": r.get("type"),
                    "strategy": r.get("strategy"),
                }
                for r in results
            ]

        except MilvusException as e:
            self.logger.error(f"Failed to retrieve all LU examples: {e}")
            return []
        except Exception as e:
            self.logger.error(
                f"Unexpected error during all LU examples retrieval: {e}"
            )
            return []
