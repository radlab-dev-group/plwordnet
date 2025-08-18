import torch
import numpy as np

from pymilvus import MilvusException
from typing import Dict, Any, List, Optional, Union

from plwordnet_handler.base.connectors.milvus.core.config import MilvusConfig
from plwordnet_handler.base.connectors.milvus.core.search_fields import (
    MilvusSearchFields,
)
from plwordnet_handler.base.connectors.milvus.search_handler import (
    MilvusWordNetSearchHandler,
)


class MilvusWordNetSemanticSearchHandler(MilvusWordNetSearchHandler):
    """
    Handler for semantic similarity search operations on WordNet Milvus collections.

    Extends MilvusWordNetSearchHandler with vector similarity search capabilities,
    allowing for finding semantically similar lexical units, synsets, and examples
    based on query_embedding vectors. Supports configurable similarity metrics and
    optional filtering of results.
    """

    @classmethod
    def from_config_file(
        cls, config_path: str
    ) -> "MilvusWordNetSemanticSearchHandler":
        """
        Create a MilvusWordNetSemanticSearchHandler instance
        from a configuration file.

        Loads Milvus configuration from a JSON file and creates a new instance of the
        semantic search handler with the loaded configuration settings. This factory
        method provides a convenient way to initialize the handler with predefined
        connection parameters.

        Args:
            config_path: Path to the JSON configuration file containing Milvus
            connection settings (host, port, credentials, etc.)

        Returns:
            MilvusWordNetSemanticSearchHandler: Handler instance configured with
            settings from the specified configuration file

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            json.JSONDecodeError: If the JSON configuration file is invalid
            KeyError: If required configuration keys are missing from the file
        """
        config = MilvusConfig.from_json_file(config_path)
        return cls(config=config)

    def search_most_similar_lu(
        self,
        query_embedding: Union[List[float], np.ndarray, torch.Tensor],
        top_k: int,
        metric_type: str = "COSINE",
        filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Search for most similar lexical unit embeddings using vector similarity.

        Performs a vector similarity search in the lexical unit collection to find
        the most semantically similar lexical units to the provided query_embedding
        vector. Results are ranked by similarity score and limited to top_k entries.

        Args:
            query_embedding: Query query_embedding vector as a list of floats
            or numpy array or torch.Tensor
            top_k: Maximum number of similar results to return
            metric_type: Similarity metric to use (default: "COSINE")
            filters: Optional dictionary of field-value pairs for filtering results

        Returns:
            Optional[List[Dict[str, Any]]]: List of dictionaries containing similar
            lexical unit embeddings with metadata and similarity scores, or None
            if an error occurs

        Note:
            Returns lexical unit data including id, lu_id, query_embedding, lemma, pos,
            domain, variant, model_name, type, and strategy fields.
        """
        try:
            collection = self._get_lu_collection()
            return self._execute_search_query_on_collection(
                collection=collection,
                query_embedding=query_embedding,
                top_k=top_k,
                output_fields=MilvusSearchFields.LU_EMBEDDING_OUT_FIELDS,
                metric_type=metric_type,
                filters=filters,
            )
        except MilvusException as e:
            self.logger.error(f"Failed to search similar LU embeddings: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during LU similarity search: {e}")
            return None

    def search_most_similar_lu_examples(
        self,
        query_embedding: Union[List[float], np.ndarray, torch.Tensor],
        top_k: int,
        metric_type: str = "COSINE",
        filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Search for most similar lexical unit example embeddings
        using vector similarity.

        Performs a vector similarity search in the lexical unit examples collection
        to find the most semantically similar examples to the provided
        query_embedding vector. Results are ranked by similarity score
        and limited to top_k entries.

        Args:
            query_embedding: Query query_embedding vector as
            a list of floats or numpy array or torch.Tensor
            top_k: Maximum number of similar results to return
            metric_type: Similarity metric to use (default: "COSINE")
            filters: Optional dictionary of field-value pairs for filtering results

        Returns:
            Optional[List[Dict[str, Any]]]: List of dictionaries containing similar
            lexical unit example embeddings with metadata and similarity scores,
            or None if an error occurs

        Note:
            Returns lexical unit example data including id, lu_id, query_embedding,
            example, model_name, type, and strategy fields.
        """
        try:
            collection = self._get_lu_examples_collection()
            return self._execute_search_query_on_collection(
                collection=collection,
                query_embedding=query_embedding,
                top_k=top_k,
                output_fields=MilvusSearchFields.LU_EXAMPLES_OUT_FIELDS,
                metric_type=metric_type,
                filters=filters,
            )
        except MilvusException as e:
            self.logger.error(f"Failed to search similar LU examples: {e}")
            return None
        except Exception as e:
            self.logger.error(
                f"Unexpected error during LU examples similarity search: {e}"
            )
            return None

    def search_most_similar_synsets(
        self,
        query_embedding: Union[List[float], np.ndarray, torch.Tensor],
        top_k: int,
        metric_type: str = "COSINE",
        filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Search for most similar synset embeddings using vector similarity.

        Performs a vector similarity search in the synset collection to find
        the most semantically similar synsets to the provided query_embedding vector.
        Results are ranked by similarity score and limited to top_k entries.

        Args:
            query_embedding: Query query_embedding vector as
            a list of floats or numpy array or torch.Tensor
            top_k: Maximum number of similar results to return
            metric_type: Similarity metric to use (default: "COSINE")
            filters: Optional dictionary of field-value pairs for filtering results

        Returns:
            Optional[List[Dict[str, Any]]]: List of dictionaries containing similar
            synset embeddings with metadata and similarity scores, or None
            if an error occurs

        Note:
            Returns synset data including id, syn_id, query_embedding, unitsstr,
            model_name, type, and strategy fields.
        """
        try:
            collection = self._get_synset_collection()
            return self._execute_search_query_on_collection(
                collection=collection,
                query_embedding=query_embedding,
                top_k=top_k,
                output_fields=MilvusSearchFields.SYN_OUT_FIELDS,
                metric_type=metric_type,
                filters=filters,
            )
        except MilvusException as e:
            self.logger.error(f"Failed to search similar synsets: {e}")
            return None
        except Exception as e:
            self.logger.error(
                f"Unexpected error during synset similarity search: {e}"
            )
            return None

    def _execute_search_query_on_collection(
        self,
        collection,
        query_embedding: Union[List[float], np.ndarray, torch.Tensor],
        top_k: int,
        output_fields: List[str],
        metric_type: str = "COSINE",
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a vector similarity search query on a Milvus collection.

        Performs the actual vector similarity search operation on the specified
        collection using the provided query_embedding vector and search parameters.
        Handles query_embedding format conversion and filter expression construction.

        Args:
            collection: Milvus collection object to search in
            query_embedding: Query query_embedding vector
            as list of floats or numpy array or torch.Tensor
            top_k: Maximum number of similar results to return
            output_fields: List of field names to include in search results
            metric_type: Similarity metric to use (default: "COSINE")
            filters: Optional dictionary of field-value pairs for filtering results

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing search results
            with embeddings, metadata, and similarity scores

        Note:
            Converts numpy arrays to lists for Milvus compatibility. Constructs
            filter expressions from the "filters" dictionary using field-value
            equality conditions joined with logical AND operators.
        """

        query_embedding = self.__proper_query_embedding(
            query_embedding=query_embedding
        )

        search_params = {"metric_type": metric_type, "params": {"nprobe": 10}}

        expr = None
        if filters:
            filter_conditions = []
            for field, value in filters.items():
                if isinstance(value, str):
                    filter_conditions.append(f'{field} == "{value}"')
                else:
                    filter_conditions.append(f"{field} == {value}")
            expr = " && ".join(filter_conditions)

        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=output_fields,
        )

        return [hit.entity.to_dict() for hit in results[0]]

    @staticmethod
    def __proper_query_embedding(query_embedding):
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        elif isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.tolist()
        return query_embedding
