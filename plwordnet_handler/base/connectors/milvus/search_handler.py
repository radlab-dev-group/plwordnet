from pymilvus import MilvusException
from typing import Dict, Any, List, Optional

from plwordnet_handler.base.connectors.milvus.config import MilvusConfig
from plwordnet_handler.base.connectors.milvus.core.base_connector import (
    MilvusBaseConnector,
)


class _MilvusSearchFields:
    """
    Configuration class for Milvus search output fields.

    Defines the standard field sets that should be returned from Milvus queries
    for different types of WordNet embeddings, ensuring consistent data retrieval
    across different search operations.
    """

    # List of base_lu_embedding Milvus fields returned from the query
    LU_EMBEDDING_OUT_FIELDS = [
        "id",
        "lu_id",
        "embedding",
        "lemma",
        "pos",
        "domain",
        "variant",
        "model_name",
    ]

    # List of base_lu_embedding_examples Milvus fields returned from the query
    LU_EXAMPLES_OUT_FIELDS = ["id", "lu_id", "embedding", "example", "model_name"]


class MilvusWordNetSearchHandler(MilvusBaseConnector):
    """
    Handler for reading data from WordNet Milvus collections.
    Extends MilvusBaseConnector with read and search capabilities.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the read handler with same parameters as base connector.
        """
        super().__init__(*args, **kwargs)

    def get_lexical_unit_embedding(self, lu_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve lexical unit embedding by ID from the lexical unit collection.
        In case when more than a single embedding is found, return the first one
        and log a warning message with information about it.

        Args:
            lu_id: Lexical unit ID to search for

        Returns:
            Optional[Dict[str, Any]] Dictionary containing search results
            with embeddings and metadata, or None if any error occurs.

        Note:
            Searches by exact ID match in the lexical unit collection. Returns
            full search results including distance scores and all metadata fields.
        """
        try:
            collection = self._get_lu_collection()

            expr = f"lu_id == {lu_id}"
            results = collection.query(
                expr=expr,
                output_fields=_MilvusSearchFields.LU_EMBEDDING_OUT_FIELDS,
            )
            if not results:
                return None

            if len(results) > 1:
                self.logger.warning(
                    f"More than one result found for {lu_id}. "
                    f"Found {len(results)} embeddings. First one is returned."
                )
            return results[0]
        except MilvusException as e:
            self.logger.error(
                f"Failed to retrieve lexical unit embedding for ID {lu_id}: {e}"
            )
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during lexical unit retrieval: {e}")
            return None

    def get_lexical_unit_examples_embeddings(
        self,
        lu_id: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve lexical unit examples embeddings by LEX_ID
        from the lexical units examples collection.

        Args:
            lu_id: Lexical unit ID (LEX_ID) to search for in lexical
            units examples collection

        Returns:
            Optional[List[Dict[str, Any]]]: List of dictionaries containing search
            results with embeddings and metadata, or None if any error occurs.

        Note:
            Searches by exact LEX_ID match in the lexical unit examples
            collection. Returns full search results including distance
            scores and all metadata fields.
        """
        try:
            collection = self._get_lu_examples_collection()
            expr = f"lu_id == {lu_id}"
            results = collection.query(
                expr=expr,
                output_fields=_MilvusSearchFields.LU_EXAMPLES_OUT_FIELDS,
            )
            if not results:
                return None
            return results
        except MilvusException as e:
            self.logger.error(
                f"Failed to retrieve lexical unit examples for LEX_ID {lu_id}: {e}"
            )
            return None
        except Exception as e:
            self.logger.error(
                f"Unexpected error during lexical unit examples retrieval: {e}"
            )
            return None

    def get_lexical_units_embeddings(
        self, lu_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve lexical units embeddings by list of IDs
        from the lexical unit collection.

        Args:
            lu_ids: List of lexical unit IDs to search for

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing
            search results with embeddings and metadata for all found IDs

        Note:
            Searches by exact ID match in the lexical unit collection.
            Returns combined results for all requested IDs.
        """
        if not lu_ids:
            self.logger.warning("Empty lu_ids list provided")
            return []

        if len(lu_ids) == 1:
            lu_e = self.get_lexical_unit_embedding(lu_id=lu_ids[0])
            if lu_e is not None:
                return [lu_e]
            return []

        try:
            collection = self._get_lu_collection()
            ids_str = ", ".join(str(lu_id) for lu_id in lu_ids)
            expr = f"lu_id in [{ids_str}]"
            results = collection.query(
                expr=expr, output_fields=_MilvusSearchFields.LU_EMBEDDING_OUT_FIELDS
            )
            if not results:
                return []
            return results
        except MilvusException as e:
            self.logger.error(
                f"Failed to retrieve lexical unit embeddings for IDs {lu_ids}: {e}"
            )
        except Exception as e:
            self.logger.error(f"Unexpected error during lexical unit retrieval: {e}")
        return []

    def get_lexical_units_examples_embedding(
        self,
        lex_ids: List[int],
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve lexical unit examples embeddings by list of LEX_IDs
        from the lexical units examples collection.

        Args:
            lex_ids: List of lexical unit IDs (LEX_IDs) to search
            for in the lexical units examples collection

        Returns:
            List[List[Dict[str, Any]]]: List of dictionaries containing
            search results with embeddings and metadata for all found LEX_IDs

        Note:
            Searches by exact LEX_ID match in the lexical unit examples collection.
            Returns combined results for all requested LEX_IDs.
        """
        if not lex_ids:
            self.logger.warning("Empty lex_ids list provided")
            return []

        if len(lex_ids) == 1:
            lu_e = self.get_lexical_unit_examples_embeddings(lu_id=lex_ids[0])
            if lu_e is not None:
                return [lu_e]
            return []

        try:
            collection = self._get_lu_examples_collection()
            ids_str = ", ".join(str(lex_id) for lex_id in lex_ids)
            expr = f"lu_id in [{ids_str}]"
            results = collection.query(
                expr=expr, output_fields=_MilvusSearchFields.LU_EXAMPLES_OUT_FIELDS
            )
            if not results:
                return []
            return results
        except MilvusException as e:
            self.logger.error(
                f"Failed to retrieve lexical unit examples for LEX_IDs {lex_ids}: {e}"
            )
            return []
        except Exception as e:
            self.logger.error(
                f"Unexpected error during lexical unit examples retrieval: {e}"
            )
            return []

    @classmethod
    def from_config_file(cls, config_path: str) -> "MilvusWordNetSearchHandler":
        """
        Create a MilvusWordNetReadHandler instance from a configuration file.

        Args:
            config_path: Path to the JSON configuration file

        Returns:
            MilvusWordNetSearchHandler: Handler instance with loaded configuration

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            json.JSONDecodeError: If the JSON is invalid
            KeyError: If required configuration keys are missing
        """
        config = MilvusConfig.from_json_file(config_path)
        return cls(config=config)
