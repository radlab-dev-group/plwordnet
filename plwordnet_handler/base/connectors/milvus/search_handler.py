from pymilvus import MilvusException
from typing import Dict, Any, List, Optional

from plwordnet_handler.base.connectors.milvus.core.config import MilvusConfig
from plwordnet_handler.base.connectors.milvus.core.search_fields import (
    MilvusSearchFields,
)
from plwordnet_handler.base.connectors.milvus.core.base_connector import (
    MilvusBaseConnector,
)


class MilvusWordNetSearchHandler(MilvusBaseConnector):
    """
    Handler for reading data from WordNet Milvus collections.
    Extends MilvusBaseConnector with read and search capabilities.
    """

    # Factor for limit when the list of lexical units embeddings is returned
    LU_LIMIT_FACTOR = 100

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
                output_fields=MilvusSearchFields.LU_EMBEDDING_OUT_FIELDS,
                limit=self.LU_LIMIT_FACTOR,
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
        Retrieve lexical unit examples embeddings by lu_id
        from the lexical units examples collection.

        Args:
            lu_id: Lexical unit ID (lu_id) to search for in lexical
            units examples collection

        Returns:
            Optional[List[Dict[str, Any]]]: List of dictionaries containing search
            results with embeddings and metadata, or None if any error occurs.

        Note:
            Searches by exact lu_id match in the lexical unit examples
            collection. Returns full search results including distance
            scores and all metadata fields.
        """
        try:
            collection = self._get_lu_examples_collection()
            expr = f"lu_id == {lu_id}"
            results = collection.query(
                expr=expr,
                output_fields=MilvusSearchFields.LU_EXAMPLES_OUT_FIELDS,
                limit=self.LU_LIMIT_FACTOR,
            )
            if not results:
                return None
            return results
        except MilvusException as e:
            self.logger.error(
                f"Failed to retrieve lexical unit examples for lu_id {lu_id}: {e}"
            )
            return None
        except Exception as e:
            self.logger.error(
                f"Unexpected error during lexical unit examples retrieval: {e}"
            )
            return None

    def get_lexical_units_embeddings(
        self, lu_ids: List[int], map_to_lexical_units: bool = False
    ) -> List[Dict[str, Any]] | Dict[Any, List[Dict[str, Any]]]:
        """
        Retrieve lexical units embeddings by list of IDs
        from the lexical unit collection.

        Args:
            lu_ids: List of lexical unit IDs to search for
            map_to_lexical_units: If set to True, then the dictionary of
            an embedding list belonging to the lexical units will be returned

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing
            search results with embeddings and metadata for all found IDs
            Or Dict[Any, List[Dict[str, Any]]] when map_to_lexical_units
            is set to True (the key is id of lexical unit).

        Note:
            Searches by exact ID match in the lexical unit collection.
            Returns combined results for all requested IDs.
        """
        if not lu_ids:
            self.logger.warning("Empty lu_ids list provided")
            return []

        try:
            return self._select_lu_ids_on_collection(
                collection=self._get_lu_collection(),
                lu_ids=lu_ids,
                map_to_lexical_units=map_to_lexical_units,
                output_fields=MilvusSearchFields.LU_EMBEDDING_OUT_FIELDS,
            )
        except MilvusException as e:
            self.logger.error(
                f"Failed to retrieve lexical unit embeddings for IDs {lu_ids}: {e}"
            )
        except Exception as e:
            self.logger.error(f"Unexpected error during lexical unit retrieval: {e}")
        return []

    def get_lexical_units_examples_embedding(
        self, lu_ids: List[int], map_to_lexical_units: bool = False
    ) -> List[Dict[str, Any]] | Dict[Any, List[Dict[str, Any]]]:
        """
        Retrieve lexical unit examples embeddings by list of lu_id
        from the lexical units examples collection.

        Args:
            lu_ids: List of lexical unit IDs (lu_ids) to search
            for in the lexical units examples collection
            map_to_lexical_units: Whether to map to lexical units mapping
            where as the result is the dictionary with lexical unit identifier
            used as a key, and as value is the list of relevant results
            from the Milvus query belonging to the lexical

        Returns:
            List[List[Dict[str, Any]]]: List of dictionaries containing
            search results with embeddings and metadata for all found lu_ids
            Or in case when map_to_lexical_units is set to True, the
            dictionary of the list of dictionaries containing search results
            with mapping to lexical units will be returned.
        Note:
            Searches by exact lu_id match in the lexical unit examples collection.
            Returns combined results for all requested lu_ids.
        """
        if not lu_ids:
            self.logger.warning("Empty lu_ids list provided")
            return []

        try:
            return self._select_lu_ids_on_collection(
                collection=self._get_lu_examples_collection(),
                lu_ids=lu_ids,
                map_to_lexical_units=map_to_lexical_units,
                output_fields=MilvusSearchFields.LU_EXAMPLES_OUT_FIELDS,
            )
        except MilvusException as e:
            self.logger.error(
                f"Failed to retrieve lexical unit examples for lu_ids {lu_ids}: {e}"
            )
            return []
        except Exception as e:
            self.logger.error(
                f"Unexpected error during lexical unit examples retrieval: {e}"
            )
            return []

    def _select_lu_ids_on_collection(
        self,
        collection,
        lu_ids: List[int],
        map_to_lexical_units: bool,
        output_fields,
    ):
        """
        Query collection for lexical unit IDs and optionally map results.

        Constructs and executes a Milvus query to retrieve records matching the
        provided lexical unit IDs. Can optionally organize results by lexical
        unit ID for easier processing.

        Args:
            collection: Milvus collection to query
            lu_ids: List of lexical unit IDs to search for
            map_to_lexical_units: Whether to group results by lexical unit ID
            output_fields: Fields to include in query results

        Returns:
            List or Dict: Query results, optionally mapped by lexical unit ID
        """
        limit = len(lu_ids) * self.LU_LIMIT_FACTOR
        ids_str = ", ".join(str(lu_id) for lu_id in lu_ids)
        expr = f"lu_id in [{ids_str}]"
        results = collection.query(
            expr=expr, output_fields=output_fields, limit=limit
        )
        if not results:
            return {} if map_to_lexical_units else []

        if map_to_lexical_units:
            results = self._map_to_lexical_unit_dict(results=results)

        return results

    @staticmethod
    def _map_to_lexical_unit_dict(results):
        """
        Group query results by lexical unit ID.

        Organizes a list of query results into a dictionary where each key
        is a lexical unit ID and the value is a list of all records associated
        with that lexical unit.

        Args:
            results: List of query result dictionaries containing lu_id field

        Returns:
            Dict[int, List]: Dictionary mapping lexical unit IDs to their records
        """
        _m = {}
        for r in results:
            if r["lu_id"] not in _m:
                _m[r["lu_id"]] = []
            _m[r["lu_id"]].append(r)
        return _m

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
