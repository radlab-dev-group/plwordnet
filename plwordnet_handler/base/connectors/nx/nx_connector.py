import pickle
import networkx as nx

from pathlib import Path
from typing import Optional, List, Any, Dict

from plwordnet_handler.base.structure.elems.synset import Synset, SynsetMapper
from plwordnet_handler.base.connectors.connector_data import GraphMapperData
from plwordnet_handler.base.structure.elems.lu import LexicalUnit, LexicalUnitMapper
from plwordnet_handler.base.connectors.connector_i import PlWordnetConnectorInterface
from plwordnet_handler.base.structure.elems.rel_type import (
    RelationType,
    RelationTypeMapper,
)
from plwordnet_handler.base.structure.elems.synset_relation import (
    SynsetRelation,
    SynsetRelationMapper,
)
from plwordnet_handler.base.structure.elems.lu_relations import (
    LexicalUnitRelation,
    LexicalUnitRelationMapper,
)
from plwordnet_handler.base.structure.elems.lu_in_synset import (
    LexicalUnitAndSynset,
    LexicalUnitAndSynsetMapper,
)
from plwordnet_handler.utils.logger import prepare_logger


class PlWordnetAPINxConnector(PlWordnetConnectorInterface):
    """
    NetworkX graph connector implementation for plWordnet API.
    Loads data from pre-saved NetworkX MultiDiGraph files
    instead of connecting to a MySQL database.
    """

    def __init__(
        self,
        nx_graph_dir: str,
        autoconnect: bool = False,
        log_level: Optional[str] = "INFO",
    ) -> None:
        """
        Initialize plWordnet NetworkX connector.

        Args:
            nx_graph_dir: Path to a directory containing NetworkX graph files
            autoconnect: If true, automatically connect to MySQL database
            log_level: Logging level (INFO as default)
            then a new one will be created.

        Raises:
            Any exception raised if autoconnect failed
        """
        self.nx_graph_dir = Path(nx_graph_dir)
        self.graphs = {}
        self._relation_types: Optional[List[RelationType]] = None
        self._lexical_units_in_synsets: Optional[List[LexicalUnitAndSynset]] = None
        self.logger = prepare_logger(logger_name=__name__, log_level=log_level)

        self._connected = False
        if autoconnect:
            try:
                self.connect()
            except Exception as e:
                raise e

    def connect(self) -> bool:
        """
        Load NetworkX graphs from the directory.

        Returns:
            bool: True if graphs loaded successfully, False otherwise
        """
        try:
            for g_type, g_file in GraphMapperData.GRAPH_TYPES.items():
                self.graphs[g_type] = self._load_graph(g_file)
            self.logger.info(
                f"Successfully loaded NetworkX graphs from {self.nx_graph_dir}"
            )

            self._load_relation_types()
            if self._relation_types is None:
                self.logger.error("No relation_types loaded!")
                return False
            self.logger.info("Successfully loaded relation_types!")

            self._load_lexical_units_and_synsets()
            if self._lexical_units_in_synsets is None:
                self.logger.error("No lexical_units_in_synsets loaded!")
                return False
            self.logger.info("Successfully loaded lexical_units_in_synsets!")

            self._connected = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to load NetworkX graphs: {e}")
            return False

    def disconnect(self) -> None:
        """
        Clear loaded graphs from memory.
        """
        self.graphs.clear()
        self._connected = False
        self.logger.info("Disconnected from NetworkX graphs")

    def is_connected(self) -> bool:
        """
        Check if graphs are loaded.

        Returns:
            bool: True if connected, False otherwise
        """
        return self._connected

    def get_lexical_unit(self, lu_id: int) -> Optional[LexicalUnit]:
        """
        Retrieves a lexical unit from the NetworkX graph by its unique identifier.

        This method fetches lexical unit data from the in-memory NetworkX
        graph structure and maps it to a LexicalUnit object. It first verifies
        the graph connection is active before attempting to access the node data.

        Args:
            lu_id (int): The unique identifier of the lexical unit to retrieve

        Returns:
            Optional[LexicalUnit]: The lexical unit object if found and successfully
            mapped, None if not connected, node doesn't exist, or mapping fails

        Note:
            This method accesses data from NetworkX graphs loaded in memory,
            providing faster access compared to database queries but requiring
            the graphs to be preloaded and connected.
        """

        if not self.is_connected():
            self.logger.error("Not connected to NetworkX graphs")
            return None

        try:
            node_data = (
                self.graphs[GraphMapperData.G_LU].nodes[lu_id].get("data", {})
            )
            if node_data is not None and len(node_data):
                return LexicalUnitMapper.map_from_dict(data=node_data)
            else:
                self.logger.error(f"No data found for lexical unit {lu_id}")
        except Exception as e:
            self.logger.error(f"Failed to get lexical unit {lu_id}: {e}")
            return None
        return None

    def get_lexical_relations(
        self, limit: Optional[int] = None
    ) -> Optional[List[LexicalUnitRelation]]:
        """
        Get lexical relations from the lexical units graph edges.

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of lexical relations or None if an error occurs
        """
        return self._run_relation_mapper(
            limit=limit,
            g_type=GraphMapperData.G_LU,
            mapper=LexicalUnitRelationMapper,
        )

    def get_synset(self, syn_id: int) -> Optional[Synset]:
        """
        Retrieves a synset from the NetworkX graph by its unique identifier.

        This method fetches synset data from the in-memory NetworkX graph structure
        and maps it to a Synset object. It first verifies the graph connection is
        active before attempting to access the node data.

        Args:
            syn_id (int): The unique identifier of the synset to retrieve

        Returns:
            Optional[Synset]: The synset object if found and successfully mapped,
            None if not connected, node doesn't exist, or mapping fails

        Note:
            This method accesses data from NetworkX graphs loaded in memory,
            providing faster access compared to database queries but requiring
            the graphs to be preloaded and connected.
        """

        if not self.is_connected():
            self.logger.error("Not connected to NetworkX graphs")
            return None

        try:
            node_data = (
                self.graphs[GraphMapperData.G_SYN].nodes[syn_id].get("data", {})
            )
            if node_data and len(node_data):
                return SynsetMapper.map_from_dict(data=node_data)
            else:
                self.logger.error(f"No data found for synset {syn_id}")
        except Exception as e:
            self.logger.error(f"Failed to get synset {syn_id}: {e}")
            return None
        return None

    def get_synset_relations(
        self, limit: Optional[int] = None
    ) -> Optional[List[SynsetRelation]]:
        """
        Get synset relations from the synsets graph edges.

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of synset relations or None if error
        """
        return self._run_relation_mapper(
            limit=limit,
            g_type=GraphMapperData.G_SYN,
            mapper=SynsetRelationMapper,
        )

    def get_units_and_synsets(
        self, limit: Optional[int] = None
    ) -> Optional[List[LexicalUnitAndSynset]]:
        """
        Retrieves lexical units and synsets data with an optional limit.

        This method provides access to the loaded lexical units and synsets data,
        optionally applying a limit to restrict the number of returned items.

        Args:
            limit (Optional[int]): Maximum number of items to return.
            If None, returns all available data

        Returns:
            Optional[List[LexicalUnitAndSynset]]: Limited list of lexical unit
            and synset objects, or None if no data is available
        """
        return self._apply_limit(self._lexical_units_in_synsets, limit)

    def get_relation_types(
        self, limit: Optional[int] = None
    ) -> Optional[List[RelationType]]:
        """
        Retrieves relation types data with optional limit.

        This method provides access to the loaded relation types data,
        optionally applying a limit to restrict the number of returned items.

        Args:
            limit (Optional[int]): Maximum number of items to return.
            If None, returns all available data

        Returns:
            Optional[List[RelationType]]: Limited list of relation type objects,
            or None if no data is available
        """
        return self._apply_limit(self._relation_types, limit)

    def get_lexical_units(
        self, limit: Optional[int] = None
    ) -> Optional[List[LexicalUnit]]:
        """
        Get lexical units from the lexical units graph.

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of lexical units or None if error
        """
        if not self.is_connected():
            self.logger.error("Not connected to NetworkX graphs")
            return None

        try:
            graph = self.graphs[GraphMapperData.G_LU]
            lu_data_list = []
            for node_id in graph.nodes():
                node_data = graph.nodes[node_id].get("data", {})
                if node_data:
                    lu_data_list.append(node_data)
            lu_data_list = self._apply_limit(lu_data_list, limit)
            return LexicalUnitMapper.map_from_dict_list(lu_data_list)

        except Exception as e:
            self.logger.error(f"Error getting lexical units: {e}")
            return None

    def get_synsets(self, limit: Optional[int] = None) -> Optional[List[Synset]]:
        """
        Get synsets from the synsets graph.

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of Synsets or None if error
        """
        return self._map_graph_to_objects(
            g_type=GraphMapperData.G_SYN,
            mapper=SynsetMapper,
            limit=limit,
        )

    def _map_graph_to_objects(
        self, g_type: str, mapper, limit: Optional[int] = None
    ):
        """
        Maps NetworkX graph nodes to objects using the specified mapper
        with an optional limit.

        This protected method iterates through all nodes in the specified
        graph type, extracts their data dictionaries, and uses the provided
        mapper to convert them into domain objects. It applies an optional
        limit to restrict the number of processed nodes and handles connection
        validation and error cases.

        Args:
            g_type (str): The graph type identifier to process
            mapper: The mapper object used to convert node data to domain objects
            limit (Optional[int], optional): Maximum number of nodes to process.
            Defaults to None (no limit).

        Note:
            This method processes all nodes in the graph sequentially,
            extracting their data and applying the mapper transformation
            before optionally limiting the result set.
        """

        if not self.is_connected():
            self.logger.error("Not connected to NetworkX graphs")
            return None

        try:
            data_list = []
            graph = self.graphs[g_type]
            for node_id in graph.nodes():
                node_data = graph.nodes[node_id].get("data", {})
                if node_data:
                    data_list.append(node_data)
            data_list = self._apply_limit(data_list, limit)
            return mapper.map_from_dict_list(data_list)
        except Exception as e:
            self.logger.error(f"Error getting {g_type}: {e}")
            return None

    def _rel_types_from_g_type(
        self, g_type: str, relation_types_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Extracts relation type metadata from the edges of a specific graph type.

        This protected method iterates through all edges in the specified graph and
        collects relation type information from edge data. For each unique
        relation ID found, it creates a standardized relation type entry with
        metadata including object type, name, and display properties. The object
        type is determined based on whether the graph represents synsets
        or lexical units.

        Args:
            g_type (str): The graph type identifier to process edges from
            relation_types_data (dict[str, Any]): Existing relation types
            dictionary to update with new entries

        Returns:
            dict[str, Any]: Updated relation types dictionary containing
            both existing and newly discovered relation type metadata

        Note:
            The method assigns object type 2 for synset graphs and 1 for
            other types and uses the relation_type from edge data
            as the name, falling back to a generated name if not available.
        """

        _g = self.graphs[g_type]
        for _, _, edge_data in _g.edges(data=True):
            rel_id = edge_data.get("relation_id")
            if rel_id is not None and rel_id not in relation_types_data:
                relation_types_data[rel_id] = {
                    "ID": rel_id,
                    "objecttype": 2 if g_type == GraphMapperData.G_SYN else 1,
                    "PARENT_ID": None,
                    "REVERSE_ID": None,
                    "name": edge_data.get("relation_type", f"relation_{rel_id}"),
                    "description": "",
                    "posstr": "",
                    "autoreverse": 0,
                    "display": edge_data.get("relation_type", f"relation_{rel_id}"),
                    "shortcut": "",
                    "pwn": "",
                    "order": rel_id,
                }
        return relation_types_data

    def _load_graph(self, filename: str) -> Optional[nx.MultiDiGraph]:
        """
        Load a NetworkX graph from a file.

        Args:
            filename: Name of the graph file

        Returns:
            nx.MultiDiGraph: Loaded graph if the file exists None otherwise
        """
        graph_path = self.nx_graph_dir / filename
        self.logger.debug(f"Loading graph from {graph_path}")

        if not graph_path.exists():
            self.logger.error(f"Graph file {graph_path} does not exist")
            return None

        with open(graph_path, "rb") as f:
            graph = pickle.load(f)

        self.logger.debug(
            f"Loaded graph with {graph.number_of_nodes()} "
            f"nodes and {graph.number_of_edges()} edges"
        )
        return graph

    def _load_relation_types(self) -> Optional[List[RelationType]]:
        """
        Loads relation types data and stores it in the instance variable.

        This method uses the generic _load_pickled_data method to load
        relation types from the designated file. The loaded data is assigned
        to the _relation_types instance variable for use throughout the class.

        Returns:
            Optional[List[RelationType]]: List of relation type objects if loading
            is successful, None if loading fails

        Side effects:
            - Sets self._relation_types to the loaded data
            - Logs error message if loading fails
        """
        self._relation_types = self._load_pickled_data(
            filename=GraphMapperData.RELATION_TYPES_FILE,
            load_type=GraphMapperData.RELATION_TYPES,
        )

        if self._relation_types is None:
            self.logger.error("Error while loading relation types")
            return None
        return self._relation_types

    def _load_lexical_units_and_synsets(
        self,
    ) -> Optional[List[LexicalUnitAndSynset]]:
        """
        Loads lexical units and synsets data and stores it in the instance variable.

        This method uses the generic _load_pickled_data method to load
        lexical units and synsets mapping from the designated file.
        The loaded data is assigned to the _lexical_units_in_synsets
        instance variable for use throughout the class.

        Returns:
            Optional[List[LexicalUnitAndSynset]]: List of lexical unit and synset
            objects if loading is successful, None if loading fails

        Side effects:
            - Sets self._lexical_units_in_synsets to the loaded data
            - Logs an error message if loading fails
        """
        self._lexical_units_in_synsets = self._load_pickled_data(
            filename=GraphMapperData.LU_IN_SYNSET_FILE,
            load_type=GraphMapperData.UNIT_AND_SYNSET,
        )

        if self._lexical_units_in_synsets is None:
            self.logger.error("Error while loading lexical units and synsets")
            return None
        return self._lexical_units_in_synsets

    def _load_pickled_data(self, filename: str, load_type: str) -> Optional[Any]:
        """
        Generic method for loading pickled data from a file.

        This utility method provides a standardized way to load any type
        of pickled data from the graph directory. It handles file existence
        validation, error logging, and the actual unpickling process.

        Args:
            filename (str): Name of the file to load from the graph directory
            load_type (str): Descriptive name of the data type being
            loaded (used for logging)

        Returns:
            Optional[Any]: The unpickled data if loading is successful,
            None if the file doesn't exist or loading fails

        Side effects:
            - Logs debug messages about the loading process start and completion
            - Logs error message if the specified file doesn't exist

        File dependencies:
            - File must exist in the nx_graph_dir directory
            - File must be in valid pickle format
        """
        data_file = self.nx_graph_dir / filename
        self.logger.debug(f"Loading {load_type} from {data_file}")

        if not data_file.exists():
            self.logger.error(f"{load_type} file {data_file} does not exist")
            return None

        with open(data_file, "rb") as f:
            data_file_content = pickle.load(f)
        if data_file_content is not None:
            d_size = len(data_file_content)
            self.logger.debug(f"Loaded {d_size} {load_type} from {data_file}")
        else:
            self.logger.error(f"{load_type} file {data_file} does not exist?")
        return data_file_content

    def _run_relation_mapper(
        self, limit: Optional[int], g_type: str, mapper
    ) -> Optional[List[LexicalUnit | Synset | LexicalUnitAndSynset]]:
        """
        Executes a relation mapper on a specified graph type with an optional limit.

        This private method runs a relation mapping operation on the specified
        NetworkX graph using the provided mapper object. It first verifies
        the graph connection is active, then delegates to the internal relation
        mapper with the appropriate graph, mapper, and limit parameters.

        Args:
            limit (Optional[int]): Maximum number of items to process,
            None for no limit
            g_type (str): The graph type identifier to operate on
            mapper: The mapper object used to transform graph data

        Returns:
            Optional[List[LexicalUnit | Synset | LexicalUnitAndSynset]]: List of
            mapped objects if successful, None if not connected or an error occurs

        Note:
            This method serves as a wrapper that adds connection validation and
            error handling around the core relation mapping functionality.
        """
        if not self.is_connected():
            self.logger.error("Not connected to NetworkX graphs")
            return None
        try:
            return self._relation_mapper(
                graph=self.graphs[g_type],
                mapper_obj=mapper,
                limit=limit,
            )
        except Exception as e:
            self.logger.error(f"Error getting {g_type}: {e}")
            return None

    def _relation_mapper(
        self, graph: nx.MultiDiGraph, mapper_obj, limit: Optional[int] = None
    ):
        """
        Extract and map relation data from graph edges.

        This method iterates through all edges in the provided NetworkX MultiDiGraph,
        extracts relation data from edge attributes, and maps the data using the
        provided mapper object.

        Args:
            graph: The NetworkX MultiDiGraph to extract relations from.
            mapper_obj: The mapper object used to transform the relation data from
                        dictionary format to the desired output format.
            limit: Maximum number of relations to process.
                   If None, all relations are processed.

        Returns:
            The mapped relation data in the format determined by the mapper_obj.
            The exact return type depends on the mapper implementation.
        """
        relations_data = []
        for parent_id, child_id, edge_data in graph.edges(data=True):
            relation_data = edge_data.get("data", {})
            if relation_data:
                relations_data.append(relation_data)
        relations_data = self._apply_limit(relations_data, limit)
        return mapper_obj.map_from_dict_list(relations_data)

    @staticmethod
    def _apply_limit(data_list: List, limit: Optional[int]) -> List:
        """
        Apply a limit to a list of data if specified.

        Args:
            data_list: List of data to limit
            limit: Optional limit for number of results

        Returns:
            Limited list or original list if no limit specified
        """
        if limit is not None and limit > 0:
            return data_list[:limit]
        return data_list
