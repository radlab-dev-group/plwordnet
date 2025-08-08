import pickle
import networkx as nx

from pathlib import Path
from typing import Optional, List

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

            self._connected = True
            self.logger.info(
                f"Successfully loaded NetworkX graphs from {self.nx_graph_dir}"
            )
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

    def get_lexical_relations(
        self, limit: Optional[int] = None
    ) -> Optional[List[LexicalUnitRelation]]:
        """
        Get lexical relations from the lexical units graph edges.

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of lexical relations or None if error
        """
        if not self.is_connected():
            self.logger.error("Not connected to NetworkX graphs")
            return None

        try:
            return self._relation_mapper(
                graph=self.graphs[GraphMapperData.G_LU],
                mapper_obj=LexicalUnitRelationMapper,
                limit=limit,
            )
        except Exception as e:
            self.logger.error(f"Error getting lexical relations: {e}")
            return None

    def get_synsets(self, limit: Optional[int] = None) -> Optional[List[Synset]]:
        """
        Get synsets from the synsets graph.

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of Synsets or None if error
        """
        if not self.is_connected():
            self.logger.error("Not connected to NetworkX graphs")
            return None

        try:
            graph = self.graphs[GraphMapperData.G_SYN]
            synset_data_list = []
            for node_id in graph.nodes():
                node_data = graph.nodes[node_id].get("data", {})
                if node_data:
                    synset_data_list.append(node_data)
            synset_data_list = self._apply_limit(synset_data_list, limit)
            return SynsetMapper.map_from_dict_list(synset_data_list)
        except Exception as e:
            self.logger.error(f"Error getting synsets: {e}")
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
        if not self.is_connected():
            self.logger.error("Not connected to NetworkX graphs")
            return None

        try:
            return self._relation_mapper(
                graph=self.graphs[GraphMapperData.G_SYN],
                mapper_obj=SynsetRelationMapper,
                limit=limit,
            )
        except Exception as e:
            self.logger.error(f"Error getting synset relations: {e}")
            return None

    def get_units_and_synsets(
        self, limit: Optional[int] = None
    ) -> Optional[List[LexicalUnitAndSynset]]:
        """
        Get units and synsets data from the units_and_synsets graph.

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of LexicalUnitAndSynset or None if error
        """
        if not self.is_connected():
            self.logger.error("Not connected to NetworkX graphs")
            return None

        try:
            return self._relation_mapper(
                graph=self.graphs[GraphMapperData.G_UAS],
                mapper_obj=LexicalUnitAndSynsetMapper,
                limit=limit,
            )
        except Exception as e:
            self.logger.error(f"Error getting units and synsets: {e}")
            return None

    def get_relation_types(
        self, limit: Optional[int] = None
    ) -> Optional[List[RelationType]]:
        """
        Get relation types from both lexical units and synsets graphs.
        Extracts unique relation types from edge data.

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of relation types or None if error occurred
        """
        if not self.is_connected():
            self.logger.error("Not connected to NetworkX graphs")
            return None

        try:
            relation_types_data = {}
            lu_graph = self.graphs[GraphMapperData.G_LU]
            for _, _, edge_data in lu_graph.edges(data=True):
                rel_id = edge_data.get("relation_id")
                if rel_id is not None and rel_id not in relation_types_data:
                    # Create relation type data with all required fields
                    relation_types_data[rel_id] = {
                        "ID": rel_id,
                        "objecttype": 1,  # Default value for lexical unit relations
                        "PARENT_ID": None,
                        "REVERSE_ID": None,
                        "name": edge_data.get("relation_type", f"relation_{rel_id}"),
                        "description": "",
                        "posstr": "",
                        "autoreverse": 0,
                        "display": edge_data.get(
                            "relation_type", f"relation_{rel_id}"
                        ),
                        "shortcut": "",
                        "pwn": "",
                        "order": rel_id,  # Use relation ID as order
                    }

            synset_graph = self.graphs[GraphMapperData.G_SYN]
            for _, _, edge_data in synset_graph.edges(data=True):
                rel_id = edge_data.get("relation_id")
                if rel_id is not None and rel_id not in relation_types_data:
                    relation_types_data[rel_id] = {
                        "ID": rel_id,
                        "objecttype": 2,  # Default value for synset relations
                        "PARENT_ID": None,
                        "REVERSE_ID": None,
                        "name": edge_data.get("relation_type", f"relation_{rel_id}"),
                        "description": "",
                        "posstr": "",
                        "autoreverse": 0,
                        "display": edge_data.get(
                            "relation_type", f"relation_{rel_id}"
                        ),
                        "shortcut": "",
                        "pwn": "",
                        "order": rel_id,
                    }

            relation_types_list = list(relation_types_data.values())
            relation_types_list = self._apply_limit(relation_types_list, limit)

            return RelationTypeMapper.map_from_dict_list(relation_types_list)
        except Exception as e:
            self.logger.error(f"Error getting relation types: {e}")
            return None

    def _load_graph(self, filename: str) -> nx.MultiDiGraph:
        """
        Load a NetworkX graph from a file.

        Args:
            filename: Name of the graph file

        Returns:
            nx.MultiDiGraph: Loaded graph
        """
        graph_path = self.nx_graph_dir / filename
        self.logger.debug(f"Loading graph from {graph_path}")

        with open(graph_path, "rb") as f:
            graph = pickle.load(f)

        self.logger.debug(
            f"Loaded graph with {graph.number_of_nodes()} "
            f"nodes and {graph.number_of_edges()} edges"
        )
        return graph

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
