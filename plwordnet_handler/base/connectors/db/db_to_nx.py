import os
import tqdm
import pickle
import networkx as nx

from typing import Optional, List, Any

from plwordnet_handler.utils.logger import prepare_logger
from plwordnet_handler.base.structure.elems.synset import Synset
from plwordnet_handler.base.structure.elems.lu import LexicalUnit
from plwordnet_handler.base.structure.elems.comment import ParsedComment
from plwordnet_handler.base.structure.elems.synset_relation import SynsetRelation
from plwordnet_handler.base.structure.elems.lu_relations import LexicalUnitRelation
from plwordnet_handler.base.structure.polishwordnet import PolishWordnet
from plwordnet_handler.base.connectors.connector_data import GraphMapperData
from plwordnet_handler.base.connectors.db.db_connector import (
    PlWordnetAPIMySQLDbConnector,
)


class GraphMapper(GraphMapperData):
    """
    Class for mapping Polish Wordnet database elements to NetworkX MultiDiGraph.
    """

    def __init__(self, polish_wordnet: PolishWordnet, limit: Optional[int] = None):
        """
        Initialize the mapper with a PolishWordnet instance.

        Args:
            polish_wordnet: PolishWordnet instance to use for data retrieval
            limit: Optional limit for the number of results
        """
        self.limit = limit
        self.polish_wordnet = polish_wordnet
        self.logger = prepare_logger(
            logger_name=__name__,
            use_default_config=False,
        )

        self._graphs = {}
        self._rel_types = {}
        self._units_and_synsets = {}

    def convert_to_synset_graph(self) -> nx.MultiDiGraph:
        """
        Create a graph where synsets are nodes and synset relations are edges.

        Returns:
            NetworkX MultiDiGraph with synsets as nodes and relations as edges

        Raises:
            ValueError: If synsets or synset relations data is None
        """
        self.logger.info("Creating synset graph")

        synsets = self.polish_wordnet.get_synsets(limit=self.limit)
        if synsets is None:
            raise ValueError("Synsets data is None")

        synset_relations = self.polish_wordnet.get_synset_relations(limit=self.limit)
        if synset_relations is None:
            raise ValueError("Synset relations data is None")

        s_g = self._prepare_graph(
            to_nodes=synsets, to_edges=synset_relations, graph_type=self.G_SYN
        )

        if self.limit is None or self.limit < 1:
            self.__check_list_graph_cohesion_no_limit(
                list_elems=synsets,
                list_relations=synset_relations,
                graph=s_g,
                g_type=self.G_SYN,
            )

        return s_g

    def convert_to_lexical_unit_graph(self) -> nx.MultiDiGraph:
        """
        Create a graph where lexical units are nodes
        and lexical unit relations are edges.

        Returns:
            NetworkX MultiDiGraph with lexical units as nodes and relations as edges

        Raises:
            ValueError: If lexical units or lexical unit relations data is None
        """
        self.logger.info("Creating lexical unit graph")

        lexical_units = self.polish_wordnet.get_lexical_units(limit=self.limit)
        if lexical_units is None:
            raise ValueError("Lexical units data is None")

        lu_relations = self.polish_wordnet.get_lexical_relations(limit=self.limit)
        if lu_relations is None:
            raise ValueError("Lexical unit relations data is None")

        l_g = self._prepare_graph(
            to_nodes=lexical_units, to_edges=lu_relations, graph_type=self.G_LU
        )

        if self.limit is None or self.limit < 1:
            self.__check_list_graph_cohesion_no_limit(
                list_elems=lexical_units,
                list_relations=lu_relations,
                graph=l_g,
                g_type=self.G_LU,
            )

        return l_g

    def prepare_all_graphs(self):
        """
        Prepares all available NetworkX graph types from Polish Wordnet data.

        Checks which graph types haven't been created yet and generates them
        based on database data. Supports three graph types:
        - Synset graph (G_SYN): nodes are synsets, edges are synset relations
        - Lexical unit graph (G_LU): nodes are lexical units,
          edges are lexical relations
        - Synsets with lexical units graph (G_UAS): nodes are synsets containing
        lists of lexical units

        Raises:
            ValueError: When an unknown graph type is encountered.
        """
        for g_type in self.GRAPH_TYPES:
            if g_type not in self._graphs:
                if g_type == self.G_SYN:
                    self.convert_to_synset_graph()
                elif g_type == self.G_LU:
                    self.convert_to_lexical_unit_graph()
                else:
                    raise ValueError(f"Unknown graph type: {g_type}")

        self.prepare_mappings_before_store_to_file()

    def prepare_mappings_before_store_to_file(self):
        """
        Prepares data mappings before storing them to a file.

        This method retrieves and prepares two essential types
        of mappings from the Polish wordnet:
        1. Relation types mappings (_rel_types)
        2. Units and synsets mappings (_units_and_synsets)

        For each mapping type:
        - Fetches data with the limit specified by self.limit
        - Validates that data was successfully exported
        - Logs information about the number of retrieved mappings
        - Raises ValueError if no data is available

        Raises:
            ValueError: When relation types or units/synsets data export fails

        Side effects:
            - Sets self._rel_types to the retrieved relation types
            - Sets self._units_and_synsets to the retrieved units and synsets
            - Logs the data retrieval process at debug level
        """
        self.logger.debug("Preparing mappings to store to the file")

        self.logger.debug(" - relation types mapping")
        self._rel_types = self.polish_wordnet.get_relation_types(limit=self.limit)
        if self._rel_types is None:
            self.logger.error("No relation types data exported!")
            raise ValueError("No relation types data exported!")
        else:
            self.logger.debug(f"  -> number of mappings {len(self._rel_types)}")

        self.logger.debug(" - units and synsets mapping")
        self._units_and_synsets = self.polish_wordnet.get_units_and_synsets(
            limit=self.limit
        )
        if self._units_and_synsets is None:
            self.logger.error("No units and synsets data exported!")
            raise ValueError("No units and synsets data exported!")
        else:
            self.logger.debug(
                f"  -> number of units and synsets {len(self._units_and_synsets)}"
            )
        self.logger.debug("The mapping has been prepared correctly")

    def store_to_dir(self, out_dir_path: str):
        """
        Store all graphs from _graphs to directory as separate pickle files.

        Creates a subdirectory GRAPH_DIR within out_dir_path and saves
        each graph from _graphs as a separate pickle file.

        Args:
            out_dir_path: Path to the output directory
        """
        graphs_dir = (
            os.path.join(out_dir_path, self.GRAPH_DIR)
            if self.GRAPH_DIR not in out_dir_path
            else out_dir_path
        )
        os.makedirs(graphs_dir, exist_ok=True)

        self.logger.info(f"Storing graphs to directory: {graphs_dir}")
        for graph_type, graph in self._graphs.items():
            g_f_name = self.GRAPH_TYPES[graph_type]
            file_path = os.path.join(graphs_dir, g_f_name)
            with open(file_path, "wb") as f:
                pickle.dump(graph, f)
            self.logger.info(f"Saved graph '{graph_type}' to {file_path}")
        self.logger.info(f"Successfully stored {len(self._graphs)} graphs")

        rel_file_path = os.path.join(graphs_dir, self.RELATION_TYPES_FILE)
        self.logger.info(f"Storing relation types to file: {rel_file_path}")
        with open(rel_file_path, "wb") as f:
            pickle.dump(self._rel_types, f)

        uas_file_path = os.path.join(graphs_dir, self.LU_IN_SYNSET_FILE)
        self.logger.info(f"Storing units and synsets to file: {uas_file_path}")
        with open(uas_file_path, "wb") as f:
            pickle.dump(self._units_and_synsets, f)

    def _prepare_graph(
        self,
        to_nodes: List[Any],
        to_edges: List[Any],
        graph_type: str,
    ):
        """
        Create and populate a NetworkX MultiDiGraph from node and edge data.

        This method constructs a directed multigraph by adding nodes and edges from
        the provided data collections. It enriches edge data with relation type
        information and stores the completed graph in the internal graphs cache.

        Args:
            to_nodes: Collection of node objects that will be added to the graph.
                     Each node object must have an ID attribute and to_dict() method.
            to_edges: Collection of edge/relation objects that define connections
                     between nodes. Each edge must have PARENT_ID, CHILD_ID, REL_ID
                     attributes, and the `to_dict()` method.
            graph_type: Identifier for the type of graph being created.
                        Used for logging and internal storage.

        Returns:
            nx.MultiDiGraph: The constructed graph with all nodes and edges added.

        Raises:
            ValueError: If relation types data cannot be retrieved from the wordnet.

        Side Effects:
            - Stores the created graph in self._graphs[graph_type]
            - Logs information about the graph creation process
        """

        relation_types = self.polish_wordnet.get_relation_types(limit=self.limit)
        if relation_types is None:
            raise ValueError("Relation types data is None")

        # ID to Relation mapper
        rel_type_map = {rt.ID: rt for rt in relation_types}

        # Add nodes
        graph = nx.MultiDiGraph()
        for obj_n in to_nodes:
            graph.add_node(obj_n.ID, data=obj_n.to_dict())

        # Add edges
        for relation in to_edges:
            rel_type = rel_type_map.get(relation.REL_ID)
            edge_data = {
                "relation_id": relation.REL_ID,
                "relation_type": rel_type.name if rel_type else None,
                "data": relation.to_dict(),
            }
            graph.add_edge(relation.PARENT_ID, relation.CHILD_ID, **edge_data)

        self.logger.info(
            f"Created {graph_type} graph with {graph.number_of_nodes()} "
            f"nodes and {graph.number_of_edges()} edges"
        )

        self._graphs[graph_type] = graph
        return graph

    def __check_list_graph_cohesion_no_limit(
        self,
        g_type: str,
        graph: nx.MultiDiGraph,
        list_elems: List[Synset | LexicalUnit],
        list_relations: List[LexicalUnitRelation | SynsetRelation],
    ):
        """
        Performs comprehensive cohesion validation between graph data and list data.

        This private method validates the consistency between NetworkX graph
        representation and corresponding list data structures. It compares node
        counts, edge counts, and verifies that all list elements exist in the graph
        with proper data. The method generates detailed logging output, including
        error reports for any inconsistencies found during validation.

        Args:
            g_type (str): The graph type identifier being validated
            graph (nx.MultiDiGraph): The NetworkX graph to validate against
            list_elems (List[Synset | LexicalUnit]): List of domain objects to verify
            list_relations (List[LexicalUnitRelation | SynsetRelation]): List of
            relations to verify

        Raises:
            TypeError if the `node.comment` type is not supported (only dict
            and ParsedComment are valid)

        Note:
            This method is used for data integrity verification during graph
            construction or maintenance. It logs detailed reports including
            counts, mismatches, and missing data, helping identify potential
            issues in the graph preparing process.
        """

        list_count = len(list_elems)
        nodes_count = graph.number_of_nodes()
        self.logger.info("===============================================")
        self.logger.info(f"Report of cohesion of {g_type} graph")
        self.logger.info(f" - number of {g_type} (graph): {nodes_count}")
        self.logger.info(f" - number of {g_type} (list): {list_count}")
        if nodes_count != list_count:
            self.logger.error(
                f" [ERROR] Number of {g_type} {nodes_count} (graph) does "
                f"not match number of {g_type} (list): {list_count}!"
            )

        s_r_count = len(list_relations)
        n_r_count = graph.number_of_edges()
        self.logger.info(f" - number of {g_type} relations (graph): {n_r_count}")
        self.logger.info(f" - number of {g_type} relations (list): {s_r_count}")
        if s_r_count != n_r_count:
            self.logger.error(
                f"[ERROR] Number of the {g_type} relations {n_r_count} "
                f"(graph) does not match the number of the {g_type} "
                f"relations (list): {s_r_count}"
            )
            raise RuntimeError(
                f"[ERROR] Number of {g_type} relations "
                f"are not valid {s_r_count} !={n_r_count} "
            )

        self.logger.info(f"~> Verification of {g_type} comments")
        with tqdm.tqdm(total=len(list_elems), desc="Verifying comments") as pbar:
            for s in list_elems:
                try:
                    s_node = graph.nodes[s.ID]
                    node_data = s_node.get("data", {})
                    if not len(node_data):
                        self.logger.error(
                            f" [ERROR] No data found for graph synset {s.ID}"
                        )
                        continue

                    node_data = node_data.get("comment", {})
                    if node_data is None:
                        node_orig_comment = None
                    elif type(node_data) is ParsedComment:
                        node_orig_comment = node_data.original_comment
                    else:
                        raise TypeError(
                            f"Invalid type for node_data comment: {type(node_data)}"
                        )

                    elem_orig_comment = s.comment
                    if elem_orig_comment is not None:
                        elem_orig_comment = elem_orig_comment.original_comment

                    if node_orig_comment != elem_orig_comment:
                        self.logger.error(
                            f" [ERROR] Original comment of node {s.ID} "
                            f"is different than elem.comment"
                        )
                        self.logger.error(
                            "---------------------------------------------------"
                        )
                        self.logger.error(node_orig_comment)
                        self.logger.error(elem_orig_comment)
                        self.logger.error(
                            "---------------------------------------------------"
                        )
                except KeyError:
                    self.logger.error(
                        f"[ERROR] Synset {s.ID} does not exist in graph"
                    )
                pbar.update(1)
        self.logger.info("===============================================")


def dump_to_networkx_file(
    db_config: str,
    out_dir_path: str,
    limit: Optional[int] = None,
    show_progress_bar: Optional[bool] = True,
    extract_wikipedia_articles: Optional[bool] = False,
    log_level: Optional[str] = "INFO",
) -> bool:
    """
    Exports Polish Wordnet data from a MySQL database to NetworkX graph files.

    This function establishes a database connection, extracts wordnet data,
    converts it to NetworkX MultiDiGraph format, and saves the resulting graph
    files to the specified output directory.

    Args:
        db_config (str): Path to the database configuration file
        out_dir_path (str): Directory path where NetworkX graph files will be stored
        limit (Optional[int]): Maximum number of records to process (None for no limit)
        show_progress_bar (Optional[bool]): Whether to display progress
        indication during processing
        extract_wikipedia_articles (Optional[bool]): Whether to include
        Wikipedia article data
        log_level: Logger level (INFO as default)

    Returns:
        Boolean: True for successful completion, False for error

    Raises:
        Exception: Any database connection, data processing, or file I/O
        errors are caught and logged, with the function returning error code 1
    """

    logger = prepare_logger(logger_name=__name__, log_level=log_level)
    logger.info("Starting NetworkX graph generation from database")

    try:
        # Use database connector for conversion
        connector = PlWordnetAPIMySQLDbConnector(
            db_config_path=db_config, log_level=log_level
        )

        with PolishWordnet(
            connector=connector,
            extract_wiki_articles=extract_wikipedia_articles,
            use_memory_cache=True,
            show_progress_bar=show_progress_bar,
        ) as pl_wn:
            logger.info("Converting to NetworkX MultiDiGraph...")
            g_mapper = GraphMapper(polish_wordnet=pl_wn, limit=limit)
            g_mapper.prepare_all_graphs()
            g_mapper.store_to_dir(out_dir_path=out_dir_path)
        logger.info("NetworkX graph generation completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error during NetworkX graph generation: {e}")
    return False
