from typing import Optional

from plwordnet_handler.base.structure.polishwordnet import PolishWordnet
from plwordnet_handler.base.connectors.db.db_to_nx import dump_to_networkx_file
from plwordnet_handler.base.connectors.connector_i import PlWordnetConnectorInterface
from plwordnet_handler.base.connectors.nx.nx_connector import PlWordnetAPINxConnector
from plwordnet_handler.base.connectors.nx.nx_loader import connect_to_networkx_graphs
from plwordnet_handler.base.connectors.db.db_connector import (
    PlWordnetAPIMySQLDbConnector,
)
from plwordnet_handler.base.connectors.db.db_loader import connect_to_mysql_database


class CLIWrappers:
    def __init__(self, args, verify_args: bool, logger=None):
        """
        Initializes the CLI wrapper with command-line
        arguments and optional verification.

        Args:
            args: Parsed command-line arguments containing configuration options
            verify_args (bool): Whether to validate the provided arguments
            logger: Logger instance for recording operation status and errors
        """

        self.args = args
        self.logger = logger

        self.pl_wn = None
        self.last_connector = None

        if verify_args:
            self.are_args_correct(args=args)

    def are_args_correct(self, args=None) -> bool:
        """
        Validates the correctness of command-line arguments.

        Args:
            args: Optional arguments to validate (uses self.args if None)

        Returns:
            bool: True if arguments are valid, False otherwise

        Note:
            Currently returns True as validation
            logic is not yet implemented (TODO).
        """

        _args = self.args if args is None else args

        # TODO: Check options are correctly passed

        return True

    def dump_to_networkx_file(self) -> bool:
        """
        Private wrapper method that extracts database configuration and parameters
        from args and delegates to the main dump_to_networkx_file function
        to export graph data to NetworkX format.

        Returns:
            Boolean: True for successful completion, False for error
        """
        return dump_to_networkx_file(
            db_config=self.args.db_config,
            out_dir_path=self.args.nx_graph_dir,
            limit=self.args.limit,
            show_progress_bar=self.args.show_progress_bar,
            extract_wikipedia_articles=self.args.extract_wikipedia_articles,
            logger=self.logger,
        )

    def connect_to_networkx_graphs(self) -> Optional[PlWordnetAPINxConnector]:
        """
        Establishes a connection to NetworkX graph files using
        the configured directory path.

        This method creates and connects a NetworkX-based connector using
        the graph directory specified in the instance's arguments.
        It delegates to the load_from_networkx_graphs function with connection
        enabled and provides the instance logger for status tracking.

        Returns:
            Optional[PlWordnetAPINxConnector]: Connected NetworkX connector
            instance on success, None if a connection establishment fails
        """
        self.last_connector = connect_to_networkx_graphs(
            nx_graph_dir=self.args.nx_graph_dir,
            connect=True,
            logger=self.logger,
        )
        return self.last_connector

    def connect_to_database(self) -> Optional[PlWordnetAPIMySQLDbConnector]:
        """
        Establishes a connection to the MySQL database using configured parameters.

        This method creates a MySQL database connector using
        the database configuration path from arguments and
        attempts connection based on the use_database flag.

        Returns:
            Optional[PlWordnetAPIMySQLDbConnector]: Connected database connector
            instance on success, None if connection fails
        """
        self.last_connector = connect_to_mysql_database(
            db_config_path=str(self.args.db_config),
            connect=self.args.use_database,
            logger=self.logger,
        )
        return self.last_connector

    def prepare_wordnet_with_connector(
        self,
        connector: Optional[PlWordnetConnectorInterface],
        use_memory_cache: bool = True,
    ) -> Optional[PolishWordnet]:
        """
        Creates a PolishWordnet instance using the provided
        or last available connector.

        This method initializes the main PolishWordnet API with the
        specified connector, enabling access to wordnet data and operations.
        Falls back to the last used connector if none is explicitly provided.

        Args:
            connector (Optional[PlWordnetConnectorInterface]): Specific
            connector to use (uses self.last_connector if None)
            use_memory_cache (bool): Whether to enable memory caching for
            better performance

        Returns:
            Optional[PolishWordnet]: Configured PolishWordnet instance on success,
            None if no connector is available, or initialization fails
        """

        _c = connector if connector is not None else self.last_connector
        if _c is None:
            if self.logger:
                self.logger.error(
                    "Api cannot be prepared because no connectors found "
                    "in self.last_connector and no `connector` "
                    "is passed to `prepare_api`"
                )
            return None
        try:
            self.pl_wn = PolishWordnet(
                connector=_c,
                db_config_path=None,
                nx_graph_dir=None,
                extract_wiki_articles=False,
                use_memory_cache=use_memory_cache,
                show_progress_bar=False,
            )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error while preparing wordnet: {e}")
            return None

        if self.logger:
            self.logger.debug("Polish Wordnet connection established")
        return self.pl_wn

    def test_plwordnet(self):
        """
        Performs a comprehensive test of the PolishWordnet API functionality.

        This method validates the wordnet connection by retrieving and logging sample
        data from various API endpoints, including lexical units, lexical relations,
        synsets, synset relations, and relation types. Uses the configured limit
        or defaults to 5 items per test.

        The test results are logged for verification and debugging purposes.
        """

        limit = self.args.limit
        if limit is None:
            limit = 5

        if self.logger:
            self.logger.info(f"Testing plwordnet with limit: {limit}")

        try:
            sample_lu = self.pl_wn.api.get_lexical_units(limit=limit)
            if self.logger:
                self.logger.info(f"Number of lexical units: {len(sample_lu)}")
                for l in sample_lu:
                    self.logger.info(f"   - LU: {l}")

            sample_lu_rels = self.pl_wn.api.get_lexical_relations(limit=limit)
            if self.logger:
                self.logger.info(
                    f"Number of lexical relations: {len(sample_lu_rels)}"
                )
                for rel_lu in sample_lu_rels:
                    self.logger.info(f"   - Rel LU: {rel_lu}")

            sample_syn = self.pl_wn.api.get_synsets(limit=limit)
            if self.logger:
                self.logger.info(f"Number of synsets: {len(sample_syn)}")
                for s in sample_syn:
                    self.logger.info(f"   - Synset: {s}")

            sample_syn_rels = self.pl_wn.api.get_synset_relations(limit=limit)
            if self.logger:
                self.logger.info(
                    f"Number of synset relations: {len(sample_syn_rels)}"
                )
                for rel_syn in sample_syn_rels:
                    self.logger.info(f"   - Rel Synset: {rel_syn}")

            rel_types = self.pl_wn.api.get_relation_types(limit=limit)
            if self.logger:
                self.logger.info(f"Number of relation types: {len(rel_types)}")
                for rel_t in rel_types:
                    self.logger.info(f"   - Rel type: {rel_t}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error while testing plwordnet: {e}")
