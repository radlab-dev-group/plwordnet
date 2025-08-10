from typing import Optional

from plwordnet_handler.utils.logger import prepare_logger
from plwordnet_handler.base.structure.polishwordnet import PolishWordnet
from plwordnet_handler.dataset.exporter.rel_types import RelationTypesExporter
from plwordnet_handler.base.connectors.db.db_to_nx import dump_to_networkx_file
from plwordnet_handler.dataset.exporter.embedder import WordnetToEmbedderConverter
from plwordnet_handler.base.connectors.connector_i import PlWordnetConnectorInterface
from plwordnet_handler.base.connectors.nx.nx_connector import PlWordnetAPINxConnector
from plwordnet_handler.base.connectors.nx.nx_loader import connect_to_networkx_graphs
from plwordnet_handler.base.connectors.db.db_connector import (
    PlWordnetAPIMySQLDbConnector,
)
from plwordnet_handler.base.connectors.db.db_loader import connect_to_mysql_database


class CLIWrappers:
    def __init__(self, args, verify_args: bool, log_level: str = "INFO") -> None:
        """
        Initializes the CLI wrapper with command-line
        arguments and optional verification.

        Args:
            args: Parsed command-line arguments containing configuration options
            verify_args (bool): Whether to validate the provided arguments
            log_level: Logger level (INFO default)
        """

        self.args = args
        self.log_level = log_level
        self.logger = prepare_logger(logger_name=__name__, log_level=log_level)

        self.pl_wn = None
        self.last_connector = None

        if verify_args:
            self.are_args_correct(args=args)

    def are_args_correct(self, args=None):
        """
        Validates the correctness of command-line arguments.

        Args:
            args: Optional arguments to validate (uses self.args if None)

        Raises:
            TypeError: If any errors occur
        """
        if args is None:
            args = self.args

        if args is None:
            raise TypeError("No arguments to check are provided")

        # If --test-api, Then no converters may be used
        if args.test_api:
            # no: --dump-embedder-dataset-to-filE
            if args.dump_embedder_dataset_to_file:
                raise TypeError(
                    "--test-api cannot be used with --dump-embedder-dataset-to-file"
                )
            # no: --convert-to-nx-graph
            if args.convert_to_nx:
                raise TypeError(
                    "--test-api cannot be used with --convert-to-nx-graph"
                )
            # no: --dump-relation-types-to-file
            if args.dump_relation_types_to_file:
                raise TypeError(
                    "--test-api cannot be used with --dump-relation-types-to-file"
                )

        # Check if --convert-to-nx-graph
        if args.convert_to_nx:
            # check --db-config
            if not self.args.db_config:
                raise TypeError(
                    "No database configuration provided, to prepare graph "
                    "you have to pass --db-config to proper database."
                )
            # check --nx-graph-dir
            if not self.args.nx_graph_dir:
                raise TypeError(
                    "No output directory is provided, to prepare graph you have to "
                    "pass --nx-graph-dir to store converted NetworkX graphs."
                )

        # Check if --dump-embedder-dataset-to-file
        if args.dump_embedder_dataset_to_file:
            # --xlsx-relations-weights <- required
            if not args.xlsx_relations_weights:
                raise TypeError(
                    "--dump-embedder-dataset-to-file requires additional option "
                    "--xlsx-relations-weights (path to relations weights)"
                )

        return True

    def dump_to_networkx_file(self) -> bool:
        """
        Private wrapper method that extracts database configuration and parameters
        from args and delegates to the main dump_to_networkx_file function
        to export graph data to NetworkX format.

        Returns:
            Boolean: True for successful completion, False for error
        """
        self.logger.debug("Starting dump_to_networkx_file")
        return dump_to_networkx_file(
            db_config=self.args.db_config,
            out_dir_path=self.args.nx_graph_dir,
            limit=self.args.limit,
            show_progress_bar=self.args.show_progress_bar,
            extract_wikipedia_articles=self.args.extract_wikipedia_articles,
            log_level=self.log_level,
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
        try:
            self.last_connector = connect_to_networkx_graphs(
                nx_graph_dir=self.args.nx_graph_dir,
                connect=True,
                log_level=self.log_level,
            )
            if self.last_connector is None:
                self.logger.error(
                    "In case when graphs files do not exist. Run plwordnet-cli "
                    "with option --convert-to-nx-graph to use plwordnet stored "
                    "in NetworkX format or pass --use-database in case to use "
                    "database with default database config or--db-config=PATH "
                    "to use database specified in the config."
                )
            return self.last_connector
        except Exception as e:
            self.logger.error(e)
            return None

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
            log_level=self.log_level,
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
            self.logger.error(f"Error while preparing wordnet: {e}")
            return None

        self.logger.debug("Polish Wordnet connection established")
        return self.pl_wn

    def test_plwordnet(self) -> bool:
        """
        Performs a comprehensive test of the PolishWordnet API functionality.

        This method validates the wordnet connection by retrieving and logging sample
        data from various API endpoints, including lexical units, lexical relations,
        synsets, synset relations, and relation types. Uses the configured limit
        or defaults to 5 items per test.

        The test results are logged for verification and debugging purposes.

        Returns:
            bool: True if the test passes, False otherwise
        """
        if not self._connection_ok():
            self.logger.error(
                "Testing is not possible because no connection is established!"
            )
            return False

        limit = self.args.limit
        if limit is None:
            limit = 5

        self.logger.info(f"Testing plwordnet with limit: {limit}")

        try:
            sample_lu = self.pl_wn.api.get_lexical_units(limit=limit)
            self.logger.info(f"Number of lexical units: {len(sample_lu)}")
            for l in sample_lu:
                self.logger.info(f"   - LU: {l} ({l.comment}")

            sample_lu_rels = self.pl_wn.api.get_lexical_relations(limit=limit)
            self.logger.info(f"Number of lexical relations: {len(sample_lu_rels)}")
            for rel_lu in sample_lu_rels:
                self.logger.info(f"   - Rel LU: {rel_lu}")

            sample_syn = self.pl_wn.api.get_synsets(limit=limit)
            self.logger.info(f"Number of synsets: {len(sample_syn)}")
            for s in sample_syn:
                self.logger.info(f"   - Synset: {s} ({s.comment})")

            sample_syn_rels = self.pl_wn.api.get_synset_relations(limit=limit)
            self.logger.info(f"Number of synset relations: {len(sample_syn_rels)}")
            for rel_syn in sample_syn_rels:
                self.logger.info(f"   - Rel Synset: {rel_syn}")

            rel_types = self.pl_wn.api.get_relation_types(limit=limit)
            self.logger.info(f"Number of relation types: {len(rel_types)}")
            for rel_t in rel_types:
                self.logger.info(f"   - Rel type: {rel_t}")
        except Exception as e:
            self.logger.error(f"Error while testing plwordnet: {e}")
            return False

        self.logger.info("Test passed. All data are correct.")
        return True

    def dump_relation_types_to_file(self) -> bool:
        """
        Exports relation types data to an Excel file using the active connector.

        This method creates a RelationTypesExporter instance and exports wordnet
        relation types to an XLSX file. The export uses the output file path
        specified in arguments and respects the configured data limit.

        Returns:
            bool: True if export is completed successfully, False if no connector
            is available, or export operation fails

        Note:
            Requires an active connector (self.last_connector) to be established
            before calling this method.
        """
        if not self._connection_ok():
            return False

        exporter = RelationTypesExporter(connector=self.last_connector)
        success = exporter.export_to_xlsx(
            output_file=self.args.dump_relation_types_to_file, limit=self.args.limit
        )
        if not success:
            self.logger.error("Failed to dump relation types to file")
            return False
        return True

    def dump_embedder_dataset_to_file(self):
        """
        Dumps embedder dataset to a file using WordNet relations and weights.

        This method creates a WordNet to embedder converter instance and exports
        the processed dataset to a specified output file. It first checks if the
        database connection is available, then initializes a converter with the
        configured Excel relations weights file and database connector, and finally
        exports the dataset with the specified parameters.

        Returns:
            bool: True if the dataset was successfully exported, False if the
                  database connection failed or export encountered an error

        Note:
            The method relies on command-line arguments for configuration,
            including the Excel weights file path, output file path,
            data limit, and low-to-high ratio for dataset balancing.
        """
        if not self._connection_ok():
            return False

        converter = WordnetToEmbedderConverter(
            xlsx_path=self.args.xlsx_relations_weights,
            pl_wordnet=self.pl_wn,
            init_converter=True,
        )

        return converter.export(
            output_file=self.args.dump_embedder_dataset_to_file,
            limit=self.args.limit,
            low_high_ratio=self.args.embedder_low_high_ratio,
        )

    def _connection_ok(self):
        """
        Checks if the connection is properly initialized and available.

        This private method validates whether the last_connector attribute contains
        a valid database/networkx connector instance. If no connector
        is available, it logs an error message indicating that operations
        requiring database access cannot proceed.

        Returns:
            bool: True if a valid database connector exists,
            False if the connector is None or not initialized

        Note:
            This method serves as a guard clause for connector-dependent
            operations and provides appropriate error logging when
            the connection is unavailable.
        """

        if self.last_connector is None:
            self.logger.error(
                "Cannot dump relation types to file. "
                "No connection was initialized."
            )
            return False
        return True
