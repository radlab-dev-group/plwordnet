from abc import ABC, abstractmethod
from typing import Optional

from plwordnet_handler.utils.logger import prepare_logger
from plwordnet_handler.base.structure.polishwordnet import PolishWordnet
from plwordnet_handler.base.connectors.connector_i import PlWordnetConnectorInterface
from plwordnet_handler.base.connectors.nx.nx_connector import PlWordnetAPINxConnector
from plwordnet_handler.base.connectors.nx.nx_loader import connect_to_networkx_graphs
from plwordnet_handler.base.connectors.db.db_connector import (
    PlWordnetAPIMySQLDbConnector,
)
from plwordnet_handler.base.connectors.db.db_loader import connect_to_mysql_database


class CLIWrapperBase(ABC):
    """
    Abstract class for any CLI wrapper
    """

    def __init__(
        self,
        args,
        verify_args: bool,
        log_level: str = "INFO",
        log_name: Optional[str] = None,
        log_filename: Optional[str] = None,
    ) -> None:
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
        self.logger = prepare_logger(
            logger_name=log_name, log_level=log_level, logger_file_name=log_filename
        )

        self.pl_wn = None
        self.last_connector = None

        if verify_args:
            self.are_args_correct(args=args)

    @abstractmethod
    def are_args_correct(self, args=None) -> bool:
        """
        Validates the correctness of command-line arguments.

        Args:
            args: Optional arguments to validate (uses self.args if None)

        Returns:
            bool: Whether the arguments were correct

        Raises:
            TypeError: If any errors occur
        """
        raise NotImplementedError("Not implemented")

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
        if self.last_connector is None:
            self.logger.error(
                "Error during connecting to MySQL database. Check database config."
            )
            return None
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

    def prepare_wordnet_based_on_args(
        self, use_memory_cache: bool = True
    ) -> Optional[PolishWordnet]:
        """
        Prepare the connector based on arguments. Graph or database connection
        will be established and used into internal wordnet API.

        When:
          --use-database is given, then the MySQL DB connector will be used.
          --nx-graph-dir is given, then NetworkX connector will be used.

        Arguments:
            use_memory_cache (bool): Whether to enable memory caching
            for efficient api usage

        Raises:
            ValueError: If no connector arguments are provided
        """
        if self.args.use_database:
            connector = self.connect_to_database()
        elif self.args.nx_graph_dir:
            connector = self.connect_to_networkx_graphs()
        else:
            raise Exception(
                "--use-database or --nx-graph-dir is required "
                "to connect to Plwordnet API Interface"
            )

        if connector is None:
            self.logger.error("Could not connect to Plwordnet API!")
            return None

        # Prepare wordnet with connector
        wordnet = self.prepare_wordnet_with_connector(
            connector=connector, use_memory_cache=use_memory_cache
        )
        if wordnet is None:
            self.logger.error(
                "Could not connect to Plwordnet API with actual connector!"
            )
            self.logger.error("Try to change connector parameters and try again.")
            return None
        return wordnet

    def _connection_ok(self):
        """
        Checks if the connection (connector) is properly initialized and available.

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
            self.logger.error("Connector has not been initialized!")
            return False
        return True
