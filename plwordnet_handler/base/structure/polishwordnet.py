from typing import Optional

from plwordnet_handler.base.api.plwordnet import PlWordnetAPI
from plwordnet_handler.base.connectors.db.db_connector import (
    PlWordnetAPIMySQLDbConnector,
)
from plwordnet_handler.base.connectors.connector_i import PlWordnetConnectorInterface
from plwordnet_handler.base.connectors.nx.nx_connector import PlWordnetAPINxConnector


class PolishWordnet:
    """
    Class representing a Słowosieć structure
    """

    def __init__(
        self,
        connector: PlWordnetConnectorInterface = None,
        db_config_path: Optional[str] = None,
        nx_graph_dir: Optional[str] = None,
        extract_wiki_articles: bool = False,
        use_memory_cache: bool = False,
        show_progress_bar: bool = False,
    ):
        """
        Initialize PolishWordnet with PlWordnetAPI.

        Args:
            connector: Optional connector instance (PlWordnetConnectorInterface)
            db_config_path: Optional path to a database configuration file.
            nx_graph_dir: Optional path to NetworkX graphs directory.
            extract_wiki_articles: Whether to extract Wikipedia articles.
            use_memory_cache: Whether to use memory cache.
            show_progress_bar: Whether to show a progress bar.
        """
        # Create connector with provided or default configuration
        if connector is None:
            if nx_graph_dir:
                connector = PlWordnetAPINxConnector(nx_graph_dir=nx_graph_dir)
            elif db_config_path:
                connector = PlWordnetAPIMySQLDbConnector(
                    db_config_path=db_config_path
                )
            else:
                raise Exception(
                    "Connector, nx_graph_dir, or db_config_path must be provided."
                )

        self.api = PlWordnetAPI(
            connector=connector,
            extract_wiki_articles=extract_wiki_articles,
            use_memory_cache=use_memory_cache,
            show_progress_bar=show_progress_bar,
        )

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying PlWordnetAPI instance
        or use implementation from this class.

        Args:
            name: Name of the attribute to access

        Returns:
            The requested attribute from the PlWordnetAPI instance or self
        """
        if name in self.api.DELEGATED_METHODS:
            return getattr(self.api, name)

        try:
            return super().__getattribute__(name)
        except AttributeError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def __enter__(self):
        """
        Context manager entry - establish connection.
        """
        if self.api.connect():
            return self
        else:
            raise ConnectionError("Failed to load connector")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - close connection.
        """
        self.api.disconnect()
