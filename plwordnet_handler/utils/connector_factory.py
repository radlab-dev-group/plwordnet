from typing import Optional

from plwordnet_handler.base.connectors.nx.nx_connector import PlWordnetAPINxConnector
from plwordnet_handler.utils.resource_paths import ResourcePaths


class ConnectorFactory:
    """
    Factory for creating connectors with proper default paths.

    ``` python
    from plwordnet_handler.utils.connector_factory import ConnectorFactory
    from plwordnet_handler.base.structure.polishwordnet import PolishWordnet

    connector = ConnectorFactory.create_nx_connector()
    if connector:
        with PolishWordnet(connector=connector) as pl_wn:
            synsets = pl_wn.get_synsets(limit=10)
    else:
        print(
            "No installed graphs found. Please install with"
            "--data-graph-test or --data-graph-full")
    ```
    """

    @classmethod
    def create_nx_connector(
        cls, graph_dir: Optional[str] = None
    ) -> Optional[PlWordnetAPINxConnector]:
        """
        Create a NetworkX connector with installed graphs.

        Args:
            graph_dir: Custom graph directory (overrides auto-detection)

        Returns:
            PlWordnetAPINxConnector or None if no graphs are available
        """
        if graph_dir:
            return PlWordnetAPINxConnector(nx_graph_dir=graph_dir)

        # Auto-detect installed graphs
        graphs_dir = ResourcePaths.get_installed_graphs_dir()
        if graphs_dir:
            return PlWordnetAPINxConnector(nx_graph_dir=str(graphs_dir))
        return None
