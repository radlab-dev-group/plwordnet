from typing import Optional

from plwordnet_handler.base.connectors.nx.nx_connector import PlWordnetAPINxConnector


def connect_to_networkx_graphs(
    nx_graph_dir: str, connect: bool = True, logger=None
) -> Optional[PlWordnetAPINxConnector]:
    """
    Creates and optionally connects a NetworkX-based Polish Wordnet API connector.

    This function initializes a connector that can work with Polish Wordnet data
    stored in NetworkX graph format files. The connector can be created with or
    without an immediate connection establishment.

    Args:
        nx_graph_dir (str): Directory path containing NetworkX graph files
        connect (bool): Whether to establish connection immediately (default: True)
        logger: Logger instance for recording operation status and errors

    Returns:
        Optional[PlWordnetAPINxConnector]: Configured connector instance on success,
        None if connection fails when connect=True

    Note:
        If connect=False, the connector is returned without attempting connection,
        allowing for a deferred connection establishment.
    """
    if logger:
        logger.info("Loading data from NetworkX graphs")

    connector = PlWordnetAPINxConnector(nx_graph_dir=nx_graph_dir)
    if connect:
        try:
            connector.connect()
        except Exception as e:
            if logger:
                logger.error(f"Error loading NetworkX graphs: {e}")
                return None
    return connector
