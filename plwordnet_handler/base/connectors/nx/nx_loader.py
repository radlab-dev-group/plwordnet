from typing import Optional

from plwordnet_handler.base.connectors.nx.nx_connector import PlWordnetAPINxConnector
from plwordnet_handler.utils.logger import prepare_logger


def connect_to_networkx_graphs(
    nx_graph_dir: str, connect: bool = True, log_level: str = "INFO"
) -> Optional[PlWordnetAPINxConnector]:
    """
    Creates and optionally connects a NetworkX-based Polish Wordnet API connector.

    This function initializes a connector that can work with Polish Wordnet data
    stored in NetworkX graph format files. The connector can be created with or
    without an immediate connection establishment.

    Args:
        nx_graph_dir (str): Directory path containing NetworkX graph files
        connect (bool): Whether to establish connection immediately (default: True)
        log_level: Logger level (default is INFO)

    Returns:
        Optional[PlWordnetAPINxConnector]: Configured connector instance on success,
        None if connection fails when connect=True

    Note:
        If connect=False, the connector is returned without attempting connection,
        allowing for a deferred connection establishment.
    """
    logger = prepare_logger(logger_name=__name__, log_level=log_level)
    logger.info("Loading data from NetworkX graphs")

    try:
        connector = PlWordnetAPINxConnector(
            nx_graph_dir=nx_graph_dir, log_level=log_level
        )
        if connect:
            if not connector.connect():
                logger.error("Connection failed")
                return None
        return connector
    except Exception as e:
        logger.error(f"Error loading NetworkX graphs: {e}")
        return None
