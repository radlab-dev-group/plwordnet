from plwordnet_handler.base.connectors.db.db_connector import (
    PlWordnetAPIMySQLDbConnector,
)
from plwordnet_handler.utils.logger import prepare_logger


def connect_to_mysql_database(
    db_config_path: str, connect: bool = True, log_level: str = "INFO"
):
    """
    Creates and optionally connects a MySQL database connector
    for Polish Wordnet API.

    This function initializes a MySQL database connector using the
    provided configuration file and can establish an immediate
    connection or create the connector for later use.

    Args:
        db_config_path (str): Path to the database configuration file
        connect (bool): Whether to establish database
        connection immediately (default: True)
        log_level: Logger level (INFO default)

    Returns:
        PlWordnetAPIMySQLDbConnector or None: Connected database connector
        instance on success, None if connection fails when connect=True

    Note:
        If connect=False, the connector is returned without attempting
        connection, allowing for a deferred connection establishment.
    """
    logger = prepare_logger(logger_name=__name__, log_level=log_level)
    logger.info("Loading data from MySQL database")

    try:
        connector = PlWordnetAPIMySQLDbConnector(
            db_config_path=db_config_path,
            log_level=log_level,
        )
        if connect:
            if not connector.connect():
                logger.error("Connection failed")
                return None
        return connector
    except Exception as e:
        logger.error(f"Error loading from MySQL database: {e}")
        return None
