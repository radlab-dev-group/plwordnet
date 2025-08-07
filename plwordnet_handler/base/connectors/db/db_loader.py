from plwordnet_handler.base.connectors.db.db_connector import (
    PlWordnetAPIMySQLDbConnector,
)


def connect_to_mysql_database(
    db_config_path: str, connect: bool = True, logger=None
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
        logger: Logger instance for recording connection status and error messages

    Returns:
        PlWordnetAPIMySQLDbConnector or None: Connected database connector
        instance on success, None if connection fails when connect=True

    Note:
        If connect=False, the connector is returned without attempting connection,
        allowing for a deferred connection establishment.
    """

    if logger:
        logger.info("Loading data from MySQL database")

    try:
        connector = PlWordnetAPIMySQLDbConnector(db_config_path=db_config_path)
        if connect:
            if not connector.connect():
                if logger:
                    logger.error("Connection failed")
                return None
        return connector
    except Exception as e:
        if logger:
            logger.error(f"Error loading from MySQL database: {e}")
        return None
