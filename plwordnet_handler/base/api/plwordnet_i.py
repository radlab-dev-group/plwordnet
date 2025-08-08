import logging

from abc import ABC
from typing import Optional

from plwordnet_handler.base.connectors.connector_i import PlWordnetConnectorInterface


class PlWordnetAPIBase(ABC):
    def __init__(
        self,
        connector: PlWordnetConnectorInterface,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize plWordnetAPI with a given connector.

        Args:
            connector: connector to plwordnet (PlWordnetConnectorInterface)
            logger: logger to use for logging, if not given,
            then a new one will be created.
        """
        self.connector = connector
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def connect(self) -> bool:
        """
        Establish connection using connector

        Returns:
            bool: True if the connection is successful, False otherwise
        """
        return self.connector.connect()

    def disconnect(self) -> None:
        """
        Close connection from connector.
        """
        self.connector.disconnect()

    def is_connected(self) -> bool:
        """
        Check if the connector is active.

        Returns:
            bool: True if connected, False otherwise
        """
        return self.connector.is_connected()

    def __enter__(self):
        """
        Context manager entry - establish connection.
        """
        if self.connect():
            return self
        else:
            raise ConnectionError("Failed to establish database connection")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - close connection.
        """
        self.disconnect()
