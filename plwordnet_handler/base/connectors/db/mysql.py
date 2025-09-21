import mysql.connector

from abc import ABC
from mysql.connector import Error
from typing import Optional, Dict, Any, List, Tuple

from rdl_ml_utils.utils.logger import prepare_logger


class MySQLConnectionI(ABC):
    def __init__(
        self,
        host: str,
        database: str,
        user: str,
        password: str,
        port: int = 3306,
        log_level: Optional[str] = "INFO",
    ):
        """
        Initialize connection to MySQL DBMS.

        Args:
            host: MySQL Host
            database: Name of database
            user: Name of user
            password: User's password
            port: MySQL port (optional, default: 3306)
        """
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.connection: Optional[mysql.connector.MySQLDbConnection] = None
        self.cursor: Optional[mysql.connector.cursor.MySQLCursor] = None

        self.logger = prepare_logger(logger_name=__name__, log_level=log_level)

    def connect(self) -> bool:
        """
        Establishes connection to MySQL database.

        Returns:
            bool: True if success, False otherwise.
        """
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port,
                charset="utf8mb4",
                use_unicode=True,
                autocommit=False,
            )

            if self.connection.is_connected():
                self.cursor = self.connection.cursor(dictionary=True)
                db_info = self.connection.get_server_info()
                self.logger.info(f"Connected to MySQL server version {db_info}")
                return True
            else:
                return False
        except Error as e:
            self.logger.error(f"Error connecting to MySQL: {e}")
            return False

    def disconnect(self) -> None:
        """
        Closes the MySQL database connection.
        """
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection and self.connection.is_connected():
                self.connection.close()
                self.logger.info("MySQL connection was closed.")
        except Error as e:
            self.logger.error(f"Error closing connection:: {e}")

    def execute_select_query(
        self, query: str, params: Optional[Tuple] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Executes a SELECT query and returns the results.

        Args:
            query: The SQL query to execute params
            params: Additional parameters to pass to the query (optional)

        Returns:
            A list of dictionaries with the query results,
            or None in case of an error.
        """
        try:
            if not self.cursor:
                self.logger.error("No active database connection")
                return None
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            results = self.cursor.fetchall()
            return results
        except Error as e:
            self.logger.error(f"Error during executing query: {e}")
            return None

    def is_connected(self) -> bool:
        """
        Checks if the database connection is active.

        Returns:
            bool: True if the connection is active, False otherwise.
        """
        return self.connection and self.connection.is_connected()


class MySQLDbConnection(MySQLConnectionI):
    def __enter__(self):
        """
        Method for the context manager - establishes the connection.
        """
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Method for the context manager - closes the connection.
        """
        self.disconnect()
