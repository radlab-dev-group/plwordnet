import logging

from abc import ABC
from typing import Optional, List, Dict, Any, Tuple

from plwordnet_handler.base.structure.elems.lu import LexicalUnit, LexicalUnitMapper
from plwordnet_handler.base.structure.elems.rel_type import (
    RelationType,
    RelationTypeMapper,
)
from plwordnet_handler.base.structure.elems.synset import Synset, SynsetMapper
from plwordnet_handler.base.structure.elems.synset_relation import (
    SynsetRelation,
    SynsetRelationMapper,
)
from plwordnet_handler.base.structure.elems.lu_relations import (
    LexicalUnitRelation,
    LexicalUnitRelationMapper,
)
from plwordnet_handler.base.structure.elems.lu_in_synset import (
    LexicalUnitAndSynset,
    LexicalUnitAndSynsetMapper,
)
from plwordnet_handler.base.connectors.db.config import DbSQLConfig
from plwordnet_handler.base.connectors.db.mysql import MySQLDbConnection
from plwordnet_handler.base.connectors.connector_i import PlWordnetConnectorInterface


class _PlWordnetAPIMySQLDbConnectorBase(PlWordnetConnectorInterface, ABC):
    """
    Database connector implementation for plWordnet API using MySQL.
    """

    def __init__(self, db_config_path: str):
        """
        Initialize plWordnet database connector.

        Args:
            db_config_path: Path to JSON file with database configuration

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            json.JSONDecodeError: If the JSON is invalid
            KeyError: If required configuration keys are missing
        """
        self.db_config_path = db_config_path
        self.config: Optional[DbSQLConfig] = None
        self.connection: Optional[MySQLDbConnection] = None
        self.logger = logging.getLogger(__name__)

        # Initialize configuration and connection
        self.__load_config()
        self.__initialize_connection()

    def connect(self) -> bool:
        """
        Establish connection to the database.

        Returns:
            bool: True if the connection is successful, False otherwise
        """
        if not self.connection:
            self.logger.error("Connection not initialized")
            return False
        success = self.connection.connect()
        if success:
            self.logger.info("Successfully connected to plWordnet database")
        else:
            self.logger.error("Failed to connect to plWordnet database")
        return success

    def disconnect(self) -> None:
        """
        Close database connection.
        """
        if self.connection:
            self.connection.disconnect()
            self.logger.info("Disconnected from plWordnet database")

    def is_connected(self) -> bool:
        """
        Check if the database connection is active.

        Returns:
            bool: True if connected, False otherwise
        """
        return self.connection and self.connection.is_connected()

    def _execute_select_query(
        self, query: str, params: Optional[Tuple] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a database query.

        Args:
            query: SQL query to execute
            params: Optional parameters for the query

        Returns:
            List of dictionaries with query results, or None if error
        """
        if not self.is_connected():
            self.logger.error("Not connected to database")
            return None

        try:
            results = self.connection.execute_select_query(
                query=query, params=params
            )
            self.logger.debug(
                f"Query executed successfully, returned "
                f"{len(results) if results else 0} results"
            )
            return results
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            return None

    def __load_config(self) -> None:
        """
        Load MySQL configuration from a JSON file.
        """
        try:
            self.config = DbSQLConfig.from_json_file(config_path=self.db_config_path)
            self.logger.info(
                f"Database configuration loaded successfully "
                f"for database: {self.config.database}"
            )
        except Exception as e:
            self.logger.error(f"Failed to load database configuration: {e}")
            raise

    def __initialize_connection(self) -> None:
        """
        Initialize database connection.
        """
        if not self.config:
            raise ValueError("Configuration not loaded")

        try:
            self.connection = MySQLDbConnection(**self.config.to_dict())
            self.logger.info("Database connection initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize connection: {e}")
            raise


class _PlWordnetAPIMySQLDbConnectorQueries(_PlWordnetAPIMySQLDbConnectorBase, ABC):
    """
    Abstract base class containing SQL queries and helper methods
    for PlWordnet MySQL database operations.

    Contains predefined SQL queries and methods for executing queries
    with optional result limiting functionality.
    """

    LIMIT_QUERY = "LIMIT"
    # lexical units
    GET_ALL_LU = "LEXICAL_UNITS"
    GET_LU_WITH_ID = "LEXICAL_UNIT_WITH_ID"
    GET_ALL_LU_RELS = "LEXICAL_UNITS_RELATIONS"
    # synsets
    GET_ALL_SYN = "SYN_SETS"
    GET_SYN_WITH_ID = "SYNSET_WITH_ID"
    GET_ALL_SYN_RELATION = "SYN_SETS_RELATIONS"
    # lu in synset
    GET_ALL_LU_IN_SYN = "LU_IN_SYNSET"
    # relations
    GET_ALL_REL_TYPES = "RELATION_TYPES"

    Q = {
        GET_ALL_LU: "SELECT * FROM lexicalunit",
        GET_LU_WITH_ID: "SELECT * FROM lexicalunit WHERE ID = %s",
        GET_ALL_LU_RELS: "SELECT * FROM lexicalrelation",
        GET_ALL_SYN: "SELECT * FROM synset",
        GET_SYN_WITH_ID: "SELECT * FROM synset WHERE ID = %s",
        GET_ALL_SYN_RELATION: "SELECT * FROM synsetrelation",
        GET_ALL_LU_IN_SYN: "SELECT * FROM unitandsynset",
        GET_ALL_REL_TYPES: "SELECT * FROM relationtype",
    }

    def _limit_query(self, query: str, limit: int):
        """
        Adds a LIMIT clause to the SQL query.

        Args:
            query: SQL query to modify
            limit: Maximum number of results (must be > 0)

        Returns:
            str: SQL query with added LIMIT clause

        Raises:
            AssertionError: If limit <= 0
        """
        assert limit > 0
        return f"{query} {self.LIMIT_QUERY} {limit}"

    def _execute_query_with_limit_opt(
        self, query: str, params: Optional[Tuple] = None, limit: Optional[int] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Executes a SELECT query with optional result limiting.

        Args:
            query: SQL query to execute
            params: Optional query parameters
            limit: Optional limit for number of results

        Returns:
            List of dictionaries with query results or None on error
        """
        if limit:
            query = self._limit_query(query=query, limit=limit)
        return self._execute_select_query(query=query, params=params)


class PlWordnetAPIMySQLDbConnector(_PlWordnetAPIMySQLDbConnectorQueries):
    """
    Main MySQL database connector class for PlWordnet API.

    Implements the PlWordnetConnectorInterface and provides
    concrete methods for retrieving lexical units and lexical
    relations from a MySQL database.
    """

    def get_lexical_unit(self, lu_id: int) -> Optional[LexicalUnit]:
        """
        Retrieves a lexical unit by its unique identifier.

        This method executes a database query to fetch lexical unit data using the
        provided lexical unit ID and maps the result to a LexicalUnit object using
        the LexicalUnitMapper.

        Args:
            lu_id (int): The unique identifier of the lexical unit to retrieve

        Returns:
            Optional[LexicalUnit]: The lexical unit object if found, None if no
            lexical unit exists with the given ID or if the query fails
        """
        if not self.is_connected():
            self.logger.error("Not connected to database")
            return None

        data_list = self._execute_select_query(
            query=self.Q[self.GET_LU_WITH_ID], params=(lu_id,)
        )
        if not data_list:
            return None
        return LexicalUnitMapper.map_from_dict(data=data_list[0])

    def get_lexical_units(
        self, limit: Optional[int] = None
    ) -> Optional[List[LexicalUnit]]:
        """
        Retrieves lexical units from the database.

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of LexicalUnit objects or None if no data available
        """
        if not self.is_connected():
            self.logger.error("Not connected to database")
            return None

        data_list = self._execute_query_with_limit_opt(
            query=self.Q[self.GET_ALL_LU],
            limit=limit,
        )
        if not data_list:
            return None
        return LexicalUnitMapper.map_from_dict_list(data_list=data_list)

    def get_lexical_relations(
        self, limit: Optional[int] = None
    ) -> Optional[List[LexicalUnitRelation]]:
        """
        Retrieves lexical relations from the database.

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of LexicalUnitRelation objects or None on error
        """
        if not self.is_connected():
            self.logger.error("Not connected to database")
            return None

        data_list = self._execute_query_with_limit_opt(
            query=self.Q[self.GET_ALL_LU_RELS],
            limit=limit,
        )
        if not data_list:
            return None
        return LexicalUnitRelationMapper.map_from_dict_list(data_list=data_list)

    def get_synset(self, syn_id: int) -> Optional[Synset]:
        """
        Retrieves a synset by its unique identifier.

        This method executes a database query to fetch synset data
        using the provided synset ID and maps the result to
        a Synset object using the SynsetMapper.

        Args:
            syn_id (int): The unique identifier of the synset to retrieve

        Returns:
            Optional[Synset]: The synset object if found, None if
            no synset exists with the given ID or if the query fails
        """
        if not self.is_connected():
            self.logger.error("Not connected to database")
            return None

        data_list = self._execute_select_query(
            query=self.Q[self.GET_SYN_WITH_ID], params=(syn_id,)
        )
        if not data_list:
            return None
        return SynsetMapper.map_from_dict(data=data_list[0])

    def get_synsets(self, limit: Optional[int] = None) -> Optional[List[Synset]]:
        """
        Retrieves synsets from the database.

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of Synset objects or None if no data available
        """
        if not self.is_connected():
            self.logger.error("Not connected to database")
            return None

        data_list = self._execute_query_with_limit_opt(
            query=self.Q[self.GET_ALL_SYN],
            limit=limit,
        )
        if not data_list:
            return None
        return SynsetMapper.map_from_dict_list(data_list=data_list)

    def get_synset_relations(
        self, limit: Optional[int] = None
    ) -> Optional[List[SynsetRelation]]:
        """
        Retrieves synset relations from the database.

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of SynsetRelation objects or None if no data available
        """
        if not self.is_connected():
            self.logger.error("Not connected to database")
            return None

        data_list = self._execute_query_with_limit_opt(
            query=self.Q[self.GET_ALL_SYN_RELATION],
            limit=limit,
        )
        if not data_list:
            return None
        return SynsetRelationMapper.map_from_dict_list(data_list=data_list)

    def get_units_and_synsets(
        self, limit: Optional[int] = None
    ) -> Optional[List[LexicalUnitAndSynset]]:
        """
        Retrieves `units and synsets` from the database.

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of LexicalUnitAndSynset objects or None if no data available
        """
        if not self.is_connected():
            self.logger.error("Not connected to database")
            return None

        data_list = self._execute_query_with_limit_opt(
            query=self.Q[self.GET_ALL_LU_IN_SYN],
            limit=limit,
        )
        if not data_list:
            return None
        return LexicalUnitAndSynsetMapper.map_from_dict_list(data_list=data_list)

    def get_relation_types(
        self, limit: Optional[int] = None
    ) -> Optional[List[RelationType]]:
        """
        Retrieves relation types from the database.

        Args:
            limit: Optional limit for the number of results

        Returns:
            List of RelationType objects or None on error
        """
        if not self.is_connected():
            self.logger.error("Not connected to database")
            return None

        data_list = self._execute_query_with_limit_opt(
            query=self.Q[self.GET_ALL_REL_TYPES],
            limit=limit,
        )
        if not data_list:
            return None
        return RelationTypeMapper.map_from_dict_list(data_list=data_list)
