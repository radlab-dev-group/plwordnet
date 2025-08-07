"""
Default configuration constants for the Polish Wordnet applications.

This module defines the default values used throughout the application for
logging, file paths, and database configuration. These constants provide
fallback values when the user does not provide specific configurations.

Constants:
    DEFAULT_LOG_LEVEL: Default logging level for application messages
    DEFAULT_NX_GRAPHS_DIR: Default directory containing NetworkX graph files,
                          with fallback if resource path utility returns None
    DEFAULT_DB_CFG_PATH: Default path to MySQL database configuration file,
                        with fallback if resource path utility returns None
"""

from plwordnet_handler.utils.resource_paths import (
    get_default_graph_path,
    get_default_db_config_path,
)

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_NX_OUT_DIR = "resources/plwordnet"
DEFAULT_NX_GRAPHS_DIR = get_default_graph_path() or f"{DEFAULT_NX_OUT_DIR}/nx/graphs"
DEFAULT_DB_CFG_PATH = (
    get_default_db_config_path() or "resources/plwordnet-mysql-db.json"
)
