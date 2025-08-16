"""
Default configuration constants for the Polish Wordnet applications.

This module defines the default values used throughout the application for
logging, file paths, and database configuration. These constants provide
fallback values when the user does not provide specific configurations.

Constants:
    DEFAULT_LOG_LEVEL: Default logging level for application messages
    DEFAULT_MILVUS_DB_CFG_PATH: Default path to Milvus database configuration
        file, with fallback if resource path utility returns None

"""

from plwordnet_handler.utils.resource_paths import (
    get_default_milvus_db_config_path,
    ResourcePaths,
)

DEFAULT_MILVUS_DB_CFG_PATH = (
    get_default_milvus_db_config_path()
    or f"{ResourcePaths.RESOURCES_SUBDIR}/{ResourcePaths.DEFAULT_PLWN_MILVUS_CONFIG}"
)
