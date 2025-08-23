"""
Default configuration constants for the Polish Wordnet applications.

This module defines the default values used throughout the application for
logging, file paths, and database configuration. These constants provide
fallback values when the user does not provide specific configurations.

Constants:
    DEFAULT_MILVUS_DB_CFG_PATH: Default path to the Milvus database configuration
        file, with fallback if the resource path utility returns None

"""

from plwordnet_handler.utils.resource_paths import (
    get_default_embedder_model_config_path,
    get_default_milvus_db_config_path,
    ResourcePaths,
)

DEFAULT_MILVUS_DB_CFG_PATH = (
    get_default_milvus_db_config_path()
    or f"{ResourcePaths.RESOURCES_SUBDIR}/"
    f"{ResourcePaths.DEFAULT_PLWN_MILVUS_CONFIG}"
)


DEFAULT_EMBEDDER_CFG_PATH = (
    get_default_embedder_model_config_path()
    or f"{ResourcePaths.RESOURCES_SUBDIR}/"
    f"{ResourcePaths.DEFAULT_EMBEDDER_MODEL_CONFIG}"
)
