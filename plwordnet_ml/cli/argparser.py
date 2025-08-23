import argparse

from plwordnet_ml.cli.example_usage import EXAMPLE_USAGE
from plwordnet_handler.cli.base_argparser import prepare_base_parser
from plwordnet_ml.cli.constants import (
    DEFAULT_MILVUS_DB_CFG_PATH,
    DEFAULT_EMBEDDER_CFG_PATH,
)


def prepare_parser() -> argparse.ArgumentParser:
    """
    Creates and configures the command-line argument parser for the application.

    This function sets up an ArgumentParser instance with all the necessary
    command-line options and arguments required for the Milvus processing
    operations, including database configuration (schema, indexes) etc.

    Returns:
        argparse.ArgumentParser: Fully configured argument parser ready to parse
        command-line arguments
    """
    parser = prepare_base_parser(
        description="Polish Wordnet Milvus connector",
        example_usage=EXAMPLE_USAGE,
    )

    # -------------------------------------------------------------------------
    # General connection options
    parser.add_argument(
        "--milvus-config",
        dest="milvus_config",
        type=str,
        default=DEFAULT_MILVUS_DB_CFG_PATH,
        help="Path to JSON file with database configuration.",
    )

    parser.add_argument(
        "--embedder-config",
        dest="embedder_config",
        type=str,
        default=DEFAULT_EMBEDDER_CFG_PATH,
        help="Path to JSON file with embedder configuration.",
    )

    # -------------------------------------------------------------------------
    # Operations
    parser.add_argument(
        "--prepare-database",
        dest="prepare_database",
        default=False,
        action="store_true",
        help="If option is given the database will be prepared.",
    )

    parser.add_argument(
        "--prepare-base-embeddings-lu",
        dest="prepare_base_embeddings_lu",
        default=False,
        action="store_true",
        help="If option is given the base embeddings for lexical units "
        "and its examples will be prepared. LU emb is mean of examples.",
    )

    parser.add_argument(
        "--prepare-base-mean-empty-embeddings-lu",
        dest="prepare_mean_empty_base_embeddings_lu",
        default=False,
        action="store_true",
        help="If option is given the mean base embedding will be prepared "
        "and inserted in Milvus in case when other LU from synset are available.",
    )

    parser.add_argument(
        "--prepare-base-embeddings-synset",
        dest="prepare_base_embeddings_synset",
        default=False,
        action="store_true",
        help="If option is given the base embeddings for synset will be prepared.",
    )

    # -------------------------------------------------------------------------

    parser.add_argument(
        "--export-relgat-mapping",
        dest="export_relgat_mapping",
        default=False,
        action="store_true",
        help="Export dataset to RelGAT (relations embeddings) "
        "training to given output path.",
    )
    parser.add_argument(
        "--relgat-mapping-directory",
        dest="relgat_mapping_directory",
        type=str,
        help="Path to directory with RelGAT mapping.",
    )

    # -------------------------------------------------------------------------
    # General, debug options
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default="cpu",
        help="Device to use (cpu, cuda, cuda:0) - depends on machine",
    )

    return parser
