import argparse

from plwordnet_trainer.cli.constants import (
    DEFAULT_LOG_LEVEL,
    DEFAULT_MILVUS_DB_CFG_PATH,
)
from plwordnet_trainer.cli.example_usage import EXAMPLE_USAGE


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
    parser = argparse.ArgumentParser(
        description="Polish Wordnet Milvus connector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLE_USAGE,
    )

    # -------------------------------------------------------------------------
    # General connection options
    parser.add_argument(
        "--milvus-config",
        dest="milvus_config",
        type=str,
        default=DEFAULT_MILVUS_DB_CFG_PATH,
        help="Path to JSON file with database configuration "
        "(used for --convert-to-nx-graph or --use-database)",
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
        "--prepare-base-embeddings",
        dest="prepare_base_embeddings",
        default=False,
        action="store_true",
        help="If option is given the base embeddings will be prepared.",
    )

    # -------------------------------------------------------------------------
    # General, debug options
    parser.add_argument(
        "--log-level",
        dest="log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=DEFAULT_LOG_LEVEL,
        help="Set the logging level",
    )

    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default="cpu",
        help="Device to use (cpu, cuda, cuda:0) - depends on machine",
    )

    return parser
