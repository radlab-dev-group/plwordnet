import argparse

from plwordnet_handler.cli.constants import (
    DEFAULT_LOG_LEVEL,
    DEFAULT_NX_GRAPHS_DIR,
    DEFAULT_DB_CFG_PATH,
)


def prepare_base_parser(
    description: str, example_usage: str
) -> argparse.ArgumentParser:
    """
    Creates and configures the command-line argument parser for the application.

    This function sets up an ArgumentParser instance with all base command-line
    options and arguments required for the Polish Wordnet processing
    operations, including database configuration and various operational flags.

    Returns:
        argparse.ArgumentParser: Fully configured argument parser ready to parse
        command-line arguments
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=example_usage,
    )

    # -------------------------------------------------------------------------
    # General connection options
    parser.add_argument(
        "--db-config",
        dest="db_config",
        type=str,
        default=DEFAULT_DB_CFG_PATH,
        help="Path to JSON file with database configuration "
        "(used for plwordnet-cli --convert-to-nx-graph "
        "or any database connection: --use-database)",
    )
    parser.add_argument(
        "--use-database",
        dest="use_database",
        action="store_true",
        help="Use MySQL database directly instead of "
        "NetworkX graphs (requires --db-config)",
    )
    parser.add_argument(
        "--nx-graph-dir",
        dest="nx_graph_dir",
        type=str,
        default=DEFAULT_NX_GRAPHS_DIR,
        help=f"Path to NetworkX graphs directory "
        f"(default: {DEFAULT_NX_GRAPHS_DIR}). For plwordnet-cli "
        f"--convert-to-nx-graph, this is the output directory base. "
        f"For loading graphs, this should point to the graphs subdirectory.",
    )

    # -------------------------------------------------------------------------
    # General, debug options, limit etc.
    parser.add_argument(
        "--log-level",
        dest="log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=DEFAULT_LOG_LEVEL,
        help="Set the logging level",
    )
    parser.add_argument(
        "--limit",
        dest="limit",
        type=int,
        required=False,
        help="Limit the number of results to check app is proper working.",
    )

    return parser
