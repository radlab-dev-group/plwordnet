import argparse

from plwordnet_handler.cli.constants import (
    DEFAULT_LOG_LEVEL,
    DEFAULT_NX_GRAPHS_DIR,
    DEFAULT_DB_CFG_PATH,
)
from plwordnet_handler.cli.example_usage import EXAMPLE_USAGE


def prepare_parser() -> argparse.ArgumentParser:
    """
    Creates and configures the command-line argument parser for the application.

    This function sets up an ArgumentParser instance with all the necessary
    command-line options and arguments required for the Polish Wordnet processing
    operations, including database configuration, output paths, processing limits,
    and various operational flags.

    Returns:
        argparse.ArgumentParser: Fully configured argument parser ready to parse
        command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Polish Wordnet handler - supports both "
        "NetworkX graphs and MySQL database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLE_USAGE,
    )

    parser.add_argument(
        "--db-config",
        dest="db_config",
        type=str,
        default=DEFAULT_DB_CFG_PATH,
        help="Path to JSON file with database configuration "
        "(used for --convert-to-nx-graph or --use-database)",
    )

    parser.add_argument(
        "--nx-graph-dir",
        dest="nx_graph_dir",
        type=str,
        default=DEFAULT_NX_GRAPHS_DIR,
        help=f"Path to NetworkX graphs directory (default: {DEFAULT_NX_GRAPHS_DIR}). "
        f"For --convert-to-nx-graph, this is the output directory base. "
        f"For loading graphs, this should point to the graphs subdirectory.",
    )

    parser.add_argument(
        "--extract-wikipedia-articles",
        dest="extract_wikipedia_articles",
        action="store_true",
        help="Extract Wikipedia articles as additional LU description",
    )

    parser.add_argument(
        "--convert-to-nx-graph",
        dest="convert_to_nx",
        action="store_true",
        help="Convert from database to NetworkX graphs (requires --db-config)",
    )

    parser.add_argument(
        "--use-database",
        dest="use_database",
        action="store_true",
        help="Use MySQL database directly instead of "
        "NetworkX graphs (requires --db-config)",
    )

    parser.add_argument(
        "--limit",
        dest="limit",
        type=int,
        required=False,
        help="Limit the number of results to check app is proper working.",
    )

    parser.add_argument(
        "--log-level",
        dest="log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=DEFAULT_LOG_LEVEL,
        help="Set the logging level",
    )

    parser.add_argument(
        "--show-progress-bar",
        dest="show_progress_bar",
        action="store_true",
        help="Show progress bar",
    )

    parser.add_argument(
        "--test-api",
        dest="test_api",
        action="store_true",
        help="Test api connection. When --use-database is passed the connection "
        "to database will be tested, otherwise connection to NetworkX.",
    )

    return parser
