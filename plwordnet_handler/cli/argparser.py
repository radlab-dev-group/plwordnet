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

    # -------------------------------------------------------------------------
    # General connection options
    parser.add_argument(
        "--db-config",
        dest="db_config",
        type=str,
        default=DEFAULT_DB_CFG_PATH,
        help="Path to JSON file with database configuration "
        "(used for --convert-to-nx-graph or --use-database)",
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
        help=f"Path to NetworkX graphs directory (default: {DEFAULT_NX_GRAPHS_DIR}). "
        f"For --convert-to-nx-graph, this is the output directory base. "
        f"For loading graphs, this should point to the graphs subdirectory.",
    )

    # -------------------------------------------------------------------------
    # Converting database to NetworkX graphs
    parser.add_argument(
        "--convert-to-nx-graph",
        dest="convert_to_nx",
        action="store_true",
        help="Convert from database to NetworkX graphs (requires --db-config)",
    )

    # -------------------------------------------------------------------------
    # Relation types export
    parser.add_argument(
        "--dump-relation-types-to-file",
        dest="dump_relation_types_to_file",
        type=str,
        required=False,
        help="If option is passed, then relation types "
        "will be dumped to file with given filepath.",
    )

    # -------------------------------------------------------------------------
    # Embedder export
    parser.add_argument(
        "--dump-embedder-dataset-to-file",
        dest="dump_embedder_dataset_to_file",
        type=str,
        required=False,
        help=f"Output JSONL file path where embedder dataset will be stored.",
    )
    parser.add_argument(
        "--xlsx-relations-weights",
        dest="xlsx_relations_weights",
        type=str,
        required=False,
        help=f"Path to Excel file with relation weights. If you dont have this file, "
        f"you can generate schema to prepare relations with option "
        f"--dump-relation-types-to-file=./relations.xlsx",
    )
    parser.add_argument(
        "--embedder-low-high-ratio",
        type=float,
        dest="embedder_low_high_ratio",
        required=False,
        default=2.0,
        help="Ratio between low and high-weighted relations "
        "count during dumping embedder dataset.",
    )
    parser.add_argument(
        "--extract-wikipedia-articles",
        dest="extract_wikipedia_articles",
        action="store_true",
        help="Extract Wikipedia articles as additional LU description",
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
        "--show-progress-bar",
        dest="show_progress_bar",
        action="store_true",
        help="Show progress bar",
    )
    parser.add_argument(
        "--limit",
        dest="limit",
        type=int,
        required=False,
        help="Limit the number of results to check app is proper working.",
    )

    # -------------------------------------------------------------------------
    # Test connection
    parser.add_argument(
        "--test-api",
        dest="test_api",
        action="store_true",
        help="Test api connection. When --use-database is passed the connection "
        "to database will be tested, otherwise connection to NetworkX.",
    )

    # -------------------------------------------------------------------------

    return parser
