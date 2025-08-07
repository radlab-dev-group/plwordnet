import sys
import logging
import argparse
from pathlib import Path

from plwordnet_handler.base.structure.polishwordnet import PolishWordnet
from plwordnet_handler.base.connectors.db.db_to_nx import dump_to_networkx_file
from plwordnet_handler.base.connectors.nx.nx_connector import PlWordnetAPINxConnector
from plwordnet_handler.base.connectors.db.db_connector import (
    PlWordnetAPIMySQLDbConnector,
)
from plwordnet_handler.utils.resource_paths import (
    get_default_graph_path,
    get_default_db_config_path,
)

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_NX_OUT_DIR = "resources/plwordnet"
DEFAULT_NX_GRAPHS_DIR = get_default_graph_path() or "resources/plwordnet/nx/graphs"
DEFAULT_DB_CFG_PATH = (
    get_default_db_config_path() or "resources/plwordnet-mysql-db.json"
)


EXAMPLE_USAGE = f"""
Example usage:

# Load from NetworkX graphs (default):
python plwordnet-cli \\
        --nx-graph-dir {DEFAULT_NX_GRAPHS_DIR} \\
        --extract-wikipedia-articles \\
        --log-level {DEFAULT_LOG_LEVEL}

# Convert from database to NetworkX graphs:
python plwordnet-cli \\
        --db-config {DEFAULT_DB_CFG_PATH} \\
        --convert-to-nx-graph \\
        --nx-graph-dir {DEFAULT_NX_OUT_DIR} \\
        --log-level {DEFAULT_LOG_LEVEL}

# Load from database directly:
python plwordnet-cli \\
        --use-database \\
        --db-config {DEFAULT_DB_CFG_PATH} \\
        --log-level {DEFAULT_LOG_LEVEL}
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("plwordnet_cli.log"),
    ],
)

logger = logging.getLogger(__name__)


def prepare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Polish Wordnet handler - supports both NetworkX graphs and MySQL database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLE_USAGE,
    )

    parser.add_argument(
        "--db-config",
        dest="db_config",
        type=str,
        default=DEFAULT_DB_CFG_PATH,
        help="Path to JSON file with database configuration (used for --convert-to-nx-graph or --use-database)",
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
        help="Use MySQL database directly instead of NetworkX graphs (requires --db-config)",
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

    return parser


def _dump_to_networkx_file(args) -> int:
    """
    Private wrapper method that extracts database configuration and parameters
    from args and delegates to the main dump_to_networkx_file function
    to export graph data to NetworkX format.

    Args:
        args: Command line arguments or configuration object containing database
              settings, output directory path, data limits, progress display options,
              and extraction flags

    Returns:
        int: Exit code from the dump operation
             (typically 0 for success, non-zero for failure)
    """

    return int(
        not dump_to_networkx_file(
            db_config=args.db_config,
            out_dir_path=args.nx_graph_dir,
            limit=args.limit,
            show_progress_bar=args.show_progress_bar,
            extract_wikipedia_articles=args.extract_wikipedia_articles,
            logger=logger,
        )
    )


def load_from_networkx_graphs(args) -> int:
    """Load and work with NetworkX graphs."""
    logger.info("Loading data from NetworkX graphs")
    try:
        # Check if the graphs directory exists
        graphs_path = Path(args.nx_graph_dir)
        if not graphs_path.exists():
            logger.error(
                f"NetworkX graphs directory does not exist: {args.nx_graph_dir}"
            )
            logger.info(f"Please create graphs first using --convert-to-nx-graph")
            return 1

        # Check if required graph files exist
        required_files = [
            "lexical_units.pickle",
            "synsets.pickle",
            "units_and_synsets.pickle",
        ]
        missing_files = []
        for file_name in required_files:
            file_path = graphs_path / file_name
            if not file_path.exists():
                missing_files.append(file_name)

        if missing_files:
            logger.error(f"Missing required graph files: {missing_files}")
            logger.info(f"Please create graphs first using --convert-to-nx-graph")
            return 1

        # Use NetworkX connector
        connector = PlWordnetAPINxConnector(nx_graph_dir=args.nx_graph_dir)

        with PolishWordnet(
            connector=connector,
            extract_wiki_articles=args.extract_wikipedia_articles,
            use_memory_cache=True,
            show_progress_bar=args.show_progress_bar,
        ) as pl_wn:
            # Example usage - get some data to verify it works
            logger.info("Testing NetworkX connector...")

            rel_types = pl_wn.get_relation_types(limit=10)
            if rel_types:
                logger.info(f"Successfully loaded {len(rel_types)} relation types")
                logger.info("First few relation types:")
                for rt in rel_types[:3]:
                    logger.info(f"\t-> {rt}")
            else:
                logger.warning("No relation types found")

            lexical_units = pl_wn.get_lexical_units(limit=5)
            if lexical_units:
                logger.info(
                    f"Successfully loaded {len(lexical_units)} lexical units"
                )
                logger.info("First few lexical units:")
                for lu in lexical_units[:3]:
                    logger.info(f"\t-> {lu}")
            else:
                logger.warning("No lexical units found")

        logger.info("NetworkX graph loading completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Error loading NetworkX graphs: {e}")
        return 1


def load_from_database(args) -> int:
    """
    Load and work with the MySQL database directly.
    """
    logger.info("Loading data from MySQL database")
    try:
        # Use database connector
        connector = PlWordnetAPIMySQLDbConnector(db_config_path=args.db_config)

        with PolishWordnet(
            connector=connector,
            extract_wiki_articles=args.extract_wikipedia_articles,
            use_memory_cache=True,
            show_progress_bar=args.show_progress_bar,
        ) as pl_wn:
            logger.info("Testing MySQL database connector...")

            rel_types = pl_wn.get_relation_types(limit=10)
            if rel_types:
                logger.info(
                    f"Successfully loaded {len(rel_types)} relation types from database"
                )
                logger.info("First few relation types:")
                for rt in rel_types[:3]:
                    logger.info(f"\t-> {rt}")
            else:
                logger.warning("No relation types found in database")

        logger.info("MySQL database loading completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Error loading from MySQL database: {e}")
        return 1


def prepare_logger(args) -> None:
    logging.getLogger().setLevel(getattr(logging, args.log_level))


def main(argv=None):
    args = prepare_parser().parse_args(argv)
    prepare_logger(args=args)

    logger.info("Starting plwordnet-cli")
    logger.info(f"Arguments: {vars(args)}")

    # Priority: convert_to_nx > use_database > default (load from nx graphs)
    if args.convert_to_nx:
        return _dump_to_networkx_file(args=args)
    elif args.use_database:
        return load_from_database(args=args)
    else:
        # Default behavior: load from NetworkX graphs
        return load_from_networkx_graphs(args=args)


if __name__ == "__main__":
    sys.exit(main())


#
# from plwordnet_handler.utils.resource_paths import ResourcePaths
#
# def check_available_graphs():
#     """
#     Check and display information about available graphs.
#     """
#     available = ResourcePaths.get_available_graphs()
#
#     logger.info("Checking available graphs...")
#     for graph_type, path in available.items():
#         if path:
#             logger.info(f"✓ {graph_type.title()} graph available: {path}")
#         else:
#             logger.info(f"✗ {graph_type.title()} graph not found")
#
#     if not ResourcePaths.has_installed_graphs():
#         logger.warning("No graphs installed. Install with:")
#         logger.warning("pip install . --install-option='--data-graph-test'")
#         logger.warning("or")
#         logger.warning("pip install . --install-option='--data-graph-full'")
#         return False
#
#     return True
# Wykorzystanie:
# def main(argv=None):
#     args = prepare_parser().parse_args(argv)
#     prepare_logger(args=args)
#
#     logger.info("Starting plwordnet-cli")
#     logger.info(f"Arguments: {vars(args)}")
#     if not check_available_graphs():
#         logger.error("No graphs available. Exiting.")
#         return 1
#     # ...
