"""
Application for exporting relation types to Excel file.

python3 rel_types_dump.py \
    --db-config ../resources/plwordnet-mysql-db.json \
    --output ../resources/relation-types-weigths.xlsx
"""

import sys
import logging
import argparse

from pathlib import Path

from plwordnet_handler.dataset.exporter.rel_types import RelationTypesExporter
from plwordnet_handler.base.connectors.db.db_connector import (
    PlWordnetAPIMySQLDbConnector,
)

DEFAULT_DB_CFG_PATH = "resources/plwordnet-mysql-db.json"
DEFAULT_OUTPUT_FILE = "relation_types_export.xlsx"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


def prepare_parser() -> argparse.ArgumentParser:
    """
    Prepare command line argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Export relation types from plWordnet database to Excel file"
    )

    parser.add_argument(
        "--db-config",
        dest="db_config",
        type=str,
        default=DEFAULT_DB_CFG_PATH,
        help=f"Path to JSON file with database configuration "
        f"(default: {DEFAULT_DB_CFG_PATH})",
    )

    parser.add_argument(
        "--output",
        dest="output_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output Excel file path (default: {DEFAULT_OUTPUT_FILE})",
    )

    parser.add_argument(
        "--limit",
        dest="limit",
        type=int,
        required=False,
        help="Limit the number of results",
    )

    parser.add_argument(
        "--log-level",
        dest="log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )

    return parser


def main(argv=None):
    args = prepare_parser().parse_args(argv)

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("Starting relation types export")
    logger.info(f"Arguments: {vars(args)}")

    try:
        db_config_path = Path(args.db_config)
        if not db_config_path.exists():
            logger.error(f"Database configuration file not found: {args.db_config}")
            return 1

        connector = PlWordnetAPIMySQLDbConnector(db_config_path=args.db_config)
        exporter = RelationTypesExporter(connector=connector)
        success = exporter.export_to_xlsx(
            output_file=args.output_file, limit=args.limit
        )

        if success:
            logger.info("Export completed successfully")
            return 0
        else:
            logger.error("Export failed")
            return 1
    except Exception as e:
        logger.error(f"Unexpected error during export: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
