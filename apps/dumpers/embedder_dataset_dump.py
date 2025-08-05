"""
Application for exporting embedder dataset using WordnetToEmbedderConverter.

python3 embedder_dataset_dump.py \
    --xlsx-weights ../resources/relation-types-weights.xlsx \
    --graph-path ../resources/graphs \
    --output ../resources/embedder_dataset.jsonl \
    --limit 1000
"""

import sys
import logging
import argparse

from plwordnet_handler.dataset.exporter.embedder import WordnetToEmbedderConverter

DEFAULT_GRAPH_PATH = "resources/graphs"
DEFAULT_OUTPUT_FILE = "embedder_dataset.jsonl"
DEFAULT_XLSX_WEIGHTS_PATH = "resources/relation-types-weights.xlsx"

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
        description="Export embedder dataset from plWordnet "
        "using WordnetToEmbedderConverter"
    )

    parser.add_argument(
        "--xlsx-weights",
        dest="xlsx_weights",
        type=str,
        default=DEFAULT_XLSX_WEIGHTS_PATH,
        help=f"Path to Excel file with relation weights "
        f"(default: {DEFAULT_XLSX_WEIGHTS_PATH})",
    )

    parser.add_argument(
        "--graph-path",
        dest="graph_path",
        type=str,
        default=DEFAULT_GRAPH_PATH,
        help=f"Path to directory containing NetworkX graph files "
        f"(default: {DEFAULT_GRAPH_PATH})",
    )

    parser.add_argument(
        "--output",
        dest="output_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output JSONL file path (default: {DEFAULT_OUTPUT_FILE})",
    )

    parser.add_argument(
        "--limit",
        dest="limit",
        type=int,
        required=False,
        help="Limit the number of samples to export",
    )

    parser.add_argument(
        "--low-high-ratio",
        type=float,
        dest="low_high_ratio",
        required=False,
        default=2.0,
        help="Ratio between low and high-weighted relations count",
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

    logger.info("Starting embedder dataset export")
    logger.info(f"Arguments: {vars(args)}")

    try:
        converter = WordnetToEmbedderConverter(
            xlsx_path=args.xlsx_weights,
            graph_path=args.graph_path,
            init_converter=True,
        )
        success = converter.export(
            output_file=args.output_file,
            limit=args.limit,
            low_high_ratio=args.low_high_ratio,
        )
        if success:
            logger.info("Dataset export completed successfully")
            return 0
        else:
            logger.error("Dataset export failed")
            return 1
    except Exception as e:
        logger.error(f"Unexpected error during export: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
