import argparse

from plwordnet_handler.cli.base_argparser import prepare_base_parser
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
    parser = prepare_base_parser(
        description="Polish Wordnet handler",
        example_usage=EXAMPLE_USAGE,
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

    parser.add_argument(
        "--correct-texts",
        dest="correct_texts",
        action="store_true",
        default=False,
        help="Optional flag to correct text (wikipedia content)",
    )
    parser.add_argument(
        "--prompts-dir",
        dest="prompts_dir",
        type=str,
        default=None,
        help="Path to directory containing prompts files (*.prompt). "
        "This option is required when --correct-texts",
    )
    parser.add_argument(
        "--prompt-name-correct-text",
        dest="prompt_name_correct_text",
        type=str,
        default=None,
        help="Prompt name used to prepare correct version of text. "
        "This option is required when --correct-texts",
    )
    parser.add_argument(
        "--openapi-configs-dir",
        dest="openapi_configs_dir",
        type=str,
        default=None,
        help="Path to config with generative models available with OpenAPI. "
        "This option is required when --correct-texts",
    )

    parser.add_argument(
        "--show-progress-bar",
        dest="show_progress_bar",
        action="store_true",
        help="Show progress bar",
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
