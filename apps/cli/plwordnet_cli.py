import sys

from plwordnet_handler.cli.wrappers import CLIWrappers
from plwordnet_handler.cli.argparser import prepare_parser

from plwordnet_handler.utils.logger import prepare_logger


class Constants:
    logger_file_name = "plwordnet_cli.log"


def main(argv=None):
    args = prepare_parser().parse_args(argv)

    logger = prepare_logger(
        logger_name="plwordnet_cli",
        logger_file_name=Constants.logger_file_name,
        use_default_config=True,
        log_level=args.log_level,
    )

    cli_wrapper = CLIWrappers(
        args,
        verify_args=True,
        log_level=args.log_level,
        log_filename=Constants.logger_file_name,
    )

    logger.info("Starting plwordnet-cli")
    logger.debug(f"Arguments: {vars(args)}")

    if args.convert_to_nx:
        return cli_wrapper.dump_to_networkx_file()

    if cli_wrapper.prepare_wordnet_based_on_args(use_memory_cache=True) is None:
        # When api interface is not able to initialize
        return 1

    # Test api if --test-api passed
    if args.test_api:
        if not cli_wrapper.test_plwordnet():
            logger.error("Error while testing plwordnet")
            return 1

    # Dump rels to file if --dump-relation-types-to-file is given
    if args.dump_relation_types_to_file:
        if not cli_wrapper.dump_relation_types_to_file():
            logger.error("Could not dump relation types to file!")
            return 1

    # Dump embedder dataset if --dump-embedder-dataset-to-file is given
    if args.dump_embedder_dataset_to_file:
        if not cli_wrapper.dump_embedder_dataset_to_file(cut_weight=0.14):
            logger.error("Could not dump embedder dataset!")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
