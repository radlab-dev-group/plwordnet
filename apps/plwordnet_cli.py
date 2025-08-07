import sys

from plwordnet_handler.cli.wrappers import CLIWrappers
from plwordnet_handler.cli.argparser import prepare_parser

from plwordnet_handler.utils.logger import prepare_logger


def main(argv=None):
    args = prepare_parser().parse_args(argv)

    logger = prepare_logger(
        logger_name="plwordnet_cli",
        logger_file_name="plwordnet_cli.log",
        use_default_config=True,
        log_level=args.log_level,
    )

    try:
        cli_wrapper = CLIWrappers(args, verify_args=True, logger=logger)
    except Exception as ex:
        logger.error(ex)
        sys.exit(1)

    logger.info("Starting plwordnet-cli")
    logger.info(f"Arguments: {vars(args)}")

    if args.convert_to_nx:
        return cli_wrapper.dump_to_networkx_file()
    else:
        # Prepare connector
        if args.use_database:
            # Use database connector when --use-database
            connector = cli_wrapper.connect_to_database()
        else:
            # If the option `--use-database` is not given,
            # then try to load NetworkX graph
            connector = cli_wrapper.connect_to_networkx_graphs()

    if connector is None:
        logger.error("Could not connect to plwordnet-cli")
        return 1

    # Prepare wordnet with connector
    wordnet = cli_wrapper.prepare_wordnet_with_connector(
        connector=connector, use_memory_cache=True
    )
    if wordnet is None:
        return 1

    # Test api if --test-api passed
    if args.test_api:
        cli_wrapper.test_plwordnet()

    if args.dump_relation_types_to_file:
        cli_wrapper.dump_relation_types_to_file()

    return 0


if __name__ == "__main__":
    sys.exit(main())
