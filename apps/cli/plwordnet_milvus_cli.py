from plwordnet_handler.utils.logger import prepare_logger

from plwordnet_trainer.cli.argparser import prepare_parser
from plwordnet_trainer.cli.wrappers import CLIMilvusWrappers, Constants


def main():
    args = prepare_parser().parse_args()

    logger = prepare_logger(
        logger_name=__name__,
        logger_file_name=Constants.LOG_FILENAME,
        log_level=args.log_level,
        use_default_config=True,
    )

    cli_wrapper = CLIMilvusWrappers(
        args=args,
        verify_args=True,
        log_level=args.log_level,
        log_filename=Constants.LOG_FILENAME,
    )
    # try:
    #     cli_wrapper = CLIMilvusWrappers(
    #         args=args,
    #         verify_args=True,
    #         log_level=args.log_level,
    #         log_filename=Constants.LOG_FILENAME,
    #     )
    # except Exception as ex:
    #     logger.error(ex)
    #     return 1

    # The highest priority has schema initialization
    if args.prepare_database:
        if not cli_wrapper.prepare_database():
            logger.error("Error while preparing plwordnet Milvus database")
            return 1

    # If --prepare-base-embedding is given
    if args.prepare_base_embeddings:
        # plwn api is required
        if cli_wrapper.prepare_wordnet_based_on_args(use_memory_cache=True) is None:
            return 1
        cli_wrapper.prepare_base_embeddings(batch_size=1000)

    return 0
