from plwordnet_handler.utils.logger import prepare_logger

from plwordnet_ml.cli.argparser import prepare_parser
from plwordnet_ml.cli.wrappers import CLIMilvusWrappers, Constants


MILVUS_INSERT_BATCH_SIZE = 1000


def main(argv=None):
    args = prepare_parser().parse_args(argv)

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

    logger.info("Starting plwordnet-milvus")
    logger.debug(f"Arguments: {vars(args)}")

    # The highest priority has schema initialization
    if args.prepare_database:
        if not cli_wrapper.prepare_database():
            logger.error("Error while preparing plwordnet Milvus database")
            return 1

    # Initialize api if is required
    if cli_wrapper.is_api_required():
        if cli_wrapper.prepare_wordnet_based_on_args(use_memory_cache=True) is None:
            logger.error("Error while preparing plwordnet API.")
            return 1

    # If --prepare-base-embedding-lu is given
    if args.prepare_base_embeddings_lu:
        cli_wrapper.prepare_base_embeddings_lu(batch_size=MILVUS_INSERT_BATCH_SIZE)

    # If --prepare-base-mean-empty-embeddings
    if args.prepare_mean_empty_base_embeddings_lu:
        cli_wrapper.prepare_mean_empty_base_embeddings_lu(
            batch_size=MILVUS_INSERT_BATCH_SIZE
        )

    # If --prepare-base-embeddings-synset
    if args.prepare_base_embeddings_synset:
        cli_wrapper.prepare_base_embeddings_synsets(
            batch_size=MILVUS_INSERT_BATCH_SIZE
        )

    # if --export-dataset-to-relgat-to-directory
    if args.export_relgat_dataset_to_directory:
        cli_wrapper.export_relgat_dataset_to_directory()

    return 0
