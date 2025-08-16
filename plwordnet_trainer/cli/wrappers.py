from typing import Optional

from plwordnet_handler.cli.base_wrapper import CLIWrapperBase

from plwordnet_handler.utils.logger import prepare_logger
from plwordnet_handler.base.structure.polishwordnet import PolishWordnet
from plwordnet_handler.base.connectors.milvus.config import MilvusConfig
from plwordnet_trainer.embedder.milvus.consumer import EmbeddingMilvusConsumer
from plwordnet_trainer.embedder.generator.strategy import EmbeddingBuildStrategy
from plwordnet_handler.base.connectors.milvus.initializer import (
    MilvusWordNetSchemaInitializer,
)
from plwordnet_handler.base.connectors.milvus.insert_handler import (
    MilvusWordNetInsertHandler,
)

from plwordnet_trainer.embedder.generator.bi_encoder import (
    BiEncoderEmbeddingGenerator,
)
from plwordnet_trainer.embedder.generator.lexical_unit import (
    SemanticEmbeddingGenerator,
)


class Constants:
    SPACY_MODEL_NAME = "pl_core_news_sm"
    LOG_FILENAME = "synset-embeddings.log"


class BiEncoderModelConfig:
    """
    Bi-encoder model spec
    """

    MODEL_NAME = "Semantic-v0.1-o7reqebo"
    WANDB = (
        "http://192.168.100.61:8080/pkedzia/"
        "plWordnet-semantic-embeddings/runs/o7reqebo"
    )
    MODEL_PATH = (
        "/mnt/data2/llms/models/radlab-open/embedders/plwn-semantic-embeddingss/"
        "v0.1/EuroBERT-610m/biencoder/"
        "20250806_162431_full_dataset_ratio-2.0_train0.9_eval0.1/checkpoint-290268"
    )


class CLIMilvusWrappers(CLIWrapperBase):
    """
    Any plwordnet-milvus command wrapper.
    """

    def __init__(
        self,
        args,
        verify_args: bool,
        log_level: str = "INFO",
        log_filename: Optional[str] = Constants.LOG_FILENAME,
    ) -> None:
        """
        Initializes the Milvus CLI wrapper with command-line arguments
        and optional verification. Initialize configuration to Milvus.

        Args:
            args: Parsed command-line arguments containing configuration options
            verify_args (bool): Whether to validate the provided arguments
            log_level (str): Logger level (INFO default)
            log_filename (str): Name of the log file (as default is set
            to Constants.LOG_FILENAME)

        Raises:
            FileNotFoundError - when the Milvus configuration file does not exist
            KeyError - when the Milvus configuration file is invalid
        """
        super().__init__(
            args=args,
            verify_args=verify_args,
            log_name=__name__,
            log_level=log_level,
            log_filename=log_filename,
        )

        self.milvus_config = MilvusConfig.from_json_file(
            config_path=args.milvus_config
        )

    def are_args_correct(self, args=None):
        """
        Validates the correctness of command-line arguments.

        Args:
            args: Optional arguments to validate (uses self.args if None)

        Raises:
            TypeError: If any errors occur
        """
        if args is None:
            args = self.args
            if args is None:
                raise TypeError("No arguments to check are provided")

        opts = 0
        # If --prepare-database
        if args.prepare_database:
            opts += 1

        # if --prepare-base-embeddings
        if args.prepare_base_embeddings:
            opts += 1

        if opts == 0:
            raise TypeError(
                "No one option is given, please choose: \n"
                "  --prepare-database - to prepare database,\n"
                "  --prepare-base-embeddings - to prepare base embeddings,\n"
                "  --help to show all available options"
            )

        return True

    def is_api_required(self) -> bool:
        """
        Check if API connection is required based on current arguments.

        Determines whether an API connection to the WordNet database is needed
        by evaluating the current command-line arguments and operations requested.

        Returns:
            bool: True if API connection is required, False otherwise
        """

        if self.args.prepare_base_embeddings:
            return True
        return False

    def prepare_database(self):
        """
        Initialize and prepare the Milvus database for WordNet operations.

        Creates a Milvus schema initializer with the configured connection settings,
        performs complete database initialization including collections and indexes,
        then tests the connection status before disconnecting.

        Raises:
            Exception: If database initialization fails
        """

        handler = MilvusWordNetSchemaInitializer(config=self.milvus_config)
        _initialized = handler.initialize()
        if not _initialized:
            raise False

        handler.connect()
        self.logger.info(handler.get_status())
        handler.disconnect()

    def prepare_base_embeddings(self, batch_size: int = 1000):
        """
        Generate and insert base semantic embeddings into the Milvus database.

        Creates an embedding consumer and semantic embedding generator to process
        WordNet data and generate embeddings using a bi-encoder model. Processes
        embeddings for accepted part-of-speech categories and inserts them into
        the database in batches.

        Args:
            batch_size: Number of embeddings to process in each batch.
            Defaults to 1000
        """

        embedding_consumer = EmbeddingMilvusConsumer(
            milvus=MilvusWordNetInsertHandler(config=self.milvus_config),
            batch_size=batch_size,
        )

        self.logger.info("PolishWordnet connected")
        syn_emb_generator = SemanticEmbeddingGenerator(
            generator=BiEncoderEmbeddingGenerator(
                model_path=BiEncoderModelConfig.MODEL_PATH,
                model_name=BiEncoderModelConfig.MODEL_NAME,
                device=self.args.device,
                normalize_embeddings=True,
                log_level=self.args.log_level,
                log_filename=Constants.LOG_FILENAME,
            ),
            pl_wordnet=self.pl_wn,
            log_level=self.args.log_level,
            log_filename=Constants.LOG_FILENAME,
            spacy_model_name=Constants.SPACY_MODEL_NAME,
            strategy=EmbeddingBuildStrategy.MEAN,
            max_workers=1,
            accept_pos=[1, 2, 3, 4],
        )

        self.logger.info("SynsetEmbeddingGenerator created")
        for embeddings in syn_emb_generator.generate(split_to_sentences=True):
            for emb_dict in embeddings:
                embedding_consumer.add_embedding(
                    embedding_dict=emb_dict,
                    model_name=BiEncoderModelConfig.MODEL_NAME,
                )
        embedding_consumer.flush()
