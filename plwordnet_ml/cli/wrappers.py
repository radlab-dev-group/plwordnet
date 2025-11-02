import os.path
from typing import Optional

from plwordnet_handler.base.connectors.milvus.search_handler import (
    MilvusWordNetSearchHandler,
)
from plwordnet_handler.cli.base_wrapper import CLIWrapperBase
from plwordnet_handler.dataset.exporter.relgat import RelGATExporter
from plwordnet_ml.dataset.aligned_id.aligned_dataset_id import (
    RelGATDatasetIdentifiersAligner,
)
from plwordnet_ml.embedder.model_config import BiEncoderModelConfig
from plwordnet_ml.embedder.bi_encoder import BiEncoderEmbeddingGenerator
from plwordnet_ml.embedder.milvus.consumer import EmbeddingMilvusConsumer
from plwordnet_ml.embedder.generator.strategy import EmbeddingBuildStrategy
from plwordnet_handler.base.connectors.milvus.core.config import MilvusConfig
from plwordnet_ml.embedder.generator.base_embeddings.lexical_unit_empty import (
    SemanticEmbeddingGeneratorEmptyLu,
)
from plwordnet_ml.embedder.generator.base_embeddings.synset import (
    SemanticEmbeddingGeneratorSynset,
)
from plwordnet_handler.base.connectors.milvus.initializer import (
    MilvusWordNetSchemaInitializer,
)
from plwordnet_handler.base.connectors.milvus.insert_handler import (
    MilvusWordNetInsertHandler,
)

from plwordnet_ml.embedder.generator.base_embeddings.lexical_unit import (
    SemanticEmbeddingGeneratorLuAndExamples,
)


class Constants:
    SPACY_MODEL_NAME = "pl_core_news_sm"
    LOG_FILENAME = "synset-embeddings.log"
    # Part of speech:
    #   1 -- pl nouns
    #   2 -- pl verbs
    #   3 -- pl adjectives
    #   4 -- pl adverbs
    #   5 -- en nouns
    #   6 -- en verbs
    #   7 -- en adjectives
    #   8 -- en adverbs
    ACCEPT_POS = [1, 2, 3, 4, 5, 6, 7, 8]


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

        self.bi_encoder_model_config = BiEncoderModelConfig.from_json_file(
            config_path=args.embedder_config
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

        # if --prepare-base-embeddings-lu
        if args.prepare_base_embeddings_lu:
            opts += 1

        # if --prepare-base-mean-empty-embeddings-lu
        if args.prepare_mean_empty_base_embeddings_lu:
            if not args.embedder_config:
                raise TypeError(
                    "Embedder configuration is required for "
                    "--prepare-base-mean-empty-embeddings-lu"
                )

            if not os.path.exists(args.embedder_config):
                raise TypeError(
                    f"Embedder configuration not found: {args.embedder_config}\n"
                    "Embedder config is required for "
                    "--prepare-base-mean-empty-embeddings-lu"
                )

            opts += 1

        # if --prepare-base-embeddings-synset
        if args.prepare_base_embeddings_synset:
            opts += 1

        # if --export-relgat-mapping
        if args.export_relgat_mapping:
            if not args.relgat_mapping_directory:
                raise TypeError(
                    "--relgat-mapping-directory option is required for "
                    "--export-relgat-mapping"
                )

            opts += 1

        # if --export-relgat-dataset
        if args.export_relgat_dataset:
            if not args.milvus_config:
                raise TypeError(
                    "--milvus-config option is required for "
                    "--export-relgat-dataset"
                )

            if not args.relgat_mapping_directory:
                raise TypeError(
                    "--relgat-mapping-directory option is required for "
                    "--export-relgat-dataset"
                )

            if not args.relgat_dataset_directory:
                raise TypeError(
                    "--relgat-dataset-directory option is required for "
                    "--export-relgat-dataset"
                )

            opts += 1

        if opts == 0:
            raise TypeError(
                "\n\n"
                "No one option is given, please choose:\n"
                "  --prepare-database - to prepare database,\n"
                "  --prepare-base-embeddings-lu - to prepare base "
                "embeddings for lexical units, based on the examples,\n"
                "  --prepare-base-mean-empty-embeddings-lu - insert mean-embeddings "
                "for empty lu (without examples/base embedding)\n"
                "  --prepare-base-embeddings-synset - to prepare base "
                "synset embeddings using synonymy relation and weighted mean.\n"
                " --export-relgat-mapping - to export RelGAT mappings\n"
                " --export-relgat-dataset - to export RelGAT dataset\n"
                "  (...) \n"
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
        # --prepare-base-embeddings-lu
        if self.args.prepare_base_embeddings_lu:
            return True

        # --prepare-base-mean-empty-embeddings-lu
        if self.args.prepare_mean_empty_base_embeddings_lu:
            return True

        # --prepare-base-embeddings-synset
        if self.args.prepare_base_embeddings_synset:
            return True

        # --export-relgat-mapping
        if self.args.export_relgat_mapping:
            return True

        # --export-relgat-dataset
        if self.args.export_relgat_dataset:
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
        handler.initialize()

        try:
            handler.connect()
            self.logger.info(handler.get_status())
            handler.disconnect()
            return True
        except Exception as e:
            self.logger.error(e)
            return False

    def prepare_base_embeddings_lu(self, batch_size: int = 1000):
        """
        Generate and insert lexical units base semantic embeddings
        into the Milvus database.

        Creates an embedding consumer and embedding_generator to process
        WordNet data and generate embeddings using a bi-encoder model. Processes
        embeddings for accepted part-of-speech categories and inserts them into
        the database in batches.

        Args:
            batch_size: Number of embeddings to process in each batch.
            Defaults to 1000

        Raises:
            RuntimeError: If PlWordnet API or Milvus config is not initialized
        """
        self.__plwn_api_and_milvus_ready()

        self.logger.info("Preparing embedding consumer")
        embedding_consumer = EmbeddingMilvusConsumer(
            milvus=MilvusWordNetInsertHandler(config=self.milvus_config),
            batch_size=batch_size,
        )

        self.logger.info("Preparing base LU/LU examples embeddings generator")
        syn_emb_generator = SemanticEmbeddingGeneratorLuAndExamples(
            generator=BiEncoderEmbeddingGenerator(
                model_path=self.bi_encoder_model_config.model_path,
                model_name=self.bi_encoder_model_config.model_name,
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
            accept_pos=Constants.ACCEPT_POS,
        )

        self.logger.info("Retrieving embeddings for lu examples from Milvus DB...")
        _done_texts = []
        for lu_example in embedding_consumer.milvus.get_all_lu_examples():
            _done_texts.append(lu_example["example"])
        syn_emb_generator._added_texts = _done_texts
        self.logger.info(
            f"Found {len(_done_texts)} existing lu examples. "
            f"These examples will be skipped and not be added twice"
        )

        self.logger.info("Starting base-embeddings generation")
        for embeddings in syn_emb_generator.generate(split_to_sentences=True):
            for emb_dict in embeddings:
                embedding_consumer.add_embedding(
                    embedding_dict=emb_dict,
                    model_name=self.bi_encoder_model_config.model_name,
                )

        # Insert missing (from not full batch)
        embedding_consumer.flush()

    def prepare_mean_empty_base_embeddings_lu(self, batch_size: int = 1000):
        """
        Generate and insert synthetic (fake) embeddings for lexical units
        without base embeddings.

        Creates mean embeddings for lexical units that lack base embeddings
        by computing averages from other lexical units within the same synset.
        Uses a specialized generator and consumer to process and insert
        the synthetic embeddings into Milvus.

        Args:
            batch_size: Number of embeddings to process in each batch.
            Defaults to 1000
        """
        self.__plwn_api_and_milvus_ready()

        self.logger.info("Preparing empty mean-embeddings generator for LU")
        empty_lu_generator = SemanticEmbeddingGeneratorEmptyLu(
            milvus_config=self.milvus_config,
            pl_wordnet=self.pl_wn,
            strategy=EmbeddingBuildStrategy.MEAN,
            log_level=self.args.log_level,
            log_filename=Constants.LOG_FILENAME,
            accept_pos=Constants.ACCEPT_POS,
        )

        self.logger.info("Preparing embedding consumer for fake LU")
        embedding_consumer = EmbeddingMilvusConsumer(
            milvus=MilvusWordNetInsertHandler(config=self.milvus_config),
            batch_size=batch_size,
        )

        self.logger.info("Starting empty-embeddings generation")
        for emb_dict in empty_lu_generator.generate():
            embedding_consumer.add_embedding(
                embedding_dict=emb_dict,
                model_name=self.bi_encoder_model_config.model_name,
            )
        # Insert missing (from not full batch)
        embedding_consumer.flush()

    def prepare_base_embeddings_synsets(self, batch_size: int = 1000):
        """
        Generate and insert base embeddings for synsets using a weighted
        mean strategy based on the LU belonging to the synset.

        Creates synset embeddings by aggregating embeddings from constituent lexical
        units using a weighted mean strategy.

        Args:
            batch_size: Number of embeddings to process in each batch.
            Defaults to 1000
        """
        self.__plwn_api_and_milvus_ready()

        self.logger.info("Preparing embedding consumer")
        embedding_consumer = EmbeddingMilvusConsumer(
            milvus=MilvusWordNetInsertHandler(config=self.milvus_config),
            batch_size=batch_size,
        )

        self.logger.info("Preparing base LU/LU examples embeddings generator")
        syn_emb_generator = SemanticEmbeddingGeneratorSynset(
            milvus_config=self.milvus_config,
            pl_wordnet=self.pl_wn,
            strategy=EmbeddingBuildStrategy.MEAN_WEIGHTED,
            log_level=self.args.log_level,
            log_filename=Constants.LOG_FILENAME,
            accept_pos=Constants.ACCEPT_POS,
        )

        self.logger.info("Starting base-embeddings generation")
        for emb_dict in syn_emb_generator.generate(split_to_sentences=True):
            embedding_consumer.add_embedding(
                embedding_dict=emb_dict,
                model_name=self.bi_encoder_model_config.model_name,
            )
        # Insert missing (from not full batch)
        embedding_consumer.flush()

    def export_relgat_mapping_to_directory(self) -> bool:
        self.__plwn_api_and_milvus_ready()

        self.logger.info(
            f"Exporting RELGat mappings to directory "
            f"{self.args.relgat_mapping_directory}"
        )
        try:
            relgat_mapper = RelGATDatasetIdentifiersAligner(
                plwn_api=self.pl_wn,
                prepare_mapping=True,
                log_level=self.args.log_level,
                logger_file_name=Constants.LOG_FILENAME,
                mapping_path=None,
            )
            relgat_mapper.export_to_dir(
                out_directory=self.args.relgat_mapping_directory,
            )
        except Exception as e:
            self.logger.error(e)
            return False
        return True

    def export_relgat_dataset_to_directory(self) -> bool:
        self.__plwn_api_and_milvus_ready()

        self.logger.info(
            f"Exporting RELGat dataset to directory "
            f"{self.args.relgat_mapping_directory}"
        )
        try:
            relgat_exporter = RelGATExporter(
                plwn_api=self.pl_wn,
                milvus_handler=MilvusWordNetSearchHandler(
                    config=self.milvus_config,
                    log_level=self.log_level,
                    logger_name=Constants.LOG_FILENAME,
                    auto_connect=True,
                ),
                out_directory=self.args.relgat_dataset_directory,
                aligner=RelGATDatasetIdentifiersAligner(
                    plwn_api=None,
                    prepare_mapping=False,
                    log_level=self.args.log_level,
                    logger_file_name=Constants.LOG_FILENAME,
                    mapping_path=self.args.relgat_mapping_directory,
                ),
                accept_pos=Constants.ACCEPT_POS,
                limit=self.args.limit,
            )
            relgat_exporter.export_to_dir()
        except Exception as e:
            self.logger.error(e)
            # return False
            raise e

        self.logger.info(
            f"Successfully exported to {self.args.relgat_dataset_directory}"
        )
        return True

    def __plwn_api_and_milvus_ready(self):
        """
        Validate that required APIs and configurations are properly initialized.

        Ensures that both the Polish WordNet API connection and Milvus database
        configuration are available before proceeding with embedding operations.

        Raises:
            RuntimeError: If PlWordNet API or Milvus config is not initialized
        """
        if self.pl_wn is None:
            raise RuntimeError("PlWordNet API is not initialized")

        if self.milvus_config is None:
            raise RuntimeError("Milvus config is not initialized!")
