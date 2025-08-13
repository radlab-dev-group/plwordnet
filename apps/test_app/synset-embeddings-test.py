from plwordnet_handler.base.structure.polishwordnet import PolishWordnet
from plwordnet_handler.utils.logger import prepare_logger
from plwordnet_handler.base.connectors.milvus.core.schema import (
    MilvusWordNetSchemaHandler,
)
from plwordnet_trainer.embedder.generator.lexical_unit import (
    BiEncoderEmbeddingGenerator,
    LexicalUnitEmbeddingGenerator,
)
from plwordnet_handler.base.connectors.milvus.milvus_connector import MilvusConnector
from plwordnet_trainer.embedder.generator.strategy import EmbeddingBuildStrategy

MILVUS_CONFIG = (
    "/mnt/data2/dev/develop/radlab-plwordnet/resources/milvus-config-pk.json"
)
NX_GRAPHS = (
    "/mnt/data2/data/resources/plwordnet_handler/20250811/slowosiec_test/nx/graphs"
)

CREATE_SCHEMA = False

LOG_LEVEL = "INFO"
LOG_FILENAME = "synset-embeddings.log"

logger = prepare_logger(
    logger_name=__name__,
    logger_file_name=LOG_FILENAME,
    log_level=LOG_LEVEL,
    use_default_config=True,
)


if CREATE_SCHEMA:
    handler = MilvusWordNetSchemaHandler.from_config_file(config_path=MILVUS_CONFIG)
    handler.initialize()

    handler.connect()
    print(handler.get_status())
    handler.disconnect()
else:
    print("???????????????")
    milvus = MilvusConnector.from_config_file(
        config_path=MILVUS_CONFIG,
    )

    with PolishWordnet(nx_graph_dir=NX_GRAPHS, use_memory_cache=True) as pl_wn:
        logger.info("PolishWordnet connected")

        syn_emb_generator = LexicalUnitEmbeddingGenerator(
            generator=BiEncoderEmbeddingGenerator(
                model_path="/mnt/data2/llms/models/radlab-open/embedders/radlab_polish-bi-encoder-mean",
                device="cpu",
                normalize_embeddings=True,
                log_level=LOG_LEVEL,
                log_filename=LOG_FILENAME,
            ),
            pl_wordnet=pl_wn,
            log_level=LOG_LEVEL,
            log_filename=LOG_FILENAME,
            spacy_model_name="pl_core_news_sm",
            strategy=EmbeddingBuildStrategy.MEAN,
        )
        logger.info("SynsetEmbeddingGenerator created")

        for e_dict in syn_emb_generator.generate(split_to_sentences=True):
            pass
