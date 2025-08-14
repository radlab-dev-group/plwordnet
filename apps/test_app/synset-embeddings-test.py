from plwordnet_handler.utils.logger import prepare_logger

from plwordnet_handler.base.connectors.milvus.core.schema import (
    MilvusWordNetSchemaHandler,
)
from plwordnet_handler.base.connectors.milvus.core.io import (
    MilvusWordNetInsertHandler,
)

from plwordnet_handler.base.structure.polishwordnet import PolishWordnet
from plwordnet_trainer.embedder.generator.strategy import EmbeddingBuildStrategy
from plwordnet_trainer.embedder.generator.bi_encoder import (
    BiEncoderEmbeddingGenerator,
)
from plwordnet_trainer.embedder.generator.lexical_unit import (
    LexicalUnitEmbeddingGenerator,
)
from plwordnet_trainer.embedder.milvus.consumer import EmbeddingMilvusConsumer


DEVICE = "cuda:0"
LOG_LEVEL = "INFO"
CREATE_SCHEMA = True

MILVUS_CONFIG = (
    "/mnt/data2/dev/develop/radlab-plwordnet/resources/milvus-config-pk.json"
)
NX_GRAPHS = (
    "/mnt/data2/data/resources/plwordnet_handler/20250811/slowosiec_full/nx/graphs"
)

# Bi-encoder model spec
BI_ENCODER_MODEL_NAME = ""
BI_ENCODER_WANDB = (
    "http://192.168.100.61:8080/pkedzia/plWordnet-semantic-embeddings/runs/o7reqebo"
)
BI_ENCODER_MODEL_PATH = "/mnt/data2/llms/models/radlab-open/embedders/plwn-semantic-embeddingss/v0.1/EuroBERT-610m/biencoder/20250806_162431_full_dataset_ratio-2.0_train0.9_eval0.1/checkpoint-290268"

SPACY_MODEL_NAME = "pl_core_news_sm"
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
    logger.info(handler.get_status())
    handler.disconnect()
else:
    embedding_consumer = EmbeddingMilvusConsumer(
        milvus=MilvusWordNetInsertHandler.from_config_file(
            config_path=MILVUS_CONFIG,
        )
    )

    with PolishWordnet(nx_graph_dir=NX_GRAPHS, use_memory_cache=True) as pl_wn:
        logger.info("PolishWordnet connected")

        syn_emb_generator = LexicalUnitEmbeddingGenerator(
            generator=BiEncoderEmbeddingGenerator(
                model_path=BI_ENCODER_MODEL_PATH,
                model_name=BI_ENCODER_MODEL_NAME,
                device=DEVICE,
                normalize_embeddings=True,
                log_level=LOG_LEVEL,
                log_filename=LOG_FILENAME,
            ),
            pl_wordnet=pl_wn,
            log_level=LOG_LEVEL,
            log_filename=LOG_FILENAME,
            spacy_model_name=SPACY_MODEL_NAME,
            strategy=EmbeddingBuildStrategy.MEAN,
            max_workers=4,
        )
        logger.info("SynsetEmbeddingGenerator created")

        for e_dict in syn_emb_generator.generate(split_to_sentences=True):
            embedding_consumer.add_embedding(embedding_dict=e_dict)
