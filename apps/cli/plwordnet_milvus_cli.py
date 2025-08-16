from plwordnet_trainer.cli.argparser import prepare_parser


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


NX_GRAPHS = (
    "/mnt/data2/data/resources/plwordnet_handler/20250811/slowosiec_full/nx/graphs"
)

# Bi-encoder model spec
BI_ENCODER_MODEL_NAME = "Semantic-v0.1-o7reqebo"
BI_ENCODER_WANDB = (
    "http://192.168.100.61:8080/pkedzia/plWordnet-semantic-embeddings/runs/o7reqebo"
)
BI_ENCODER_MODEL_PATH = "/mnt/data2/llms/models/radlab-open/embedders/plwn-semantic-embeddingss/v0.1/EuroBERT-610m/biencoder/20250806_162431_full_dataset_ratio-2.0_train0.9_eval0.1/checkpoint-290268"


class Constants:
    SPACY_MODEL_NAME = "pl_core_news_sm"
    LOG_FILENAME = "synset-embeddings.log"


def main():
    args = prepare_parser().parse_args()
    logger = prepare_logger(
        logger_name=__name__,
        logger_file_name=Constants.LOG_FILENAME,
        log_level=args.log_level,
        use_default_config=True,
    )

    milvus_config = MilvusConfig.from_json_file(args.milvus_config)

    if args.prepare_database:
        handler = MilvusWordNetSchemaInitializer(config=milvus_config)
        handler.initialize()
        handler.connect()
        logger.info(handler.get_status())
        handler.disconnect()
    elif args.prepare_base_embeddings:
        embedding_consumer = EmbeddingMilvusConsumer(
            milvus=MilvusWordNetInsertHandler(config=milvus_config),
            batch_size=1000,
        )

        with PolishWordnet(nx_graph_dir=NX_GRAPHS, use_memory_cache=True) as pl_wn:
            logger.info("PolishWordnet connected")
            syn_emb_generator = SemanticEmbeddingGenerator(
                generator=BiEncoderEmbeddingGenerator(
                    model_path=BI_ENCODER_MODEL_PATH,
                    model_name=BI_ENCODER_MODEL_NAME,
                    device=args.device,
                    normalize_embeddings=True,
                    log_level=args.log_level,
                    log_filename=Constants.LOG_FILENAME,
                ),
                pl_wordnet=pl_wn,
                log_level=args.log_level,
                log_filename=Constants.LOG_FILENAME,
                spacy_model_name=Constants.SPACY_MODEL_NAME,
                strategy=EmbeddingBuildStrategy.MEAN,
                max_workers=1,
                accept_pos=[1, 2, 3, 4],
            )
            logger.info("SynsetEmbeddingGenerator created")
            for embeddings in syn_emb_generator.generate(split_to_sentences=True):
                for emb_dict in embeddings:
                    embedding_consumer.add_embedding(
                        embedding_dict=emb_dict, model_name=BI_ENCODER_MODEL_NAME
                    )
            embedding_consumer.flush()
