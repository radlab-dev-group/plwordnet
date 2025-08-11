import sys
import logging

from plwordnet_handler.base.connectors.nx.nx_connector import PlWordnetAPINxConnector
from plwordnet_trainer.embedder.synset.generator import (
    EmbeddingGenerator,
    SynsetEmbeddingGenerator,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("../plwordnet_cli.log"),
    ],
)

logger = logging.getLogger(__name__)


generator = EmbeddingGenerator(
    model_path="/mnt/data2/llms/models/radlab-open/embedders/radlab_polish-bi-encoder-mean",
    device="cpu",
)

connector = PlWordnetAPINxConnector(
    nx_graph_dir="/home/pkedzia/.local/lib/python3.10/site-packages/plwordnet_handler/resources/graphs/plwordnet_test/nx/graphs",
    autoconnect=True,
)

syn_emb_generator = SynsetEmbeddingGenerator(
    generator=generator, connector=connector
)

syn_emb_generator.run()
