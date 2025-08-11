import sys
import logging

from plwordnet_handler.base.structure.polishwordnet import PolishWordnet
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
    device="cuda:0",
)

connector = PlWordnetAPINxConnector(
    nx_graph_dir="/mnt/data2/data/resources/plwordnet_handler/20250811/slowosiec_test/nx/graphs",
    autoconnect=True,
)
pl_wn = PolishWordnet(
                connector=connector,
                db_config_path=None,
                nx_graph_dir=None,
                extract_wiki_articles=False,
                use_memory_cache=True,
                show_progress_bar=False,
            )

syn_emb_generator = SynsetEmbeddingGenerator(
    generator=generator, pl_wordnet=pl_wn
)

syn_emb_generator.run(batch_size=1024)
