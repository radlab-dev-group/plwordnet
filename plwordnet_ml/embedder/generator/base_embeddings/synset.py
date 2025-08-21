import tqdm
from typing import Iterator, Dict, Any, Optional, List, Tuple

from plwordnet_handler.base.connectors.milvus.core.config import MilvusConfig
from plwordnet_handler.base.structure.polishwordnet import PolishWordnet
from plwordnet_ml.embedder.constants.embedding_types import EmbeddingTypes
from plwordnet_ml.embedder.generator.strategy import EmbeddingBuildStrategy
from plwordnet_handler.base.connectors.milvus.search_handler import (
    MilvusWordNetSearchHandler,
)
from plwordnet_ml.embedder.generator.generator_i import (
    _AnySemanticEmbeddingGeneratorBase,
)


class SemanticEmbeddingGeneratorSynset(_AnySemanticEmbeddingGeneratorBase):
    """
    Generator for creating weighted synset embeddings from lexical unit embeddings.

    This class generates synset-level embeddings by aggregating embeddings from
    constituent lexical units using weighted mean strategies. The weights are
    calculated based on the number of examples available for each lexical unit,
    providing more influence to lexical units with richer example data.
    """

    # Smooth factor in case when no examples for LU exist
    SMOOTH_FACTOR = 1

    def __init__(
        self,
        milvus_config: MilvusConfig,
        pl_wordnet: PolishWordnet,
        strategy: EmbeddingBuildStrategy = EmbeddingBuildStrategy.MEAN_WEIGHTED,
        log_level: str = "INFO",
        log_filename: Optional[str] = None,
        accept_pos: Optional[List[int]] = None,
    ):
        """
        Initialize the synset embedding generator.

        Sets up the generator with Milvus search capabilities and WordNet access
        for creating weighted synset embeddings from constituent lexical units.

        Args:
            milvus_config: Configuration for Milvus database connection
            pl_wordnet: Polish WordNet instance for data access
            strategy: Strategy for combining embeddings. Defaults to MEAN_WEIGHTED
            log_level: Logging level for the generator. Defaults to "INFO"
            log_filename: Optional log file path for file-based logging
            accept_pos: List of accepted part-of-speech categories to process
        """
        super().__init__(
            generator=None,
            strategy=strategy,
            log_level=log_level,
            log_name=__name__,
            log_filename=log_filename,
            accept_pos=accept_pos,
            pl_wordnet=pl_wordnet,
        )

        self.milvus_search_handler = MilvusWordNetSearchHandler(
            config=milvus_config, auto_connect=True
        )

    def generate(
        self, split_to_sentences: Optional[bool] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Generate weighted embeddings for synsets based on constituent lexical units.

        Processes synsets filtered by part-of-speech criteria and generates weighted
        embeddings by aggregating embeddings from lexical units within each synset.
        Shows progress during processing and yields detailed embedding information.

        Args:
            split_to_sentences: Unused parameter for interface compatibility

        Yields:
            Dict[str, Any]: Synset embedding dictionaries containing synset data,
            computed embedding, weights, and metadata
        """

        self.logger.info(f"Preparing base embeddings for Synsets")
        proper_syn_pos = self._get_units_and_synsets_with_pos(
            accepted_pos=self.accept_pos
        )

        with tqdm.tqdm(
            total=len(proper_syn_pos), desc="Preparing synsets embeddings"
        ) as pbar:
            for syn_id, lu_ids in proper_syn_pos.items():
                pbar.update(1)
                yield from self._embedding_based_on_syn_lu_ids(
                    syn_id=syn_id, lu_ids=lu_ids
                )

    def _embedding_based_on_syn_lu_ids(self, syn_id, lu_ids):
        """
        Generate synset embedding based on constituent lexical unit embeddings.

        Retrieves the synset and its associated lexical unit embeddings,
        then computes a weighted embedding by combining lexical unit embeddings
        with weights based on the number of examples. Only processes synsets
        that have at least one lexical unit with base embeddings available.

        Args:
            syn_id: Synset ID to process
            lu_ids: List of lexical unit IDs belonging to the synset

        Yields:
            Dict[str, Any]: Synset embedding dictionary containing the synset
            object, computed embedding, processing metadata, and weight information
        """
        synset = self.pl_wordnet.get_synset(syn_id)
        if synset is not None:
            lu_embs_dict = self.milvus_search_handler.get_lexical_units_embeddings(
                lu_ids=lu_ids, map_to_lexical_units=True
            )
            # If synset have any base lexical unit embeddings
            if len(lu_embs_dict):
                lu_examples_embs_dict = (
                    self.milvus_search_handler.get_lexical_units_examples_embedding(
                        lu_ids=lu_ids, map_to_lexical_units=True
                    )
                )

                embeddings, weights = self.__convert_to_weights(
                    lu_embs_dict,
                    lu_examples_embs_dict,
                    get_first_lu_emb_if_many=True,
                )

                base_embedding = self.embedding_processor.process(
                    embeddings=embeddings,
                    weights=weights,
                    normalize_weights=True,
                    normalize_out_embedding=True,
                )

                yield {
                    "synset": synset,
                    "type": EmbeddingTypes.Base.synset,
                    "strategy": self.embedding_processor.strategy,
                    "embedding": base_embedding,
                }
        else:
            self.logger.error(f"Synset {syn_id} not found in Plwordnet API!")

    def __convert_to_weights(
        self,
        lu_embs_dict,
        lu_examples_embs_dict,
        get_first_lu_emb_if_many: bool = True,
    ) -> Tuple[List, List]:
        """
        Convert lexical unit embeddings to weighted embedding pairs.

        Extracts embeddings from lexical units and calculates corresponding weights
        based on the number of examples available for each lexical unit. Weights
        are computed using a smooth factor plus the count of examples to ensure
        lexical units with more examples have greater influence.

        Args:
            lu_embs_dict: Dictionary mapping lexical unit IDs to their embeddings
            lu_examples_embs_dict: Dictionary mapping lexical unit IDs
            to example embeddings
            get_first_lu_emb_if_many: Whether to use only the first embedding
            when multiple embeddings exist for a lexical unit

        Returns:
            Tuple[List, List]: Tuple containing a list of embeddings
            and corresponding weights
        """
        _e, _w = [], []
        for lu_id, lu_embs in lu_embs_dict.items():
            if get_first_lu_emb_if_many:
                lu_embs = lu_embs[0]["embedding"]
            else:
                lu_embs = [l["embedding"] for l in lu_embs]
            _e.append(lu_embs)
            _w.append(self.SMOOTH_FACTOR + len(lu_examples_embs_dict.get(lu_id, [])))
        return _e, _w
