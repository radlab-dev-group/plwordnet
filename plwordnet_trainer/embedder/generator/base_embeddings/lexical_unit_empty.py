import tqdm
from typing import Iterator, Dict, Any, Optional, List

from plwordnet_handler.base.connectors.milvus.config import MilvusConfig
from plwordnet_handler.base.structure.polishwordnet import PolishWordnet
from plwordnet_trainer.embedder.constants.embedding_types import EmbeddingTypes
from plwordnet_trainer.embedder.generator.strategy import EmbeddingBuildStrategy
from plwordnet_handler.base.connectors.milvus.search_handler import (
    MilvusWordNetSearchHandler,
)
from plwordnet_trainer.embedder.generator.generator_i import (
    _AnySemanticEmbeddingGeneratorBase,
)


class SemanticEmbeddingGeneratorEmptyLu(_AnySemanticEmbeddingGeneratorBase):
    """
    Generator for creating embeddings for lexical units without base embeddings.

    This class generates synthetic embeddings for lexical units that don't have
    existing embeddings by computing mean embeddings from other lexical units
    within the same synset that do have embeddings available.
    """

    def __init__(
        self,
        milvus_config: MilvusConfig,
        pl_wordnet: PolishWordnet,
        strategy: EmbeddingBuildStrategy = EmbeddingBuildStrategy.MEAN,
        log_level: str = "INFO",
        log_filename: Optional[str] = None,
        accept_pos: Optional[List[int]] = None,
    ):
        """
        Initialize the empty lexical unit embedding generator.

        Sets up the generator with Milvus search capabilities and WordNet access
        for creating embeddings for lexical units that lack base embeddings.

        Args:
            milvus_config: Configuration for Milvus database connection
            pl_wordnet: Polish WordNet instance for data access
            strategy: Strategy for combining multiple embeddings. Defaults to MEAN
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
        Generate embeddings for lexical units without base embeddings.

        Iterates through synsets to identify lexical units lacking embeddings and
        generates synthetic embeddings using mean of existing embeddings from the
        same synset. Provides comprehensive logging of processing statistics.

        Args:
            split_to_sentences: Unused parameter for interface compatibility

        Yields:
            Dict[str, Any]: Embedding dictionaries containing lexical unit data,
            computed embedding vector, type, and strategy information
        """

        lu_no_possible_to_convert = []
        for lu_example_dict in self._synsets_with_any_base_embedding():
            syn_id = lu_example_dict["syn_id"]
            lu_status = lu_example_dict["status"]

            # If any LU from synset has embedding, then skip
            if not len(lu_status["not_found"]):
                continue

            # If not found any base lexical unit embedding
            if not len(lu_status["found"]):
                lu_no_possible_to_convert.append(
                    {"syn_id": syn_id, "not_found": lu_status["not_found"]}
                )
                continue

            yield from self._prepare_embedding_for_empty_lu(lu_status)

        self.logger.info("Finished generating empty LU embeddings")
        self.logger.info(
            f"  - number of synsets without embeddings: "
            f"{len(lu_no_possible_to_convert)}"
        )

        lu_wo_embs = sum(len(i["not_found"]) for i in lu_no_possible_to_convert)
        self.logger.info(
            f"  - number of lexical units without embeddings: {lu_wo_embs}"
        )

    def _synsets_with_any_base_embedding(self):
        """
        Identify synsets and their lexical units' embedding availability status.

        Processes synsets to determine which lexical units have existing embeddings
        and which ones are missing embeddings. Shows progress during processing
        and yields status information for each synset.

        Yields:
            Dict: Dictionary containing synset ID and status with 'found' and
            'not_found' lists of lexical unit IDs
        """

        self.logger.info(f"Preparing embeddings for LU with no base embeddings")
        syn_lus = self._get_synsets_with_lu_ids_pos_filter()

        with tqdm.tqdm(
            total=len(syn_lus),
            desc="Filtering LU to prepare mean embeddings for empty LU",
        ) as pbar:
            for syn_id, lu_ids in syn_lus.items():
                pbar.update(1)
                lu_embs = self.milvus_search_handler.get_lexical_units_embeddings(
                    lu_ids=lu_ids
                )
                found_lu_ids = [res["lu_id"] for res in lu_embs]
                not_found_lu_ids = list(set(lu_ids) - set(found_lu_ids))
                yield {
                    "syn_id": syn_id,
                    "status": {
                        "found": found_lu_ids,
                        "not_found": not_found_lu_ids,
                    },
                }

    def _get_synsets_with_lu_ids_pos_filter(self):
        """
        Get synsets mapped to lexical unit IDs filtered by part-of-speech.

        Retrieves all synset-to-lexical-unit mappings and filters lexical units
        based on the accepted part-of-speech categories. Only includes synsets
        that have at least one lexical unit matching the POS filter criteria.

        Returns:
            Dict[int, List[int]]: Mapping of synset IDs to lists
            of filtered lexical unit IDs
        """

        self.logger.info(
            f"Preparing synsets with LU ids with pos filtering: {self.accept_pos}"
        )

        synset_map = {}
        lu_in_syn = self.pl_wordnet.get_units_and_synsets(return_mapping=True)
        for s_id, lu_ids in lu_in_syn.items():
            proper_ids = []
            for _lu_id in lu_ids:
                lu = self.pl_wordnet.get_lexical_unit(lu_id=_lu_id)
                if lu is None:
                    self.logger.warning(f"Lexical unit {_lu_id} not found")
                    continue

                if (
                    self.accept_pos is not None
                    and len(self.accept_pos)
                    and lu.pos not in self.accept_pos
                ):
                    continue
                proper_ids.append(lu.ID)
            if len(proper_ids):
                synset_map[s_id] = proper_ids

        return synset_map

    def _prepare_embedding_for_empty_lu(self, lu_status):
        """
        Generate synthetic embeddings for lexical units without base embeddings.

        Computes mean embedding from existing embeddings of lexical units within
        the same synset and creates synthetic embeddings for units that lack
        base embeddings.

        Args:
            lu_status: Dictionary with 'found' and 'not_found' lexical unit ID lists

        Yields:
            Dict: Embedding dictionary containing lexical unit, computed embedding,
            type marker, and processing strategy
        """

        res = self.milvus_search_handler.get_lexical_units_embeddings(
            lu_ids=lu_status["found"]
        )
        all_found_embeddings = [r["embedding"] for r in res]

        new_emb_mean = self.embedding_processor.process(
            embeddings=all_found_embeddings
        )

        for not_found_lu_id in lu_status["not_found"]:
            lu = self.pl_wordnet.get_lexical_unit(lu_id=not_found_lu_id)
            if lu is None:
                self.logger.warning(
                    f"Lexical unit {not_found_lu_id} not found in PlWordnet API!"
                )
                continue

            yield {
                "lu": lu,
                "embedding": new_emb_mean,
                "type": EmbeddingTypes.Base.lu_fake,
                "strategy": self.embedding_processor.strategy,
            }
