from abc import abstractmethod, ABC
from typing import Dict, Iterator, Any, Optional, List

from plwordnet_handler.base.structure.polishwordnet import PolishWordnet
from plwordnet_handler.utils.logger import prepare_logger
from plwordnet_trainer.embedder.generator.strategy import (
    EmbeddingBuildStrategy,
    StrategyProcessor,
)

from plwordnet_trainer.embedder.bi_encoder import (
    BiEncoderEmbeddingGenerator,
)


class _AnySemanticEmbeddingGeneratorBase(ABC):
    """
    Abstract base class for element generators.

    Defines the interface for generators that process elements and yield
    structured data with embeddings or other processed content.
    """

    def __init__(
        self,
        generator: Optional[BiEncoderEmbeddingGenerator] = None,
        strategy: Optional[EmbeddingBuildStrategy] = EmbeddingBuildStrategy.MEAN,
        log_level: str = "INFO",
        log_name: str = None,
        log_filename: str = None,
        accept_pos: Optional[List[int]] = None,
        pl_wordnet: Optional[PolishWordnet] = None,
    ):
        """
        Initialize the embedding processor with configuration and logging setup.

        Sets up the embedding generator, processing strategy, and logger
        configuration for handling semantic embedding operations.

        Args:
            generator: Optional bi-encoder embedding generator instance
            strategy: Strategy for building embeddings from multiple components.
            Defaults to MEAN strategy
            log_level: Logging level for the processor. Defaults to "INFO"
            log_name: Custom logger name. Uses module name if None
            log_filename: Optional log file path for file-based logging
            accept_pos: List of accepted POS (integers) tags (default: None).
            pl_wordnet: List of accepted POS (integers) tags (default: None).
        """
        self.accept_pos = accept_pos
        self.pl_wordnet = pl_wordnet
        self.embedding_generator = generator

        self.embedding_processor = (
            StrategyProcessor(strategy=strategy) if strategy is not None else None
        )

        self.logger = prepare_logger(
            logger_name=__name__ if log_name is None else log_name,
            log_level=log_level,
            logger_file_name=log_filename,
        )

    @abstractmethod
    def generate(
        self, split_to_sentences: Optional[bool] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Generate processed data from elements.

        Args:
            split_to_sentences: Whether to split text content into individual
            sentences for more granular processing

        Yields:
            Dict[str, Any]: Dictionary containing processed element data

        Note:
            Must be implemented by concrete subclasses
        """

        raise NotImplementedError

    def _get_units_and_synsets_with_pos(
        self, accepted_pos: Optional[List[int]] = None
    ) -> Dict[int, List[int]]:
        """
        Filter synset-to-lexical-unit mappings by part-of-speech categories.

        Retrieves all synset-to-lexical-unit mappings and filters lexical units
        based on accepted part-of-speech categories. Returns only synsets that
        contain at least one lexical unit matching the POS criteria.

        Args:
            accepted_pos: List of accepted part-of-speech category IDs.
            If None or empty, returns all mappings without filtering

        Returns:
            Dict[int, List[int]]: Dictionary mapping synset IDs to lists of
            filtered lexical unit IDs that match the POS criteria
        """

        self.logger.info(
            f"Filtering lu in synsets to LU ids with pos: {self.accept_pos}"
        )

        lu_in_syn = self.pl_wordnet.get_units_and_synsets(return_mapping=True)
        if accepted_pos is None or not len(accepted_pos):
            return lu_in_syn

        syn_map = {}
        lu_in_syn = self.pl_wordnet.get_units_and_synsets(return_mapping=True)
        for s_id, lu_ids in lu_in_syn.items():
            proper_ids = []
            for _lu_id in lu_ids:
                lu = self.pl_wordnet.get_lexical_unit(lu_id=_lu_id)
                if lu is None:
                    self.logger.warning(f"Lexical unit {_lu_id} not found")
                    continue
                if lu.pos not in accepted_pos:
                    continue
                proper_ids.append(lu.ID)
            if len(proper_ids):
                syn_map[s_id] = proper_ids
        return syn_map
