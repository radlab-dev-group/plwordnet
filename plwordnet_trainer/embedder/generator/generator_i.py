from abc import abstractmethod, ABC
from typing import Dict, Iterator, Any, Optional, List

from plwordnet_handler.base.structure.polishwordnet import PolishWordnet
from plwordnet_handler.utils.logger import prepare_logger
from plwordnet_trainer.embedder.generator.strategy import (
    EmbeddingBuildStrategy,
    StrategyProcessor,
)

from plwordnet_trainer.embedder.generator.bi_encoder import (
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
