from abc import abstractmethod, ABC
from typing import Dict, Iterator, Any


class _ElemGeneratorBase(ABC):
    """
    Abstract base class for element generators.

    Defines the interface for generators that process elements and yield
    structured data with embeddings or other processed content.
    """

    @abstractmethod
    def generate(self, split_to_sentences: bool = False) -> Iterator[Dict[str, Any]]:
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
