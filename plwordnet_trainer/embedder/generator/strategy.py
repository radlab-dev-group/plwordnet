import numpy
import torch

from enum import Enum
from typing import List, Optional


class EmbeddingBuildStrategy(Enum):
    """
    Enumeration defining strategies for building embeddings.

    Attributes:
        MEAN: Mean strategy for combining embeddings
    """

    MEAN = "mean"


class StrategyProcessor:
    def __init__(self, strategy: EmbeddingBuildStrategy):
        """
        Initialize processor with given strategy.

        Args:
            strategy: Strategy to build embeddings
        """
        self.strategy = strategy

    def process(
        self, embeddings: List[torch.Tensor], check_shape: bool = False
    ) -> torch.Tensor:
        """
        Process a list of embeddings according to the selected strategy.

        Args:
            embeddings: List of embedding tensors to process
            check_shape: If True, check the shape of embeddings
            and Raise exception if not all shapes are the same.

        Returns:
            torch.Tensor: Processed embedding tensor according to the strategy

        Raises:
            ValueError: If an embedding list is empty or strategy is not supported
            RuntimeError: If embeddings have incompatible shapes
        """
        if not len(embeddings):
            raise ValueError("Embeddings list cannot be empty")

        if len(embeddings) == 1:
            return embeddings[0]

        if check_shape:
            shapes = [emb.shape for emb in embeddings]
            if not all(shape == shapes[0] for shape in shapes):
                raise RuntimeError(
                    f"All embeddings must have the same shape, got: {shapes}"
                )

        if self.strategy == EmbeddingBuildStrategy.MEAN:
            return torch.mean(torch.Tensor(numpy.array(embeddings)), dim=0)
