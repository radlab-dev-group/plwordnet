import numpy
import torch

from enum import Enum
from typing import List, Optional


class EmbeddingBuildStrategy(Enum):
    """
    Enumeration defining strategies for building embeddings.

    Attributes:
        MEAN: Mean strategy for combining embeddings
        MEAN_WEIGHTED: Mean weighted by given weights
    """

    MEAN = "mean"
    MEAN_WEIGHTED = "mean_weighted"


class StrategyProcessor:
    def __init__(self, strategy: EmbeddingBuildStrategy):
        """
        Initialize processor with given strategy.

        Args:
            strategy: Strategy to build embeddings
        """
        self.strategy = strategy

    def process(
        self,
        embeddings: List[torch.Tensor | List[torch.Tensor]],
        check_shape: bool = False,
        weights: Optional[torch.Tensor] = None,
        normalize_weights: bool = True,
        normalize_out_embedding: bool = False,
    ) -> torch.Tensor:
        """
        Process a list of embeddings according to the selected strategy.
        Produce a single embedding for the given list of embeddings
        using the selected strategy and normalization.

        Args:
            embeddings: List of embedding tensors to process
            check_shape: If True, check the shape of embeddings
            and Raise exception if not all shapes are the same.
            weights: Optional[torch.Tensor] with weights used with AVG weighted
            normalize_weights: Whether to normalize the weights (sum = 1).
            normalize_out_embedding: Whether to normalize the output embeddings.

        Returns:
            torch.Tensor: Processed embedding tensor according to the strategy

        Raises:
            ValueError: If an embedding list is empty or strategy is not supported
            or weights are not provided when weighted strategy should be used.
            RuntimeError: If embeddings have incompatible shapes
        """
        if not len(embeddings):
            raise ValueError("Embeddings list cannot be empty")

        if check_shape:
            shapes = [emb.shape for emb in embeddings]
            if not all(shape == shapes[0] for shape in shapes):
                raise RuntimeError(
                    f"All embeddings must have the same shape, got: {shapes}"
                )

        if weights is not None:
            if type(weights) is not torch.Tensor:
                weights = torch.tensor(weights)

            if normalize_weights:
                weights = weights / sum(weights)

        out_emb = None
        embeddings = torch.Tensor(numpy.array(embeddings))
        if self.strategy == EmbeddingBuildStrategy.MEAN:
            if len(embeddings) == 1:
                out_emb = embeddings[0]
            else:
                out_emb = torch.mean(embeddings, dim=0)
        elif self.strategy == EmbeddingBuildStrategy.MEAN_WEIGHTED:
            if weights is None:
                raise ValueError("Weights cannot be None for weighted strategy")
            # tensor to simply processing
            out_emb = torch.Tensor(numpy.array(embeddings))
            # Reshape weights instead of emb transposing
            out_emb = out_emb * weights.view(-1, 1)
            out_emb = torch.mean(out_emb, dim=0)

        if out_emb is None:
            raise RuntimeError("Probably bad strategy used to build embeddings!")

        if normalize_out_embedding:
            out_emb = out_emb / sum(out_emb)

        return out_emb
