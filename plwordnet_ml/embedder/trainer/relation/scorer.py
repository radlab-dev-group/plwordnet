import torch
import torch.nn as nn


class DistMultScorer(nn.Module):
    """
    DistMult knowledge‑graph scoring module.

    Implements the DistMult scoring function
    ``score = (src ⊙ rel ⊙ dst) · 1``
    where ``⊙`` denotes element‑wise multiplication. The module learns a single
    embedding matrix for relation types; node embeddings are expected to be
    provided externally and are kept frozen during training.

    Parameters
    ----------
    num_rel : int
        Number of distinct relation types in the knowledge graph.
    rel_dim : int, default 200
        Dimensionality of the relation embeddings.  This dimension
        should match the dimensionality of the node embeddings that
        will be supplied to :meth:`forward`.

    Notes
    -----
    * The relation embeddings are initialized with the Xavier
      uniform distribution to facilitate stable training.
    * ``forward`` returns a score vector of shape ``[B]`` where
      higher values correspond to more plausible triples.

    Examples
    --------
    >>> scorer = DistMultScorer(num_rel=10, rel_dim=128)
    >>> src = torch.randn(32, 128)
    >>> dst = torch.randn(32, 128)
    >>> rel_ids = torch.randint(0, 10, (32,))
    >>> scores = scorer(src, rel_ids, dst)
    >>> scores.shape
    torch.Size([32])
    """

    def __init__(self, num_rel: int, rel_dim: int = 200):
        """
        Initialize the DistMult scorer.

        Parameters
        ----------
        num_rel : int
            Number of distinct relation types in the knowledge graph.
        rel_dim : int, optional
            Dimensionality of the relation embeddings.
        """

        super().__init__()
        self.rel_emb = nn.Embedding(num_rel, rel_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(
        self, src_emb: torch.Tensor, rel_ids: torch.Tensor, dst_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DistMult scores for a batch of triples.

        Parameters
        ----------
        src_emb : torch.Tensor
            Source node embeddings of shape ``[B, D]``. (D = rel_dim)
        rel_ids : torch.Tensor
            Relation indices of shape ``[B]``.
        dst_emb : torch.Tensor
            Destination node embeddings of shape ``[B, D]``.

        Returns
        -------
        torch.Tensor
            Score tensor of shape ``[B]``; higher scores indicate
            more plausible triples.
        """
        # [B, D] shape
        rel_emb = self.rel_emb(rel_ids)

        # dot product
        score = (src_emb * rel_emb * dst_emb).sum(dim=-1)
        return score


class TransEScorer(nn.Module):
    """
    TransE knowledge‑graph scoring module.

    Implements the TransE scoring function
    ``score = -|| src + rel - dst ||_2``.  The negative sign
    turns the Euclidean distance into a similarity measure: a
    larger score indicates a more plausible triple.  Only relation
    embeddings are learned; node embeddings are expected to be
    provided externally.

    Parameters
    ----------
    num_rel : int
        Number of distinct relation types in the knowledge graph.
    rel_dim : int, default 200
        Dimensionality of the relation embeddings.

    Notes
    -----
    * Relation embeddings are initialized with the Xavier uniform
      distribution.
    * ``forward`` returns a negative L2 distance; this is convenient
      when using a margin‑ranking loss that prefers larger scores.

    Examples
    --------
    >>> scorer = TransEScorer(num_rel=10, rel_dim=128)
    >>> src = torch.randn(32, 128)
    >>> dst = torch.randn(32, 128)
    >>> rel_ids = torch.randint(0, 10, (32,))
    >>> scores = scorer(src, rel_ids, dst)
    >>> scores.shape
    torch.Size([32])
    """

    def __init__(self, num_rel: int, rel_dim: int = 200):
        """
        Initialize the TransE scorer.

        Parameters
        ----------
        num_rel : int
            Number of distinct relation types in the knowledge graph.
        rel_dim : int, optional
            Dimensionality of the relation embeddings.
        """

        super().__init__()
        self.rel_emb = nn.Embedding(num_rel, rel_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(
        self, src_emb: torch.Tensor, rel_ids: torch.Tensor, dst_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute TransE scores for a batch of triples.

        Parameters
        ----------
        src_emb : torch.Tensor
            Source node embeddings of shape ``[B, D]``.
        rel_ids : torch.Tensor
            Relation indices of shape ``[B]``.
        dst_emb : torch.Tensor
            Destination node embeddings of shape ``[B, D]``.

        Returns
        -------
        torch.Tensor
            Score tensor of shape ``[B]``; larger values correspond to
            more plausible triples (since the distance is negated).
        """
        # [B, D] shape
        rel_emb = self.rel_emb(rel_ids)

        # L2 distance
        distance = torch.norm(src_emb + rel_emb - dst_emb, p=2, dim=-1)

        # we return the *negative* distance so that a higher score = better
        return -distance
