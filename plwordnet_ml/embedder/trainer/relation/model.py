import torch
import torch.nn as nn

from plwordnet_ml.embedder.trainer.relation.layer import RelGATLayer
from plwordnet_ml.embedder.trainer.relation.scorer import (
    DistMultScorer,
    TransEScorer,
)


class RelGATModel(nn.Module):
    """
    Relation‑aware graph neural network for knowledge‑graph completion.

    The model consists of three stages:

    1. **Fixed node embeddings** – A static matrix (typically obtained
       from a large‑scale bi‑encoder) that is never updated during
       training.  The matrix shape is ``(N, in_dim)`` where *N* is the
       number of nodes in the graph.

    2. **Relational Graph Attention layer** – A single
       :class:`~plwordnet_ml.embedder.trainer.relation.layer.RelGATLayer`
       that propagates information across edges while conditioning on the
       relation type.  It accepts the frozen embeddings, an adjacency
       matrix (``edge_index``) and a tensor of relation identifiers
       (``edge_type``).  The output is a new node representation of
       shape ``(N, out_dim * heads)`` that incorporates context from
       neighbours.

    3. **Scoring head** – Either a DistMult or TransE module that
       converts a pair of node embeddings together with a relation
       identifier into a scalar compatibility score.  Only the
       relation‑specific parameters of the scorer are learned; the node
       embeddings remain fixed.

    Parameters
    ----------
    node_emb : torch.Tensor
        Pre‑computed node embedding matrix of shape ``(N, in_dim)``.
        The tensor is treated as immutable during training.

    edge_index : torch.Tensor
        COO representation of the graph adjacency, shape ``(2, E)``.
        The first row contains source node IDs and the second row
        destination node IDs.

    edge_type : torch.Tensor
        Relation identifiers for each edge, shape ``(E,)``.  Values are
        integers in ``[0, num_rel)``.

    num_rel : int
        Total number of distinct relation types in the knowledge graph.

    gat_out_dim : int
        Output dimensionality of a single attention head in the
        Rel‑GAT layer.

    gat_heads : int
        Number of attention heads in the Rel‑GAT layer.

    dropout : float, default 0.2
        Drop‑out probability applied to the concatenated output of
        the attention heads.

    scorer_type : str, default "distmult"
        Either ``"distmult"`` or ``"transe"``.  Determines which
        scoring module is used for the final triple score.

    Attributes
    ----------
    node_emb : torch.Tensor
        Static node embeddings (unchanged during training).

    gat_layer : RelGATLayer
        Learnable Rel‑GAT layer that refines node vectors.

    scorer : nn.Module
        Instance of :class:`~plwordnet_ml.embedder.trainer.relation.scorer.DistMultScorer`
        or :class:`~plwordnet_ml.embedder.trainer.relation.scorer.TransEScorer`.

    Methods
    -------
    forward(src_ids, rel_ids, dst_ids)
        Computes scores for a batch of triples.  It first runs the
        Rel‑GAT layer on the static embeddings, then extracts the
        updated vectors for the source and destination nodes in the
        batch, and finally applies the scoring head.

    Notes
    -----
    * The entire model is differentiable; gradients flow only through
      the Rel‑GAT parameters and the scorer's relation embeddings.
    * ``node_emb`` must have the same dimensionality as the
      ``in_dim`` expected by the Rel‑GAT layer.
    * The model is intentionally lightweight: only a single Rel‑GAT
      layer and the scorer are trainable, which keeps memory usage
      low while still allowing the graph structure to influence node
      representations.

    Examples
    --------
    >>> # Assume ``node_emb`` and graph metadata are already loaded.
    >>> model = RelGATModel(
    ...     node_emb=node_emb,
    ...     edge_index=edge_index,
    ...     edge_type=edge_type,
    ...     num_rel=len(rel2idx),
    ...     gat_out_dim=200,
    ...     gat_heads=6,
    ...     dropout=0.2,
    ...     scorer_type="distmult",
    ... )
    >>> src_ids = torch.tensor([1, 5, 23], dtype=torch.long)
    >>> rel_ids = torch.tensor([2, 0, 3], dtype=torch.long)
    >>> dst_ids = torch.tensor([7, 11, 9], dtype=torch.long)
    >>> scores = model(src_ids, rel_ids, dst_ids)
    >>> scores.shape
    torch.Size([3])
    """

    def __init__(
        self,
        node_emb: torch.Tensor,  # [N, 1152] (frozen)
        edge_index: torch.Tensor,  # [2, E]
        edge_type: torch.Tensor,  # [E]
        num_rel: int,
        scorer_type: str = "distmult",  # "transe"
        gat_out_dim: int = 200,
        gat_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.register_buffer("node_emb_fixed", node_emb)  # not a Parameter
        self.edge_index = edge_index
        self.edge_type = edge_type

        self.gat = RelGATLayer(
            in_dim=node_emb.size(1),
            out_dim=gat_out_dim,
            num_rel=num_rel,
            heads=gat_heads,
            dropout=dropout,
            use_bias=True,
        )

        scorer_dim = gat_out_dim * gat_heads
        if scorer_type.lower() == "distmult":
            self.scorer = DistMultScorer(num_rel, rel_dim=scorer_dim)
        elif scorer_type.lower() == "transe":
            self.scorer = TransEScorer(num_rel, rel_dim=scorer_dim)
        else:
            raise ValueError(f"Unknown scorer_type: {scorer_type}")

    def forward(
        self, src_ids: torch.Tensor, rel_ids: torch.Tensor, dst_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute scores for a batch of triples.
         src_ids, rel_ids, dst_ids : [B]  (int64)

        Parameters
        ----------
        src_ids : torch.LongTensor [B]
            Source node indices.
        rel_ids : torch.LongTensor [B]
            Relation indices.
        dst_ids : torch.LongTensor [B]
            Destination node indices.

        Returns
        -------
        torch.Tensor [B]
            Compatibility scores (higher ⇒ more plausible).
        """
        # 1️⃣ Refine node vectors once per forward pass (could be cached)
        refined_nodes = self.gat(
            self.node_emb_fixed, self.edge_index, self.edge_type
        )  # [N, D']

        src_vec = refined_nodes[src_ids]  # [B, D']
        dst_vec = refined_nodes[dst_ids]  # [B, D']

        # 2️⃣ Score
        scores = self.scorer(src_vec, rel_ids, dst_vec)  # [B]
        return scores
