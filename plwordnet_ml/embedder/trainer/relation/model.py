import torch
import torch.nn as nn

from plwordnet_ml.embedder.trainer.relation.layer import RelGATLayer
from plwordnet_ml.embedder.trainer.relation.scorer import (
    DistMultScorer,
    TransEScorer,
)


class RelGATModel(nn.Module):
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
