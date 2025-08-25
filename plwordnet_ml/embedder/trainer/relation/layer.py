import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_add


class RelGATLayer(nn.Module):
    """
    One relational‑GAT layer with multi‑head attention.
    - `in_dim`  : dimension of *input* node vectors (1152 in your case)
    - `out_dim` : dimension of *output* node vectors (e.g. 200)
    - `num_rel` : number of distinct relation types (≈40‑50)
    - `heads`   : number of attention heads
    - `use_bias`: optional scalar weight per relation (β_r) to modulate influence

        A relational Graph Attention Network (Rel‑GAT) layer that aggregates
        information from neighbouring nodes while explicitly modelling
        relation types between them.

        The implementation follows the multi‑head attention mechanism of
        *Relational Graph Attention Networks* with the following design
        choices:

        * **Input / output dimensionality** – The layer accepts node
          embeddings of dimension ``in_dim`` and produces updated node
          vectors of dimension ``heads * out_dim``.  Each attention head
          operates in its own ``out_dim`` dimensional space and the
          outputs are concatenated.
        * **Relation‑specific attention** – For every head a trainable
          attention vector of shape ``[num_rel, out_dim]`` is used.
          The attention score for an edge ``i → j`` belonging to relation
          ``r`` is computed as ``LeakyReLU((h_i ⊙ a_h[r]).sum)`` where
          ``h_i`` is the projected source embedding.
        * **Optional bias per relation** – A learnable scalar
          ``β_r`` can be added to messages of relation ``r`` after
          aggregation.  When disabled the bias is omitted.
        * **Softmax normalisation** – Attention weights are normalised
          per destination node and per head using the
          :func:`torch_scatter.scatter_add` primitive, ensuring that
          every neighbour contributes proportionally to its relation type
          and relation‑specific attention.
        * **Dropout** – A dropout layer is applied to the concatenated
          output to regularise the model.

        Parameters
        ----------
        in_dim : int
            Dimensionality of the input node embeddings.
        out_dim : int
            Dimensionality of each attention head's output space.
        num_rel : int
            Number of distinct relation types in the knowledge graph.
        heads : int, default 4
            Number of independent attention heads.
        dropout : float, default 0.2
            Drop‑out probability applied to the concatenated output.
        use_bias : bool, default True
            Whether to learn an additive bias for each relation type.

        Notes
        -----
        * All linear projections share the same parameters across
          relations; only the attention vectors and optional bias are
          relation‑specific.
        * The implementation assumes that the node embeddings are fixed
          (e.g., pre‑trained or stored in a database) and only the
          relation embeddings and attention parameters are updated
          during training.
        * The forward method expects ``edge_index`` in COO format
          ``[2, E]`` and ``edge_type`` of shape ``[E]``.

        Examples
        --------
        >>> layer = RelGATLayer(in_dim=1152, out_dim=200, num_rel=45, heads=4)
        >>> node_emb = torch.randn(1000, 1152)
        >>> edge_index = torch.randint(0, 1000, (2, 5000))
        >>> edge_type = torch.randint(0, 45, (5000,))
        >>> out = layer(node_emb, edge_index, edge_type)
        >>> out.shape
        torch.Size([1000, 800])   # 4 heads × 200 features each
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_rel: int,
        heads: int = 4,
        dropout: float = 0.2,
        use_bias: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.num_rel = num_rel
        self.dropout = nn.Dropout(dropout)

        # Linear projection per head (shared across relations)
        self.proj = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=False) for _ in range(heads)]
        )

        # Relation‑specific attention vectors (one per head)
        self.attn_vec = nn.ParameterList(
            [nn.Parameter(torch.empty(num_rel, out_dim)) for _ in range(heads)]
        )

        # Optional scalar bias per relation (β_r)
        if use_bias:
            self.rel_bias = nn.Parameter(torch.zeros(num_rel))
        else:
            self.rel_bias = None

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.proj:
            nn.init.xavier_uniform_(lin.weight)
        for a in self.attn_vec:
            nn.init.xavier_uniform_(a)

    def forward(
        self,
        node_emb: torch.Tensor,
        edge_index: torch.Tensor,  # shape [2, E]  (src, dst)
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform a single Rel‑GAT message‑passing step.

        The method receives *fixed* node embeddings, a graph
        adjacency represented in COO format (`edge_index`) and the
        relation type for each edge.  It outputs updated node
        embeddings that incorporate information from neighbouring
        nodes while weighting them with relation‑specific
        attention.

        Parameters
        ----------
        node_emb : torch.Tensor
            Node feature matrix of shape ``[N, in_dim]`` where *N* is the
            number of nodes in the graph.  These embeddings are not
            updated in‑place (fixed embeddings); a new tensor of shape ``[N, heads*out_dim]``
            is returned.
        edge_index : torch.Tensor
            Edge indices in COO format of shape ``[2, E]``.  The first
            row contains source node IDs and the second row contains
            destination node IDs (src ids, dst ids).
        edge_type : torch.Tensor
            Relation identifiers for each edge, shape ``[E]``.  The
            values should be integers in ``[0, num_rel)`` (relation id for each edge).
        Returns
        -------
        torch.Tensor
            Updated node embeddings of shape ``[N, heads*out_dim]``.
            Each row corresponds to a node and is the concatenation of
            the outputs from all attention heads.

        Notes
        -----
        1. **Projection** – Each attention head applies a linear
           projection (without bias) to all node embeddings
           (`self.proj[h]`) to obtain ``proj_src`` and ``proj_dst``
           for the source and destination nodes of every edge.

        2. **Attention scores** – For head *h* the un‑normalised
           attention for edge *i → j* belonging to relation *r*
           is calculated as

           .. math::
               e_{ij} = \operatorname{LeakyReLU}\big(
                   ( \text{proj\_src}_h[i] \odot
                     \text{attn\_vec}_h[r] )
                   \cdot \mathbf{1}\big),

           where ``⊙`` denotes element‑wise product and
           ``·`` is the dot product over feature dimension.
           The leaky ReLU uses a slope of 0.2.

        3. **Softmax normalisation** – The scores are exponentiated
           and normalised per destination node and per head using
           ``torch_scatter.scatter_add``.  This yields a weight
           ``α_{ij}^h`` for each incoming edge.

        4. **Message passing** – The source projection is multiplied by
           the corresponding weight, and messages are summed for each
           destination node.  Optionally a relation‑specific bias
           ``β_r`` is added to every aggregated message of relation
           *r*.

        5. **Output** – The aggregated messages from all heads are
           concatenated, optionally passed through dropout, and
           returned as the new node embeddings.

        Examples
        --------
        >>> layer = RelGATLayer(in_dim=1152, out_dim=200, num_rel=45, heads=4)
        >>> node_emb = torch.randn(1000, 1152)
        >>> edge_index = torch.randint(0, 1000, (2, 5000))
        >>> edge_type = torch.randint(0, 45, (5000,))
        >>> out = layer(node_emb, edge_index, edge_type)
        >>> out.shape
        torch.Size([1000, 800])   # 4 heads × 200 features each
        """
        src, dst = edge_index
        N = node_emb.size(0)

        # 1. Project node vectors for each head
        # proj_src[h] : [E, out_dim]
        proj_src = [lin(node_emb)[src] for lin in self.proj]  # list of length heads
        proj_dst = [lin(node_emb)[dst] for lin in self.proj]

        # 2. Compute un‑normalized attention
        # For head h:  e_ij = (proj_src_h ⊙ a_h[rel]) · 1   (LeakyReLU)
        attn_scores = []
        for h in range(self.heads):
            # a_h[rel] : [E, out_dim]
            rel_att = self.attn_vec[h][edge_type]  # gather per edge
            # element‑wise product then sum over feature dim
            e = (proj_src[h] * rel_att).sum(dim=-1)  # [E]
            e = F.leaky_relu(e, negative_slope=0.2)
            attn_scores.append(e)

        # 3. Softmax over neighbours (per destination)
        # Stack heads -> [heads, E]
        attn = torch.stack(attn_scores, dim=0)  # [H, E]
        # Apply softmax separately for each head
        attn = torch.exp(attn)
        # denominator: sum over incoming edges for each dst node, per head
        denom = scatter_add(attn, dst, dim=1, dim_size=N)  # [H, N]
        # avoid division by zero
        denom = denom + 1e-16
        attn = attn / denom[:, dst]  # normalized, [H, E]

        # 4. Message passing
        # Multiply source projection by attention weight
        msgs = [proj_src[h] * attn[h].unsqueeze(-1) for h in range(self.heads)]
        # Aggregate messages per destination node
        out = [
            scatter_add(msg, dst, dim=0, dim_size=N) for msg in msgs
        ]  # list of [N, out_dim]

        # 5. Optional relation bias
        if self.rel_bias is not None:
            # bias per edge, broadcast to destination node after aggregation
            bias = self.rel_bias[edge_type]  # [E]
            bias = scatter_add(bias, dst, dim=0, dim_size=N)  # [N]
            bias = bias.unsqueeze(-1)  # [N,1]
            out = [h + bias for h in out]

        # 6. Concatenate heads
        out = torch.cat(out, dim=-1)  # [N, heads*out_dim]
        out = self.dropout(out)
        return out
