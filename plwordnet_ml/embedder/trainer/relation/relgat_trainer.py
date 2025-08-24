import torch
import random
import torch.nn.functional as F

from tqdm import tqdm
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader

from plwordnet_ml.utils.wandb_handler import WanDBHandler
from plwordnet_ml.embedder.trainer.relation.model import RelGATModel
from plwordnet_ml.embedder.trainer.relation.dataset import EdgeDataset


class RelGATTrainer:
    """
    Orchestrator for training a relation‑aware GNN.

    The trainer receives a pre‑computed embedding matrix,
    the edge list and relation mapping, and an optional
    multi‑threading configuration.  It exposes a single
    :meth:`train` method that runs the full training loop:

    1.  Build a :class:`~plwordnet_ml.utils.graph_builder.EdgeDataset`
        from the graph file.
    2.  Initialise a :class:`~plwordnet_ml.utils.graph_builder.RelGATModel`
        with the frozen embeddings.
    3.  Iterate over mini‑batches, compute scores with
        ``model.forward()``, and apply the margin‑ranking loss.
    4.  Use an Adam optimiser to update only the GAT and scorer
        parameters.
    5.  Log progress to the standard output and to an optional
        TensorBoard/Weights & Biases logger.

    Parameters
    ----------
    destination_path : pathlib.Path
        Directory where the graph data file resides.
    bi_encoder_cfg : BiEncoderModelConfig
        Configuration for the underlying bi‑encoder model.
    max_process_examples : int, default=100
        Maximum number of example texts to keep per lexical unit.
    auto_load : bool, default=True
        If ``True`` immediately load the bi‑encoder model during construction.
    max_workers : int, default=1
        Number of worker threads for parallel text extraction.
    """

    def __init__(
        self,
        node2emb: Dict[int, torch.Tensor],
        rel2idx: Dict[str, int],
        edge_index_raw: List[Tuple[int, int, str]],
        run_config: Dict,
        wandb_config,
        train_batch_size: int = 1024,
        num_neg: int = 4,
        train_ratio: float = 0.90,
        scorer_type: str = "distmult",
        gat_out_dim: int = 200,
        gat_heads: int = 6,
        dropout: float = 0.2,
        device: torch.device | None = None,
        run_name: str | None = None,
        log_every_n_steps: int = 100,
    ):
        """
        Initialise a full training pipeline for a relation‑aware GAT.

        Parameters
        ----------
        node2emb : dict[int, torch.Tensor]
            Mapping from node identifier to its frozen embedding vector
            (shape ``[1152]`` for the Polish BERT‑base encoder).
        rel2idx : dict[str, int]
            Mapping from relation name to an integer index.
        edge_index_raw : list[tuple[int, int, str]]
            Raw list of edges in the form ``(src_id, dst_id, relation_name)``.
            The list is shuffled internally to create a random train/test split.
        run_config : dict
            Arbitrary dictionary that will be logged as the run configuration
            in Weights & Biases (W&B).  It may contain hyper‑parameter values,
            timestamps, or experiment identifiers.
        wandb_config : object
            Namespace or simple object that provides the attributes
            ``PROJECT_NAME``, ``PROJECT_TAGS``, ``PREFIX_RUN`` and
            ``BASE_RUN_NAME`` used to initialise a W&B run.
        train_batch_size : int, default=1024
            Batch size for the training DataLoader.
        num_neg : int, default=4
            Number of negative samples to generate for each positive triple
            during dataset construction.
        train_ratio : float, default=0.90
            Fraction of edges that are allocated to the training split.
            The remaining edges are reserved for evaluation.
        scorer_type : str, default="distmult"
            Scoring head to use for relation prediction.
            Acceptable values are ``"distmult"`` and ``"transe"``.
        gat_out_dim : int, default=200
            Output dimensionality of each graph‑attention head.
        gat_heads : int, default=6
            Number of parallel attention heads.
        dropout : float, default=0.2
            Drop‑out probability applied after each GAT layer.
        device : torch.device | None, default=None
            Target device for the model and tensors.  If ``None`` the
            constructor selects ``cuda`` if available, otherwise ``cpu``.
        run_name : str | None, default=None
            Optional custom name for the W&B run.  When omitted a
            deterministic name is generated from ``wandb_config`` and
            ``run_config``.

        Notes
        -----
        * The constructor performs the following logical steps in order:

          1.  **Data preparation** – The edge list is shuffled, split into
              training and evaluation subsets, and wrapped in
              :class:`EdgeDataset`.  Two :class:`torch.utils.data.DataLoader`
              instances are created (shuffled for training, fixed for
              evaluation).

          2.  **Graph tensors** – ``edge_index`` and ``edge_type`` are
              converted to long‑tensors and stored as class attributes;
              the number of relation types is derived from ``rel2idx``.
              The frozen node embedding matrix is assembled from
              ``node2emb`` and moved to ``device``.

          3.  **Model creation** – A :class:`RelGATModel` is instantiated
              with the previously prepared tensors and hyper‑parameters,
              and the Adam optimiser is configured.

          4.  **W&B initialisation** – The helper
              :func:`WanDBHandler.init_wandb` is called so that
              hyper‑parameters and runtime metadata are logged
              automatically.  The call is idempotent; it will
              raise a warning if a run with the same name already
              exists.

        * The edge split is purely random; if reproducibility is required
          a global random seed should be set before constructing the object.

        * The ``EdgeDataset`` constructor expects the ``all_node_ids``
          argument – it is derived from the keys of ``node2emb`` and
          sorted for deterministic ordering.

        * The optimizer and model parameters are stored on the device
          chosen by the ``device`` argument, but the node embeddings are
          never updated during training – they act as fixed input
          features.

        Examples
        --------
        >>> trainer = RelGATTrainer(
        ...     node2emb=emb_dict,
        ...     rel2idx=rel_to_idx,
        ...     edge_index_raw=edges,
        ...     run_config={"epochs": 20},
        ...     wandb_config=wandb_cfg,
        ...     train_batch_size=512,
        ...     scorer_type="transe",
        ... )
        >>> trainer.train(num_epochs=20)
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.train_batch_size = train_batch_size
        self.num_neg = num_neg
        self.scorer_type = scorer_type
        self.gat_out_dim = gat_out_dim
        self.gat_heads = gat_heads
        self.dropout = dropout
        self.run_name = run_name

        # Logging
        self.global_step = 0
        self.log_every_n_steps = max(1, int(log_every_n_steps))

        # Dataset and dataset division
        self.node2emb = node2emb
        self.rel2idx = rel2idx
        self.edge_index_raw = edge_index_raw
        self.all_node_ids = sorted(node2emb.keys())

        # Embeddubgs id to proper idx
        self.id2idx = {nid: i for i, nid in enumerate(self.all_node_ids)}

        self.node_emb_matrix = torch.stack(
            [torch.as_tensor(self.node2emb[nid]) for nid in self.all_node_ids], dim=0
        ).to(self.device)

        # random division of edges to train/test
        random.shuffle(self.edge_index_raw)
        n_edges = len(self.edge_index_raw)
        n_train = int(train_ratio * n_edges)

        # Remap edges on compact indexes
        def _map_edge(e):
            s, d, r = e
            return self.id2idx[s], self.id2idx[d], r

        mapped_edges = [_map_edge(e) for e in self.edge_index_raw]
        self.train_edges = mapped_edges[:n_train]
        self.eval_edges = mapped_edges[n_train:]

        print(f"Number of edges (relations): {n_edges}")
        print(f" - train: {len(self.train_edges)} ({train_ratio*100:.1f} %)")
        print(f" - eval: {len(self.eval_edges)} ({100-train_ratio*100:.1f} %)")

        # Dataset / DataLoader
        self.train_dataset = EdgeDataset(
            edge_index=self.train_edges,
            node2emb=self.node2emb,
            rel2idx=self.rel2idx,
            num_neg=self.num_neg,
            all_node_ids=list(range(len(self.all_node_ids))),
        )
        self.eval_dataset = EdgeDataset(
            edge_index=self.eval_edges,
            node2emb=self.node2emb,
            rel2idx=self.rel2idx,
            num_neg=self.num_neg,
            all_node_ids=list(range(len(self.all_node_ids))),
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda batch: batch,
        )
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.train_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: batch,
        )

        # Mapping lu from triples on compact indexes
        src_list, dst_list, rel_list = zip(*mapped_edges)
        self.edge_index = torch.tensor([src_list, dst_list], dtype=torch.long).to(
            self.device
        )
        self.edge_type = torch.tensor(
            [self.rel2idx[r] if isinstance(r, str) else int(r) for r in rel_list],
            dtype=torch.long,
        ).to(self.device)
        self.num_rel = len(self.rel2idx)

        # Model + optimizer
        self.model = RelGATModel(
            node_emb=self.node_emb_matrix,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            num_rel=self.num_rel,
            scorer_type=self.scorer_type,
            gat_out_dim=self.gat_out_dim,
            gat_heads=self.gat_heads,
            dropout=self.dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # W&B initialization
        self.wandb_config = wandb_config
        self.run_config = run_config
        # keep logging freq in run config for reproducibility/visibility
        self.run_config["log_every_n_steps"] = self.log_every_n_steps
        WanDBHandler.init_wandb(
            wandb_config=self.wandb_config,
            run_config=self.run_config,
            training_args=None,
            run_name=self.run_name,
        )

    # Helper methods (loss, ranking, evaluation)
    @staticmethod
    def margin_ranking_loss(
        pos_score: torch.Tensor, neg_score: torch.Tensor, margin: float = 1.0
    ):
        pos = pos_score.unsqueeze(1).expand_as(neg_score)
        loss = F.relu(margin + neg_score - pos)
        return loss.mean()

    @staticmethod
    def compute_mrr_hits(
        scores: torch.Tensor, true_idx: int = 0, ks: Tuple[int, ...] = (1, 3, 10)
    ):
        rank = (scores > scores[true_idx]).sum().item() + 1
        mrr = 1.0 / rank
        hits = {k: 1.0 if rank <= k else 0.0 for k in ks}
        return mrr, hits

    def evaluate(self, ks: Tuple[int, ...] = (1, 3, 10)):
        self.model.eval()
        total_mrr = 0.0
        total_hits = {k: 0.0 for k in ks}
        n_examples = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluation", leave=False):
                pos, *negs = zip(*batch)

                src_ids = torch.cat(
                    [p[0] for p in pos] + [n[0] for n in sum(negs, ())], dim=0
                ).to(self.device)
                rel_ids = torch.cat(
                    [p[1] for p in pos] + [n[1] for n in sum(negs, ())], dim=0
                ).to(self.device)
                dst_ids = torch.cat(
                    [p[2] for p in pos] + [n[2] for n in sum(negs, ())], dim=0
                ).to(self.device)

                scores = self.model(src_ids, rel_ids, dst_ids)  # [B*(1+num_neg)]
                B = len(pos)
                pos_score = scores[:B]
                neg_score = scores[B:].view(B, self.num_neg)

            for i in range(B):
                cand_scores = torch.cat(
                    [pos_score[i].unsqueeze(0), neg_score[i]], dim=0
                )
                mrr, hits = self.compute_mrr_hits(cand_scores, true_idx=0, ks=ks)
                total_mrr += mrr
                for k in ks:
                    total_hits[k] += hits[k]
                n_examples += 1

        avg_mrr = total_mrr / n_examples
        avg_hits = {k: total_hits[k] / n_examples for k in ks}
        return avg_mrr, avg_hits

    # The main training loop
    def train(self, epochs: int = 12, margin: float = 1.0):
        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            # counter for incremental logging
            running_loss = 0.0
            running_examples = 0
            # ----------------- TRAIN -----------------
            for step_in_epoch, batch in enumerate(
                tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch:02d} – training",
                    leave=False,
                ),
                start=1,
            ):
                pos, *negs = zip(*batch)

                src_ids = torch.cat(
                    [p[0] for p in pos] + [n[0] for n in sum(negs, ())], dim=0
                ).to(self.device)
                rel_ids = torch.cat(
                    [p[1] for p in pos] + [n[1] for n in sum(negs, ())], dim=0
                ).to(self.device)
                dst_ids = torch.cat(
                    [p[2] for p in pos] + [n[2] for n in sum(negs, ())], dim=0
                ).to(self.device)

                scores = self.model(src_ids, rel_ids, dst_ids)
                B = len(pos)
                pos_score = scores[:B]
                neg_score = scores[B:].view(B, self.num_neg)

                loss = self.margin_ranking_loss(pos_score, neg_score, margin=margin)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # loss accumulation
                loss_item = loss.item()
                epoch_loss += loss_item * B
                running_loss += loss_item * B
                running_examples += B

                # global step and logging at every N steps
                self.global_step += 1
                if self.global_step % self.log_every_n_steps == 0:
                    avg_running_loss = running_loss / max(1, running_examples)
                    WanDBHandler.log_metrics(
                        metrics={
                            "epoch": epoch,
                            "train/loss_step": avg_running_loss,
                            "train/step_in_epoch": step_in_epoch,
                        },
                        step=self.global_step,
                    )
                    # reset counters
                    running_loss = 0.0
                    running_examples = 0

            avg_train_loss = epoch_loss / len(self.train_dataset)

            # ----------------- EVAL -----------------
            mrr, hits = self.evaluate(ks=(1, 2, 3))
            hits_str = ", ".join([f"Hits@{k}: {hits[k]:.4f}" for k in sorted(hits)])

            print(f"\n=== Epoch {epoch:02d} – loss: {avg_train_loss:.4f}")
            print(f"   - eval – MRR: {mrr:.4f} | {hits_str}")

            # ----------------- LOG TO W&B -----------------
            WanDBHandler.log_metrics(
                metrics={
                    "epoch": epoch,
                    "train/loss": avg_train_loss,
                    "val/mrr": mrr,
                    "val/hits@1": hits[1],
                    "val/hits@2": hits[2],
                    "val/hits@3": hits[3],
                },
                step=self.global_step,
            )

        # ----------------- SAVE MODEL  -----------------
        model_path = f"relgat_{self.scorer_type}_ratio{int(self.run_config.get('train_ratio', 0.9) * 100)}.pt"
        torch.save(self.model.state_dict(), model_path)
        print(f"\nTrening zakończony – model zapisano pod: {model_path}")

        # ----------------- ARTIFACT W&B -----------------
        # WanDBHandler.add_model(name=f"relgat-{self.scorer_type}", local_path="..")
        WanDBHandler.finish_wand()
