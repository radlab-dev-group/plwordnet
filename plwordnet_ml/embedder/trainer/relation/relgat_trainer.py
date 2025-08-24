import json
import torch
import random
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any

from plwordnet_ml.utils.wandb_handler import WanDBHandler
from plwordnet_ml.embedder.trainer.relation.model import RelGATModel
from plwordnet_ml.embedder.trainer.relation.dataset import EdgeDataset
from plwordnet_ml.embedder.trainer.main.parts.constants import ConstantsRelGATTrainer


class RelGATTrainer:
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
        gat_num_layers: int = 1,
        dropout: float = 0.2,
        device: Optional[torch.device] = None,
        run_name: Optional[str] = None,
        log_every_n_steps: int = 100,
        save_dir: Optional[str] = None,
        save_every_n_steps: Optional[int] = None,
        eval_every_n_steps: Optional[int] = None,
    ):
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
        self.gat_num_layers = gat_num_layers

        # Logging
        self.global_step = 0
        self.log_every_n_steps = max(1, int(log_every_n_steps))

        # LR/scheduler config
        self.scheduler = None
        self.base_lr = float(run_config["lr"])
        self.scheduler_type = str(run_config["lr_scheduler"]).lower()
        self.default_warmup_ratio = (
            ConstantsRelGATTrainer.Default.DEFAULT_WARMUP_RATIO
        )

        # Model saving
        self.save_dir = (
            Path(save_dir)
            if save_dir is not None
            else Path(ConstantsRelGATTrainer.Default.DEFAULT_TRAINER_OUT_DIR)
        )
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_steps = (
            int(save_every_n_steps)
            if save_every_n_steps is not None and int(save_every_n_steps) > 0
            else None
        )

        self.eval_every_n_steps = (
            int(eval_every_n_steps)
            if eval_every_n_steps is not None and int(eval_every_n_steps) > 0
            else None
        )

        # Dataset and dataset division
        self.node2emb = node2emb
        self.rel2idx = rel2idx
        self.edge_index_raw = edge_index_raw
        self.all_node_ids = sorted(node2emb.keys())

        # Embeddings id to proper idx
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
            gat_num_layers=self.gat_num_layers,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.base_lr)

        # W&B initialization
        self.wandb_config = wandb_config
        self.run_config = run_config
        # keep logging freq in run config for reproducibility/visibility
        self.run_config["log_every_n_steps"] = self.log_every_n_steps
        # ensure lr and scheduler info are present in run config (for reproducibility)
        self.run_config["lr"] = self.base_lr
        self.run_config["lr_scheduler"] = self.scheduler_type
        #
        self.run_config["save_every_n_steps"] = self.save_every_n_steps
        self.run_config["save_dir"] = str(self.save_dir)
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
        # ---- Scheduler (warmup + selected decay) ----
        import math
        import math as _math  # for cosine

        steps_per_epoch = max(
            1, math.ceil(len(self.train_dataset) / self.train_batch_size)
        )
        total_steps = steps_per_epoch * max(1, int(epochs))
        warmup_steps_cfg = self.run_config.get("warmup_steps", None)
        if warmup_steps_cfg is None:
            warmup_steps = int(self.default_warmup_ratio * total_steps)
        warmup_steps = min(warmup_steps, max(0, total_steps - 1))

        def _lr_lambda_linear(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - current_step)
                / float(max(1, total_steps - warmup_steps)),
            )

        def _lr_lambda_cosine(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + _math.cos(_math.pi * min(1.0, max(0.0, progress))))

        def _lr_lambda_constant(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        if self.scheduler_type == "linear":
            lr_lambda = _lr_lambda_linear
        elif self.scheduler_type == "cosine":
            lr_lambda = _lr_lambda_cosine
        elif self.scheduler_type == "constant":
            lr_lambda = _lr_lambda_constant
        else:
            raise ValueError(f"Unknown lr_scheduler type: {self.scheduler_type}")

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda
        )

        # Log scheduler info (once)
        WanDBHandler.log_metrics(
            metrics={
                "scheduler/total_steps": total_steps,
                "scheduler/warmup_steps": warmup_steps,
                "scheduler/type": self.scheduler_type,
                "train/base_lr": self.base_lr,
            },
            step=self.global_step,
        )

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

                # step scheduler after optimizer step
                if self.scheduler is not None:
                    self.scheduler.step()

                # loss accumulation
                loss_item = loss.item()
                epoch_loss += loss_item * B
                running_loss += loss_item * B
                running_examples += B

                # global step and logging at every N steps
                self.global_step += 1

                # logging every n steps
                if self.global_step % self.log_every_n_steps == 0:
                    avg_running_loss = running_loss / max(1, running_examples)
                    current_lr = (
                        self.scheduler.get_last_lr()[0]
                        if self.scheduler is not None
                        else self.optimizer.param_groups[0]["lr"]
                    )
                    print(
                        f"\nGlobal step {self.global_step} "
                        f"/in epoch {step_in_epoch}/ "
                        f"loss_step: {avg_running_loss:.8f} "
                        f"lr: {current_lr:.8f}"
                    )

                    WanDBHandler.log_metrics(
                        metrics={
                            "epoch": epoch,
                            "train/loss_step": avg_running_loss,
                            "train/step_in_epoch": step_in_epoch,
                            "train/lr": current_lr,
                        },
                        step=self.global_step,
                    )
                    # reset counters
                    running_loss = 0.0
                    running_examples = 0

                # Eval model every n steps
                if (
                    self.eval_every_n_steps is not None
                    and self.global_step % self.eval_every_n_steps == 0
                ):
                    avg_train_loss = epoch_loss / len(self.train_dataset)
                    # ----------------- EVAL -----------------
                    mrr, hits = self.evaluate(ks=(1, 2, 3))
                    hits_str = ", ".join(
                        [f"Hits@{k}: {hits[k]:.4f}" for k in sorted(hits)]
                    )
                    print(
                        f"\n=== Step {self.global_step:12d} – "
                        f"loss: {epoch_loss:.4f} "
                        f"[avg. loss: {avg_train_loss:.4f}]"
                    )
                    print(f"   - eval – MRR: {mrr:.4f} | {hits_str}")
                    # ----------------- LOG TO W&B -----------------
                    WanDBHandler.log_metrics(
                        metrics={
                            "epoch": epoch,
                            "eval/loss": avg_train_loss,
                            "eval/mrr": mrr,
                            "eval/hits@1": hits[1],
                            "eval/hits@2": hits[2],
                            "eval/hits@3": hits[3],
                        },
                        step=self.global_step,
                    )

                # saving checkpoints every n steps
                if (
                    self.save_every_n_steps is not None
                    and self.global_step % self.save_every_n_steps == 0
                ):
                    chk_dir_name = f"checkpoint-{self.global_step}"
                    self._save_checkpoint(
                        subdir=chk_dir_name, run_config=self.run_config
                    )
                    # log saved checkpoint to w&b
                    WanDBHandler.log_metrics(
                        metrics={"checkpoint/step": self.global_step},
                        step=self.global_step,
                    )

            if self.eval_every_n_steps is None:
                avg_train_loss = epoch_loss / len(self.train_dataset)
                # ----------------- EVAL -----------------
                mrr, hits = self.evaluate(ks=(1, 2, 3))
                hits_str = ", ".join(
                    [f"Hits@{k}: {hits[k]:.4f}" for k in sorted(hits)]
                )
                print(f"\n=== Epoch {epoch:02d} – loss: {avg_train_loss:.4f}")
                print(f"   - eval – MRR: {mrr:.4f} | {hits_str}")
                # ----------------- LOG TO W&B -----------------
                WanDBHandler.log_metrics(
                    metrics={
                        "epoch": epoch,
                        "eval/loss": avg_train_loss,
                        "eval/mrr": mrr,
                        "eval/hits@1": hits[1],
                        "eval/hits@2": hits[2],
                        "eval/hits@3": hits[3],
                    },
                    step=self.global_step,
                )

        # ----------------- SAVE FINAL MODEL  -----------------
        out_model_dir = (
            f"relgat_{self.scorer_type}"
            f"_ratio{int(self.run_config['train_ratio'] * 100)}"
        )
        self._save_checkpoint(subdir=out_model_dir, run_config=self.run_config)
        print(f"\nTrening zakończony – model zapisano pod: {out_model_dir}")

        # ----------------- ARTIFACT W&B -----------------
        # WanDBHandler.add_model(name=f"relgat-{self.scorer_type}", local_path="..")
        WanDBHandler.finish_wand()

    def _save_checkpoint(
        self, subdir: str, run_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Saves model state_dict into self.save_dir/subdir/OUT_MODEL_NAME
        Returns saved file path.
        """
        out_dir = self.save_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / ConstantsRelGATTrainer.Default.OUT_MODEL_NAME
        torch.save(self.model.state_dict(), out_path)

        if run_config is not None and len(run_config):
            out_cfg_path = (
                out_dir / ConstantsRelGATTrainer.Default.TRAINING_CONFIG_FILE_NAME
            )
            with open(out_cfg_path, "w") as f:
                f.write(json.dumps(run_config, indent=2, ensure_ascii=False))

        return str(out_path)
