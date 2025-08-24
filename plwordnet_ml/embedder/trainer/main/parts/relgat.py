import json
import torch
import pickle
import argparse

from plwordnet_ml.embedder.trainer.relation.relgat_trainer import RelGATTrainer


class ConstantsRelGATTrainer:
    class Default:
        EPOCHS = 12
        TRAIN_EVAL_RATIO = 0.9
        TRAIN_BATCH_SIZE = 256

        NUM_NEG = 6
        GAT_HEADS = 12
        GAT_DROPOUT = 0.25
        GAT_OUT_DIM = 300

        # Scorer, one of: {"distmult", "transe"}
        GAT_SCORER = "distmult"

    from plwordnet_ml.embedder.constants.wandb import WandbConfig as _WANDBConfig
    class WandbConfig(_WANDBConfig):
        PROJECT_NAME = "plWordnet-relgat"
        PROJECT_TAGS = ["relgat", "link-prediction"]
        PREFIX_RUN = "run_"
        BASE_RUN_NAME = "relgat"


class RelGATMainTrainerHandler:
    @staticmethod
    def load_embeddings_and_edges(
        path_to_nodes: str, path_to_rels: str, path_to_edges: str
    ):
        # node2emb – dict[int, torch.Tensor]
        print("Loading", path_to_nodes)
        with open(path_to_nodes, "rb") as f:
            _node2emb = pickle.load(f)
        _node2emb = {int(k): torch.tensor(v) for k, v in _node2emb.items()}

        # rel2idx – dict[str, int]
        print("Loading", path_to_rels)
        with open(path_to_rels, "r") as f:
            _rel2idx = json.loads(f.read())
        _rel2idx = {str(k): int(v) for k, v in _rel2idx.items()}

        # edge_index_raw – list[(src, dst, rel_str)]
        print("Loading", path_to_edges)
        with open(path_to_edges, "r") as f:
            _edge_index_raw = json.loads(f.read())
        _edge_index_raw = [
            [int(f), int(t), str(r)]
            for f, t, r in _edge_index_raw
            if f in _node2emb and t in _node2emb
        ]

        return _node2emb, _rel2idx, _edge_index_raw

    @staticmethod
    def build_trainer(
        node2emb,
        rel2idx,
        edge_index_raw,
        args: argparse.Namespace,
    ) -> RelGATTrainer:
        run_cfg = {
            "base_model": "relgat",
            "train_ratio": args.train_ratio,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "scorer": args.scorer,
            "out_dim": args.gat_out_dim,
            "num_neg": args.num_neg,
            "heads": args.heads,
            "dropout": args.dropout
        }

        trainer = RelGATTrainer(
            node2emb=node2emb,
            rel2idx=rel2idx,
            edge_index_raw=edge_index_raw,
            run_config=run_cfg,
            wandb_config=ConstantsRelGATTrainer.WandbConfig,
            train_batch_size=run_cfg["batch_size"],
            num_neg=run_cfg["num_neg"],
            train_ratio=run_cfg["train_ratio"],
            scorer_type=run_cfg["scorer"],
            gat_out_dim=run_cfg["out_dim"],
            gat_heads=run_cfg["heads"],
            dropout=run_cfg["dropout"],
            run_name=args.run_name,
            device=torch.device(args.device),
        )
        return trainer
