import argparse
import json
import torch

from plwordnet_ml.embedder.trainer.relation.relgat_trainer import RelGATTrainer


#   * node2emb   – dict[int, torch.Tensor]
#   * rel2idx    – dict[str, int]
#   * edge_index_raw – list[(src, dst, rel_str)]
def load_embeddings_and_edges(
    path_to_nodes: str, path_to_rels: str, path_to_edges: str
):
    node2emb = {}
    with open(path_to_nodes, "r") as f:
        for line in f:
            d = json.loads(line)
            node2emb[int(d["id"])] = torch.tensor(d["embedding"], dtype=torch.float)

    rel2idx = {}
    with open(path_to_rels, "r") as f:
        for line in f:
            d = json.loads(line)
            rel2idx[d["rel_name"]] = int(d["rel_id"])

    edge_index_raw = []
    with open(path_to_edges, "r") as f:
        for line in f:
            d = json.loads(line)
            edge_index_raw.append((int(d["src"]), int(d["dst"]), d["rel"]))
    return node2emb, rel2idx, edge_index_raw


parser = argparse.ArgumentParser(description="Uruchomienie RelGATTrainer")
parser.add_argument("--train-ratio", type=float, default=0.9)
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument(
    "--run-name",
    type=str,
    default=None,
    help="Nazwa runu w Weights & Biases (opcjonalnie)",
)
args = parser.parse_args()


class MyWandbConfig:
    PROJECT_NAME = "plwordnet-embedder"
    PROJECT_TAGS = ["relgat", "link-prediction"]
    PREFIX_RUN = "run_"
    BASE_RUN_NAME = "relgat"


run_cfg = {
    "base_model": "relgat",
    "train_ratio": args.train_ratio,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "scorer": "distmult",
}


node2emb, rel2idx, edge_index_raw = load_embeddings_and_edges(
    path_to_nodes="nodes.json",  # {"id": ..., "embedding": [...]}
    path_to_rels="relations.json",  # {"rel_name": ..., "rel_id": ...}
    path_to_edges="edges.json",  # {"src": ..., "dst": ..., "rel": ...}
)


trainer = RelGATTrainer(
    node2emb=node2emb,
    rel2idx=rel2idx,
    edge_index_raw=edge_index_raw,
    run_config=run_cfg,
    wandb_config=MyWandbConfig,
    train_batch_size=args.batch_size,
    num_neg=4,
    train_ratio=args.train_ratio,
    scorer_type="distmult",  # "transe"
    gat_out_dim=200,
    gat_heads=6,
    dropout=0.2,
    run_name=args.run_name,
)

trainer.train(
    epochs=args.epochs, margin=args.margin if hasattr(args, "margin") else 1.0
)
