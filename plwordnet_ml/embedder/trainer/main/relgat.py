APP_DESCRIPTION = """
RelGATTrainer runner.

This application uses dataset exported with plwordnet-milvus CLI.
If you want to use this trainer, pleas build base dataset firstly:

---

```python
plwordnet-milvus \
    --milvus-config=resources/milvus-config.json \
    --embedder-config=resources/embedder-config.json \
    --nx-graph-dir=/mnt/data2/data/resources/plwordnet_handler/20250811/slowosiec_full/nx/graphs \
    --relgat-mapping-directory=resources/aligned-dataset-identifiers/  \
    --export-relgat-dataset \
    --relgat-dataset-directory=resources/aligned-dataset-identifiers/dataset \
    --log-level="DEBUG"
```
"""

import argparse

from plwordnet_ml.embedder.trainer.main.parts.relgat import (
    ConstantsRelGATTrainer,
    RelGATMainTrainerHandler,
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=APP_DESCRIPTION)

    parser.add_argument(
        "--nodes-embeddings-path",
        type=str,
        required=True,
        help="Path to file with lexical units embeddings (nodes)",
    )
    parser.add_argument(
        "--relations-mapping",
        type=str,
        required=True,
        help="Path to file with mapping relation name to identifier (relations)",
    )
    parser.add_argument(
        "--relations-triplets",
        type=str,
        required=True,
        help="Path to file with relations triplets (edges) "
        "where the single triplet is as follow: [from_lu_idx, to_lu_idx, rel_name]",
    )

    # Optional params
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=ConstantsRelGATTrainer.Default.TRAIN_EVAL_RATIO,
        help=f"Ratio of training data "
        f"(default: {ConstantsRelGATTrainer.Default.TRAIN_EVAL_RATIO})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=ConstantsRelGATTrainer.Default.EPOCHS,
        help=f"Number of epochs "
        f"(default: {ConstantsRelGATTrainer.Default.EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=ConstantsRelGATTrainer.Default.TRAIN_BATCH_SIZE,
        help=f"Batch size "
        f"(default: {ConstantsRelGATTrainer.Default.TRAIN_BATCH_SIZE})",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Weights & Biases run name (optional)",
    )

    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default="cpu",
        help="Device to use (cpu, cuda, cuda:0) - depends on machine",
    )

    # Optional margin argument
    parser.add_argument("--margin", type=float, default=1.0)

    return parser.parse_args()


def main() -> None:
    args = get_args()
    _hdl = RelGATMainTrainerHandler

    node2emb, rel2idx, edge_index_raw = _hdl.load_embeddings_and_edges(
        path_to_nodes=args.nodes_embeddings_path,
        path_to_rels=args.relations_mapping,
        path_to_edges=args.relations_triplets,
    )

    trainer = _hdl.build_trainer(
        node2emb=node2emb,
        rel2idx=rel2idx,
        edge_index_raw=edge_index_raw,
        args=args,
    )

    trainer.train(
        epochs=args.epochs,
        margin=args.margin,
    )


if __name__ == "__main__":
    main()
