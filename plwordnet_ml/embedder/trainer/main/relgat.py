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

from plwordnet_ml.embedder.trainer.main.part.relgat import RelGATMainTrainerHandler
from plwordnet_ml.embedder.trainer.main.part.constants import ConstantsRelGATTrainer


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=APP_DESCRIPTION)

    # Dataset
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

    # Architecture definition, learning process
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
        "--log-every-n-steps",
        type=int,
        dest="log_every_n_steps",
        default=ConstantsRelGATTrainer.Default.LOG_EVERY_N_STEPS,
        help=f"Batch size "
        f"(default: {ConstantsRelGATTrainer.Default.LOG_EVERY_N_STEPS})",
    )
    parser.add_argument(
        "--scorer",
        type=str,
        choices=["distmult", "transe"],
        default=ConstantsRelGATTrainer.Default.GAT_SCORER,
        help="Scorer to use: distmult or transe "
        f"(default: {ConstantsRelGATTrainer.Default.GAT_SCORER})",
    )
    parser.add_argument(
        "--gat-out-dim",
        dest="gat_out_dim",
        type=int,
        default=ConstantsRelGATTrainer.Default.GAT_OUT_DIM,
        help=f"GAT output embedding dimension "
        f"(default: {ConstantsRelGATTrainer.Default.GAT_OUT_DIM})",
    )
    parser.add_argument(
        "--gat-num-layers",
        dest="gat_num_layers",
        type=int,
        default=ConstantsRelGATTrainer.Default.GAT_NUM_LAYERS,
        help=f"Number of stacked RelGAT layers "
        f"(default: {ConstantsRelGATTrainer.Default.GAT_NUM_LAYERS})",
    )
    parser.add_argument(
        "--num-neg",
        dest="num_neg",
        type=int,
        default=ConstantsRelGATTrainer.Default.NUM_NEG,
        help=f"Number of negative samples per positive "
        f"(default: {ConstantsRelGATTrainer.Default.NUM_NEG})",
    )
    parser.add_argument(
        "--heads",
        dest="heads",
        type=int,
        default=ConstantsRelGATTrainer.Default.GAT_HEADS,
        help=f"Number of GAT attention heads "
        f"(default: {ConstantsRelGATTrainer.Default.GAT_HEADS})",
    )
    parser.add_argument(
        "--dropout",
        dest="dropout",
        type=float,
        default=ConstantsRelGATTrainer.Default.GAT_DROPOUT,
        help=f"GAT dropout "
        f"(default: {ConstantsRelGATTrainer.Default.GAT_DROPOUT})",
    )
    parser.add_argument(
        "--lr",
        dest="lr",
        type=float,
        default=ConstantsRelGATTrainer.Default.LR,
        help=f"Learning rate base value (with warmup/decay) "
        f"(default: {ConstantsRelGATTrainer.Default.LR})",
    )
    parser.add_argument(
        "--lr-scheduler",
        dest="lr_scheduler",
        type=str,
        choices=["linear", "cosine", "constant"],
        default=ConstantsRelGATTrainer.Default.LR_SCHEDULER,
        help="Learning rate scheduler type: [linear, cosine, constant] "
        f"(default: {ConstantsRelGATTrainer.Default.LR_SCHEDULER})",
    )
    parser.add_argument(
        "--lr-decay",
        dest="lr_decay",
        type=float,
        default=1.0,
        help="Multiplicative factor applied to LR after warm‑up (default: 1.0 – no change)",
    )
    parser.add_argument(
        "--warmup-steps",
        dest="warmup_steps",
        default=None,
        help="Warmup steps (if omitted, computed automatically from total steps)",
    )
    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=0.0,
        help="L2‑regularization coefficient (default: 0.0)",
    )
    parser.add_argument(
        "--grad-clip-norm",
        dest="grad_clip_norm",
        type=float,
        default=None,
        help="If set, clips gradient norm to this value (default: None – no clipping)",
    )
    parser.add_argument(
        "--eval-every-n-steps",
        dest="eval_every_n_steps",
        type=int,
        default=None,
        help="If > 0: evaluation will be done every N training steps. "
        "If not given, then evaluation will be done after each epoch.",
    )
    parser.add_argument(
        "--early-stop-patience",
        dest="early_stop_patience",
        type=int,
        default=None,
        help="Number of evaluation steps without improvement "
        "after which training stops (default: None – disabled)",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # mixed precision
    parser.add_argument(
        "--use-amp",
        dest="use_amp",
        action="store_true",
        help="Enable Automatic Mixed Precision (AMP) training",
    )

    # Saving model to dir
    parser.add_argument(
        "--save-dir",
        dest="save_dir",
        type=str,
        default=ConstantsRelGATTrainer.Default.DEFAULT_TRAINER_OUT_DIR,
        help="Directory for saving checkpoints and the final model"
        f"(default {ConstantsRelGATTrainer.Default.DEFAULT_TRAINER_OUT_DIR})",
    )
    parser.add_argument(
        "--save-every-n-steps",
        dest="save_every_n_steps",
        type=int,
        default=0,
        help="If > 0: save checkpoint every N training steps "
        "in the subdirectory checkpoint-<step>",
    )
    parser.add_argument(
        "--max-checkpoints",
        dest="max_checkpoints",
        type=int,
        default=5,
        help="Maximum number of saved checkpoints (oldest are removed, default: 5)",
    )

    # Wandb/device etc.
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

    # Experimental features:
    # mask edges types
    parser.add_argument(
        "--disable-edge-type-mask",
        dest="disable_edge_type_mask",
        action="store_true",
        help="If set, the model will not mask edges by relation type",
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

    if args.save_every_n_steps is not None and args.save_every_n_steps <= 0:
        args.save_every_n_steps = None

    if args.warmup_steps is not None and len(str(args.warmup_steps).strip()):
        args.warmup_steps = int(args.warmup_steps)
    else:
        args.warmup_steps = None

    if args.eval_every_n_steps is not None and len(
        str(args.eval_every_n_steps).strip()
    ):
        args.eval_every_n_steps = int(args.eval_every_n_steps)
        if args.save_every_n_steps is not None and args.save_every_n_steps:
            if args.save_every_n_steps < args.eval_every_n_steps:
                raise ValueError(
                    "save_every_n_steps must be greater "
                    "than or equal to eval_every_n_steps"
                )
            if args.save_every_n_steps % args.eval_every_n_steps != 0:
                raise ValueError(
                    "Saving can only occur after evaluation. The number of saving "
                    "steps must be divisible by the number of evaluation "
                    "steps with no remainder."
                )
    else:
        args.eval_every_n_steps = None

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
