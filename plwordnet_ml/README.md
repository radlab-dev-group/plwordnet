# plwordnet_ml

An ML-oriented module around PLWordNet:
- dataset preparation (e.g., for classification or WSD tasks),
- generation and management of embeddings,
- training scripts and evaluation utilities,
- optional integration with vector indexes (e.g., Milvus) via CLI.

## Components

- `dataset/` — tools to construct and process datasets,
- `embedder/` — components to generate vector representations,
- `training_scripts/` — example training pipelines,
- `cli/` — commands for common operations.

## Quick start

1) Activate your environment:
```bash
source .venv/bin/activate
```

2) Explore available commands:
```bash
plwordnet-milvus --help
```

3) Typical workflows:
- Generate embeddings for selected dictionary elements,
- Build a dataset from a chosen subset of synsets/relations,
- Train a model on the prepared dataset and evaluate.

## Best practices

- Start with the test graph for fast iteration, then switch to the full graph,
- Keep consistent versioning and metadata for datasets/embeddings,
- If integrating with a vector DB (e.g., Milvus), ensure proper cluster and version configuration.

See `training_scripts/` for practical examples.
