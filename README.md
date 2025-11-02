# PLWordNet

PLWordNet (Handler/ML) is a toolkit for working with the Polish semantic lexicon (Słowosieć/PLWordNet). 
It provides:
- loading dictionary data from resources/graphs,
- management of dictionary elements (synsets, lexical units, relations),
- searching and exporting data,
- ML-oriented utilities (embeddings, datasets, training scripts).

The project contains two main modules:
- `plwordnet_handler` — access, management, and operations over PLWordNet data,
- `plwordnet_ml` — utilities for ML workflows leveraging PLWordNet.

Module documentation:
- Handler module: [plwordnet_handler](plwordnet_handler/README.md)
- ML module: [plwordnet_ml](plwordnet_ml/README.md)

## Installation

> **Prerequisite**: `radlab-ml-utils`
>
> This project uses the 
> [radlab-ml-utils](https://github.com/radlab-dev-group/ml-utils) 
> library for machine learning utilities 
> (e.g., experiment/result logging with Weights & Biases/wandb).
> Install it before working with ML-related parts:
>
> ```bash
> pip install git+https://github.com/radlab-dev-group/ml-utils.git
> ```
>
> For more options and details, see the library README: 
> https://github.com/radlab-dev-group/ml-utils


Use a virtual environment (virtualenv) for isolation.
``` bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -e .
```

Optional graph download during installation via environment variables:
- `PLWORDNET_DOWNLOAD_TEST=1` — downloads a small test graph,
- `PLWORDNET_DOWNLOAD_FULL=1` — downloads the full production graph.

Example:
``` bash
PLWORDNET_DOWNLOAD_TEST=1 pip install -e .
```

## Quick start (CLI)

The package includes CLI apps:
- `plwordnet-cli` — main dictionary operations (loading, querying, exporting),
- `plwordnet-milvus` — operations for vector indexes (e.g., Milvus).

Show help:
```
bash
plwordnet-cli --help
plwordnet-milvus --help
```
Learn more:
- plwordnet_handler/README.md — resources management, graphs, queries, exports,
- plwordnet_ml/README.md — embeddings, datasets, training workflows.

## Reproduce a whole process
If you want to reproduce the whole process of creating from scratch, look inside [INSTALL.md](INSTALL.md)

## License

[LICENSE](LICENSE).
