# plwordnet_handler

A module focused on:
- loading PLWordNet resources (graphs, metadata, supporting files),
- managing dictionary elements (synsets, lexical units, relations),
- searching and filtering (lemma, part of speech, relation types),
- exporting and serializing subsets of data,
- integrating local resources into the package structure.

Key capabilities:
- Works with both a lightweight test graph and a full production graph,
- CLI for common tasks (load, query, export),
- Easy handoff to plwordnet_ml (e.g., generating embeddings for lexical items).

## Resources and graphs

If you have a local resources/ directory, it can be integrated into the package. 
You may also trigger graph downloads during installation via environment variables:
- `PLWORDNET_DOWNLOAD_TEST=1` — download a test graph,
- `PLWORDNET_DOWNLOAD_FULL=1` — download a full graph.

Example:
```bash
PLWORDNET_DOWNLOAD_FULL=1 pip install -e .
```

## CLI usage

Entry point:
```bash
plwordnet-cli --help
```

Typical tasks:
- Load a graph and run queries (by lemma, relation type),
- Export subsets (reports, slices of the graph),
- Prepare inputs for ML pipelines.

## Tips

- Ensure graph resources are available (local or downloaded),
- Full graphs may require substantial memory and initialization time.

For additional usage patterns, see the CLI help and example scripts.