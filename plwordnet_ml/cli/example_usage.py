"""
Example usage documentation for the Polish Wordnet Milvus CLI application.
"""

EXAMPLE_USAGE = f"""
Example usage:

-----------------------------------------------------------------------------------------

# Help

python plwordnet-milvus --help

-----------------------------------------------------------------------------------------

# 1. Initialize database(database, schemas, indexes, collections)

plwordnet-milvus 
    --log-level=DEBUG
    --milvus-config=resources/milvus-config.json 
    --prepare-database

-----------------------------------------------------------------------------------------

# 2. Prepare base embeddings (this step have to be done before embeddings fusion)
(or --use-database to use MySQL instead of --nx-graphs-dir)

plwordnet-milvus 
    --nx-graph-dir=path/to/networkx/graphs
    --milvus-config=resources/milvus-config.json 
    --prepare-base-embeddings 
    --device="cuda:1"


-----------------------------------------------------------------------------------------

# 3. Insert embeddings for empty lexical units (without base embeddings)
(mean embeddings for empty lexical units in case when any LU from synset is available)

plwordnet-milvus 
    --milvus-config=resources/milvus-config-pk.json 
    --nx-graph-dir=path/to/networkx/graphs
    --insert-base-mean-empty-embeddings
"""
