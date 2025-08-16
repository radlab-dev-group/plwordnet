"""
Example usage documentation for the Polish Wordnet Milvus CLI application.
"""

EXAMPLE_USAGE = f"""
Example usage:

# Help
python plwordnet-milvus --help

# Initialize database(database, schemas, indexes, collections)
plwordnet-milvus 
    --log-level=DEBUG
    --milvus-config=resources/milvus-config.json 
    --prepare-database

# Prepare base embeddings (this step have to be done before embeddings fusion)
plwordnet-milvus 
    --milvus-config=resources/milvus-config.json 
    --prepare-base-embeddings 
    --device="cuda:1"
"""
