class MilvusSearchFields:
    """
    Configuration class for Milvus search output fields.

    Defines the standard field sets that should be returned from Milvus queries
    for different types of WordNet embeddings, ensuring consistent data retrieval
    across different search operations.
    """

    # List of base_lu_embedding Milvus fields returned from the query
    LU_EMBEDDING_OUT_FIELDS = [
        "id",
        "lu_id",
        "embedding",
        "lemma",
        "pos",
        "domain",
        "variant",
        "model_name",
        "type",
        "strategy",
    ]

    # List of base_lu_embedding_examples Milvus fields returned from the query
    LU_EXAMPLES_OUT_FIELDS = [
        "id",
        "lu_id",
        "embedding",
        "example",
        "model_name",
        "type",
        "strategy",
    ]

    SYN_OUT_FIELDS = [
        "id",
        "syn_id",
        "embedding",
        "unitsstr",
        "model_name",
        "type",
        "strategy",
    ]
