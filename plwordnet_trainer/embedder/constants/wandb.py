from dataclasses import dataclass


@dataclass
class WandbConfig:
    PREFIX_RUN = ""
    BASE_RUN_NAME = "semantic-embeddings"
    PROJECT_NAME = "plWordnet-semantic-embeddings"
    PROJECT_TAGS = ["plWordnet", "synset", "embedder", "bi-encoder"]
