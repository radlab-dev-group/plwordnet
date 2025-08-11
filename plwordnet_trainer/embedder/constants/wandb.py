from dataclasses import dataclass


@dataclass
class WandbConfig:
    """
    Configuration class for Weights & Biases (wandb) integration.

    This dataclass defines the default configuration parameters for wandb
    experiment tracking and logging for embedder experiments.

    Attributes:
        PREFIX_RUN: Prefix to add to run names
        BASE_RUN_NAME: Base name for wandb runs
        PROJECT_NAME: Name of the wandb project for grouping experiments
        PROJECT_TAGS: List of tags to categorize the experiments
    """

    PREFIX_RUN = ""
    BASE_RUN_NAME = "semantic-embeddings"
    PROJECT_NAME = "plWordnet-semantic-embeddings"
    PROJECT_TAGS = ["plWordnet", "synset", "embedder", "bi-encoder"]
