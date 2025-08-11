import torch
import logging

from typing import List, Dict, Union
from sentence_transformers import SentenceTransformer

from plwordnet_handler.base.connectors.connector_i import PlWordnetConnectorInterface


class EmbeddingGenerator:
    """
    A class for generating text embeddings using a specified
    sentence-transformer model.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initializes the EmbeddingGenerator.

        Args:
            model_path: The path or name of the sentence-transformer model.
            device: The device to run the model on (e.g., 'cpu', 'cuda').

        Raises:
            Exception: Throws an exception if the model cannot be loaded.
        """
        self.logger = logging.getLogger(__name__)
        try:
            self.model = SentenceTransformer(model_path, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.device = device

            self.logger.info(
                f"Model {model_path} (dim={self.embedding_dim}) "
                f"is successfully loaded to device: {self.device}"
            )
        except Exception as e:
            self.logger.error(f"Cannot load model: {model_path}. Error: {e}")
            raise Exception(f"Cannot load model: {model_path}. Error: {e}")

    def generate_embeddings(
        self, texts: List[str]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generates embeddings for a given list of texts.

        Args:
            texts: A list of strings to be processed.

        Returns:
            List[Dict[str, torch.Tensor]]: A list of dictionaries,
            where each dictionary contains the original 'text'
            and its 'embedding' as a PyTorch tensor.

        Raises:
            Exception: Throws an exception if an error occurs during
            embedding generation.
        """
        self.logger.debug(f"Generating embeddings for {len(texts)} texts")
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            results = [
                {"text": text, "embedding": embedding}
                for text, embedding in zip(texts, embeddings)
            ]
            self.logger.debug(
                f"Embeddings for {len(embeddings)} texts are properly generated."
            )
            return results
        except Exception as e:
            self.logger.error(f"Error during embedding generation: {e}")
            raise Exception(f"Error during embedding generation: {e}")



class SynsetEmbeddingGenerator:
    def __init__(self, generator: EmbeddingGenerator, connector: PlWordnetConnectorInterface):
        self.generator = generator
        self.connector = connector

        self.logger = logging.getLogger(__name__)

    def run(self):
        all_synsets = self.connector.get_units_and_synsets()
        for s in all_synsets:
            print(s)

