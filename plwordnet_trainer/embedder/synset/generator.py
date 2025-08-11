import torch
import logging

from tqdm import tqdm

from typing import List, Dict
from sentence_transformers import SentenceTransformer

from plwordnet_handler.base.structure.polishwordnet import PolishWordnet


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
        self,
        texts: List[str],
        show_progress_bar: bool = False,
        return_as_list: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generates embeddings for a given list of texts.

        Args:
            texts: A list of strings to be processed.
            show_progress_bar: Whether to show a progress bar.
            return_as_list: Whether to return a list of embeddings or dict.

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
            embeddings = self.model.encode(
                texts, convert_to_tensor=True, show_progress_bar=show_progress_bar
            )
            results = embeddings
            if not return_as_list:
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
    def __init__(self, generator: EmbeddingGenerator, pl_wordnet: PolishWordnet):
        self.generator = generator
        self.pl_wordnet = pl_wordnet

        self.logger = logging.getLogger(__name__)

    def run(self, batch_size: int = 400):
        all_lexical_units = self.pl_wordnet.get_lexical_units()

        with tqdm(
            total=len(all_lexical_units),
            desc="Generating embeddings from lexical units",
        ) as pbar:
            batch = []
            for lu in all_lexical_units:
                definition = lu.comment.definition
                if definition is None or not len(definition):
                    pbar.update(1)
                    continue

                if len(batch) < batch_size:
                    batch.append(definition)
                    pbar.update(1)
                    continue

                # lu.domain
                # lu.comment.base_domain
                # lu.comment.usage_examples
                # lu.comment.external_url_description.content
                # lu.comment.sentiment_annotations
                # print(lu)
                out = self.generator.generate_embeddings(
                    batch, show_progress_bar=True, return_as_list=True
                )
                assert len(out) == len(batch)

                pbar.update(len(batch))
                batch = []

            if len(batch):
                out = self.generator.generate_embeddings(
                    batch, show_progress_bar=False, return_as_list=True
                )
                pbar.update(len(out))
