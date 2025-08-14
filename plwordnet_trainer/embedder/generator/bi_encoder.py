import torch

from typing import List, Dict
from sentence_transformers import SentenceTransformer

from plwordnet_handler.utils.logger import prepare_logger


class BiEncoderEmbeddingGenerator:
    """
    A class for generating text embeddings using a specified
    sentence-transformer model.
    """

    def __init__(
        self,
        model_path: str,
        model_name: str,
        device: str = "cpu",
        normalize_embeddings: bool = True,
        log_level: str = "INFO",
        log_filename: str = None,
    ):
        """
        Initializes the EmbeddingGenerator.

        Args:
            model_path: The path or name of the sentence-transformer model.
            device: The device to run the model on (e.g., 'cpu', 'cuda').
            normalize_embeddings: Whether to normalize the embeddings (default: True).
            log_level: The log level to use (default: INFO).
            log_filename: The filename to save the log (default: None).

        Raises:
            Exception: Throws an exception if the model cannot be loaded.
        """
        self.logger = prepare_logger(
            logger_name=__name__, log_level=log_level, logger_file_name=log_filename
        )
        self.normalize_embeddings = normalize_embeddings
        try:
            self.model_name = model_name
            self.model = SentenceTransformer(
                model_path, device=device, trust_remote_code=True
            )
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
        truncate_text_to_max_len: bool = False,
    ) -> List[Dict[str, torch.Tensor]] | List[torch.Tensor]:
        """
        Generates embeddings for a given list of texts.

        Args:
            texts: A list of strings to be processed.
            show_progress_bar: Whether to show a progress bar.
            return_as_list: Whether to return a list of embeddings or dict.
            truncate_text_to_max_len: Whether to truncate text (default: False).

        Returns:
            List[Dict[str, torch.Tensor]]: A list of dictionaries,
            where each dictionary contains the original 'text'
            and its 'embedding' as a PyTorch tensor.

        Raises:
            Exception: Throws an exception if an error occurs during
            embedding generation.
        """
        self.logger.debug(f"Generating embeddings for {len(texts)} texts")

        if truncate_text_to_max_len:
            texts = [t[:400] for t in texts]

        try:
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=self.normalize_embeddings,
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
