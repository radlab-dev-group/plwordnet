import torch

from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

from rdl_ml_utils.utils.logger import prepare_logger

from plwordnet_ml.embedder.model_config import BiEncoderModelConfig


class BiEncoderEmbeddingGenerator:
    """
    A class for generating text embeddings using a specified
    sentence-transformer model.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        model_config: Optional[BiEncoderModelConfig] = None,
        device: str = "cpu",
        normalize_embeddings: bool = True,
        log_level: str = "INFO",
        log_filename: str = None,
    ):
        """
        Initializes the EmbeddingGenerator.

        Args:
            model_path: The path or name of the sentence-transformer model.
            Can be None if model_config is provided.
            model_name: Name identifier for the model.
            Can be None if model_config is provided.
            model_config: Optional BiEncoderModelConfig instance containing
            model configuration. Takes precedence over individual model_path
            and model_name parameters.
            device: The device to run the model on (e.g., 'cpu', 'cuda').
            normalize_embeddings: Whether to normalize the embeddings
            (default: True).
            log_level: The log level to use (default: INFO).
            log_filename: The filename to save the log (default: None).

        Raises:
            Exception: Throws an exception if the model cannot be loaded.
        """
        self.logger = prepare_logger(
            logger_name=__name__, log_level=log_level, logger_file_name=log_filename
        )
        self.normalize_embeddings = normalize_embeddings

        resolved_model_path, resolved_model_name = self._resolve_model_config(
            model_path=model_path, model_name=model_name, model_config=model_config
        )

        try:
            self.model_name = resolved_model_name
            self.model = SentenceTransformer(
                resolved_model_path, device=device, trust_remote_code=True
            )
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.device = device

            self.logger.info(
                f"Model {resolved_model_path} (name={resolved_model_name}, "
                f"dim={self.embedding_dim}) is successfully loaded to device: {self.device}"
            )
        except Exception as e:
            self.logger.error(
                f"Cannot load model: {resolved_model_path}. Error: {e}"
            )
            raise Exception(f"Cannot load model: {resolved_model_path}. Error: {e}")

    def text_to_embedding(self, text: str) -> torch.Tensor:
        return self.generate_embeddings(texts=[text], return_as_list=True)[0]

    def generate_embeddings(
        self,
        texts: List[str],
        show_progress_bar: bool = False,
        return_as_list: bool = False,
        truncate_text_to_max_len: bool = False,
    ) -> List[Dict[str, torch.Tensor] | torch.Tensor]:
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
            with torch.no_grad():
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

    def _resolve_model_config(
        self,
        model_path: Optional[str],
        model_name: Optional[str],
        model_config: Optional[BiEncoderModelConfig],
    ) -> tuple[str, str]:
        """
        Resolve the model path and name from either individual
        parameters or config object.

        Args:
            model_path: Individual model path parameter
            model_name: Individual model name parameter
            model_config: Model configuration object

        Returns:
            tuple: (resolved_model_path, resolved_model_name)

        Raises:
            ValueError: If configuration is not enough
        """
        if model_config is not None:
            if not model_config.model_path or not model_config.model_name:
                error_msg = (
                    "Model configuration is incomplete - "
                    "missing model_path or model_name"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            self.logger.info(
                f"Using model configuration: {model_config.active_model_id}"
            )
            return model_config.model_path, model_config.model_name

        if not model_path or not model_name:
            error_msg = (
                "Either provide model_config object or both "
                "model_path and model_name parameters"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info(
            f"Using individual parameters: path={model_path}, name={model_name}"
        )
        return model_path, model_name
