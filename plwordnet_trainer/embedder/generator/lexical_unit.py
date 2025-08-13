import spacy
import torch

from tqdm import tqdm
from typing import List, Dict, Iterator, Any, Optional

from plwordnet_handler.utils.logger import prepare_logger
from plwordnet_handler.base.structure.elems.lu import LexicalUnit
from plwordnet_handler.base.structure.polishwordnet import PolishWordnet
from plwordnet_trainer.embedder.generator.bi_encoder import (
    BiEncoderEmbeddingGenerator,
)
from plwordnet_trainer.embedder.generator.generator_i import _ElemGeneratorBase
from plwordnet_trainer.embedder.generator.strategy import (
    EmbeddingBuildStrategy,
    StrategyProcessor,
)


class LexicalUnitEmbeddingGenerator(_ElemGeneratorBase):
    """
    Generates embeddings for lexical unit definitions.

    This class processes lexical units from Polish WordNet and generates
    embeddings from their definitions using a provided embedding generator.
    It handles batch processing for efficient embedding generation.
    """

    def __init__(
        self,
        generator: BiEncoderEmbeddingGenerator,
        pl_wordnet: PolishWordnet,
        log_level: str = "INFO",
        log_filename: str = None,
        spacy_model_name: str = "pl_core_news_sm",
        strategy: EmbeddingBuildStrategy = EmbeddingBuildStrategy.MEAN,
    ):
        """
        Initialize the synset embedding generator.

        Args:
            generator: EmbeddingGenerator instance for creating embeddings
            pl_wordnet: PolishWordnet instance providing access
            to lexical units and synsets
            log_level: The log level to use (default: INFO).
            log_filename: The filename to save the log (default: None).
            spacy_model_name: Name of the spacy model to use (default: pl_core_news_sm)
        """

        self.generator = generator
        self.pl_wordnet = pl_wordnet
        self._emb_pos_processor = StrategyProcessor(strategy=strategy)

        self._spacy = spacy.load(spacy_model_name)

        self.logger = prepare_logger(
            logger_name=__name__, log_level=log_level, logger_file_name=log_filename
        )

    def generate(self, split_to_sentences: bool = False) -> Iterator[Dict[str, Any]]:
        """
        Generate embeddings for all lexical units with multiple processing strategies.

        Processes each lexical unit to extract text content, generates embeddings,
        and yields data using two approaches: individual text embeddings and
        processed combined embeddings.

        Args:
            split_to_sentences: Whether to split external URL descriptions into
                               individual sentences for more granular embeddings

        Yields:
            Dict[str, Any]: Dictionary containing lexical unit data, text content,
                           embeddings, and processing type information

        Note:
            Yields embeddings both for individual texts and for processed
            combinations of all texts per lexical unit
        """

        all_lexical_units = self.pl_wordnet.get_lexical_units()
        with tqdm(
            total=len(all_lexical_units),
            desc="Generating embeddings from lexical units",
        ) as pbar:
            for lu in all_lexical_units:
                possible_texts = self._get_lu_texts(
                    lu=lu, split_to_sentences=split_to_sentences
                )
                if not len(possible_texts):
                    self.logger.error(f"Lexical unit {lu} has no possible texts!")
                    continue

                embeddings = self.generator.generate_embeddings(
                    possible_texts, show_progress_bar=False, return_as_list=True
                )
                assert len(embeddings) == len(possible_texts)

                yield from self._embedding_for_each_lu_text(
                    possible_texts=possible_texts, embeddings=embeddings, lu=lu
                )
                yield from self._embedding_from_lu_processor(
                    possible_texts=possible_texts, embeddings=embeddings, lu=lu
                )

                pbar.update(1)

    @classmethod
    def _embedding_for_each_lu_text(
        cls,
        possible_texts: List[str],
        embeddings: List[torch.Tensor],
        lu: LexicalUnit,
    ):
        """
        Yield individual embeddings for each text associated with a lexical unit.

        Args:
            possible_texts: List of text strings extracted from the lexical unit
            embeddings: List of corresponding embedding vectors
            lu: LexicalUnit object being processed

        Yields:
            Dict[str, Any]: Dictionary with single text embedding data marked
                           as "single_lu_text" type
        """
        for text, emb in zip(possible_texts, embeddings):
            yield {
                "lu": lu,
                "texts": [text],
                "embedding": emb,
                "type": "single_lu_text",
            }

    def _embedding_from_lu_processor(
        self,
        possible_texts: List[str],
        embeddings: List[torch.Tensor],
        lu: LexicalUnit,
    ):
        """
        Process and yield combined embedding for all texts of a lexical unit.

        Uses the configured embedding processor to combine multiple embeddings
        into a single representation for the lexical unit.

        Args:
            possible_texts: List of text strings extracted from the lexical unit
            embeddings: List of corresponding embedding vectors
            lu: LexicalUnit object being processed

        Yields:
            Dict[str, Any]: Dictionary with processed combined embedding
            data marked with the processor's strategy type
        """
        main_embedding = self._emb_pos_processor.process(embeddings=embeddings)
        yield {
            "lu": lu,
            "text": possible_texts,
            "embedding": main_embedding,
            "type": self._emb_pos_processor.strategy,
        }

    def _get_lu_texts(
        self, lu: LexicalUnit, split_to_sentences: bool
    ) -> Optional[List[str]]:
        """
        Extract all available text content from a lexical unit's comment data.

        Collects text from various sources within the lexical unit,
        including base domain, definition, usage examples, external URL descriptions,
        and sentiment annotations. For external URL content, optionally splits
        text into individual sentences.

        Args:
            lu: LexicalUnit object containing comment data with text sources
            split_to_sentences: Whether to split external URL description
            content into individual sentences using spaCy

        Returns:
            Optional[List[str]]: List of extracted text strings, or None if no
            valid text content is found. Empty strings and whitespace-only
            content are filtered out
        """
        possible_texts = []

        # Base domain
        if lu.comment.base_domain is not None and len(
            lu.comment.base_domain.strip()
        ):
            possible_texts.append(lu.comment.base_domain.strip())

        # Definition
        if lu.comment.definition is not None and len(lu.comment.definition.strip()):
            possible_texts.append(lu.comment.definition.strip())

        # Usage examples
        for e in lu.comment.usage_examples:
            if e.text is not None and len(e.text.strip()):
                possible_texts.append(e.text.strip())

        # lu.comment.external_url_description.content
        if (
            lu.comment.external_url_description
            and lu.comment.external_url_description.content
            and len(lu.comment.external_url_description.content.strip())
        ):
            # Split to sentences?
            _t = [lu.comment.external_url_description.content.strip()]
            if split_to_sentences:
                doc = self._spacy(
                    lu.comment.external_url_description.content.strip()
                )
                _t = [
                    sent.text.strip() for sent in doc.sents if len(sent.text.strip())
                ]
            if len(_t):
                possible_texts.extend(_t)

        # Sentiment
        if lu.comment.sentiment_annotations:
            for sen_anno in lu.comment.sentiment_annotations:
                if sen_anno.example and len(sen_anno.example.strip()):
                    possible_texts.append(sen_anno.example.strip())

        return possible_texts
