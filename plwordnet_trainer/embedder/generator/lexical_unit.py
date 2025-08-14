import torch
import spacy
import threading

from tqdm import tqdm
from typing import List, Dict, Iterator, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

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


class SemanticEmbeddingGenerator(_ElemGeneratorBase):
    """
    Generates embeddings for lexical unit and synset definitions.

    This class processes lexical units and synsets from Polish WordNet
    and generates embeddings from their definitions using a provided
    embedding generator. It handles batch processing
    for efficient embedding generation.
    """

    def __init__(
        self,
        generator: BiEncoderEmbeddingGenerator,
        pl_wordnet: PolishWordnet,
        log_level: str = "INFO",
        log_filename: str = None,
        spacy_model_name: str = "pl_core_news_sm",
        strategy: EmbeddingBuildStrategy = EmbeddingBuildStrategy.MEAN,
        max_workers: int = 1,
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
            max_workers: Maximum number of worker threads (default: 4)
        """

        self.generator = generator
        self.pl_wordnet = pl_wordnet
        self.max_workers = max_workers
        self.spacy_model_name = spacy_model_name

        self._emb_pos_processor = StrategyProcessor(strategy=strategy)

        self._local_spacy = threading.local()

        self.logger = prepare_logger(
            logger_name=__name__, log_level=log_level, logger_file_name=log_filename
        )

    def generate(
        self, split_to_sentences: bool = False
    ) -> Iterator[List[Dict[str, Any]]]:
        """
        Generate embeddings for all lexical units and synsets with multiple
        processing strategies using multithreading.

        Processes each lexical unit to extract text content, generates
        embeddings, and yields data using two approaches: individual
        text embeddings and processed combined embeddings.

        Args:
            split_to_sentences: Whether to split external URL descriptions
            into individual sentences for more granular embeddings

        Yields:
            Dict[str, Any]: Dictionary containing lexical unit data,
            text content, embeddings, and processing type information

        Note:
            Yields embeddings both for individual texts and for processed
            combinations of all texts per lexical unit
        """
        all_lexical_units = self.pl_wordnet.get_lexical_units()

        if self.max_workers == 1:
            yield from self.__run_single_thread(
                all_lexical_units=all_lexical_units,
                split_to_sentences=split_to_sentences,
            )
        else:
            # yield from self.__run_multithreading(
            #     all_lexical_units=all_lexical_units,
            #     split_to_sentences=split_to_sentences,
            # )
            raise NotImplementedError("Multiprocessing not yet supported")

    # def __run_multithreading(
    #     self, all_lexical_units: List, split_to_sentences: bool
    # ):
    #     lu_wo_texts = []
    #     # Use ThreadPoolExecutor for parallel processing
    #     with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
    #         with tqdm(
    #             total=len(all_lexical_units),
    #             desc="Generating embeddings from lexical units",
    #         ) as pbar:
    #             # 1. Submit tasks to the thread pool
    #             future_to_lu = {
    #                 executor.submit(
    #                     self._process_single_lu, lu, split_to_sentences
    #                 ): lu
    #                 for lu in all_lexical_units
    #             }
    #
    #             # 2. Process completed tasks as they finish
    #             for future in as_completed(future_to_lu):
    #                 lu = future_to_lu[future]
    #                 try:
    #                     result = future.result()
    #                     if result is None:
    #                         lu_wo_texts.append(lu)
    #                     else:
    #                         yield from result
    #                 except Exception as exc:
    #                     self.logger.error(f"LU {lu} generated an exception: {exc}")
    #
    #                 pbar.update(1)
    #
    #     self.logger.info("Finished generating embeddings")
    #     self.logger.info(f"Number of LUs without texts: {len(lu_wo_texts)}")

    def __run_single_thread(self, all_lexical_units: List, split_to_sentences: bool):
        lu_wo_examples = []
        with tqdm(
            total=len(all_lexical_units),
            desc="Generating embeddings from lexical units",
        ) as pbar:
            for lu in all_lexical_units:
                pbar.update(1)
                lu_embeddings = self._process_single_lu(lu, split_to_sentences)
                if lu_embeddings is None:
                    lu_wo_examples.append(lu)
                    if len(lu_wo_examples) % 1000 == 0:
                        self.logger.info(
                            f"Lexical Units without examples {len(lu_wo_examples)}"
                        )
                    continue
                yield lu_embeddings

        self.logger.info("Finished generating embeddings")
        self.logger.info(f"LUs without examples: {[l.ID for l in lu_wo_examples]}")
        self.logger.info(f"Number of LUs without examples: {len(lu_wo_examples)}")

    def _process_single_lu(
        self, lu: LexicalUnit, split_to_sentences: bool
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Process a single lexical unit and generate embeddings.

        Args:
            lu: LexicalUnit to process
            split_to_sentences: Whether to split sentences

        Returns:
            List of embedding dictionaries or None if no texts found
        """
        possible_texts = self._get_lu_texts(
            lu=lu, split_to_sentences=split_to_sentences
        )

        if not len(possible_texts):
            return None

        embeddings = self.generator.generate_embeddings(
            possible_texts,
            show_progress_bar=False,
            return_as_list=True,
            truncate_text_to_max_len=True,
        )
        assert len(embeddings) == len(possible_texts)

        results = []
        results.extend(
            self._embedding_for_each_lu_example(
                possible_texts=possible_texts, embeddings=embeddings, lu=lu
            )
        )
        results.extend(
            self._embedding_from_lu_processor(
                possible_texts=possible_texts, embeddings=embeddings, lu=lu
            )
        )
        return results

    @classmethod
    def _embedding_for_each_lu_example(
        cls,
        possible_texts: List[str],
        embeddings: List[torch.Tensor],
        lu: LexicalUnit,
    ):
        """
        Yield individual embeddings for each example associated with a lexical unit.

        Args:
            possible_texts: List of text (examples) strings extracted
            from the lexical unit
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
                "type": "lu_example",
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
            "type": "lu",
            "strategy": self._emb_pos_processor.strategy,
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
                # Use thread-local spaCy model
                spacy_model = self._get_spacy_model()
                doc = spacy_model(
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

    def _get_spacy_model(self):
        """
        Get a thread-local spaCy model instance.
        """
        if not hasattr(self._local_spacy, "spacy_model"):
            self._local_spacy.spacy_model = spacy.load(self.spacy_model_name)
        return self._local_spacy.spacy_model
