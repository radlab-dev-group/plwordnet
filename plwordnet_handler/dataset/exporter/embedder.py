import json
import tqdm
import random
import logging

import pandas as pd

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Iterator, List, Tuple

from jinja2.filters import sync_do_list

from plwordnet_handler.base.structure.polishwordnet import PolishWordnet
from plwordnet_handler.base.connectors.connector_data import GraphMapperData
from plwordnet_handler.base.structure.elems.lu_in_synset import (
    LexicalUnitAndSynsetFakeRelation,
)


@dataclass
class EmbedderSample:
    """
    Data class representing a single sample for bi-encoder training.
    """

    text_parent: str
    text_child: str
    relation_id: int
    relation_name: str
    relation_weight: float
    source_type_parent: str  # 'usage_example', 'external_url', 'definition', etc.
    source_type_child: str
    node_id_parent: int
    node_id_child: int
    parent_id: Optional[int] = None
    child_id: Optional[int] = None


@dataclass
class NodeTextData:
    """
    Data class representing textual data extracted from a single node.
    """

    text: str
    source_type: str
    node_id: int


class WordnetToEmbedderConverter:
    """
    Class for converting Polish Wordnet (Słowosieć) to dataset
    for bi-encoder training. This class processes data using a given connector.
    Based on synsets and lexical units, extracts textual information from
    comments and applies relation weights from an Excel file to prepare
    training data for bi-encoders.
    """

    EXCEL_REL_WEIGHTS_COLUMNS = ["ID", "name", "embedder_weight_coarse"]

    def __init__(
        self,
        xlsx_path: str,
        pl_wordnet: PolishWordnet,
        init_converter: bool = False,
    ):
        """
        Initialize converter with paths and connector.

        Args:
            xlsx_path: Path to an Excel file with relation weights
            pl_wordnet: Object of PolishWordnet api tu use
            init_converter: If true, auto initialization will be performed
        """
        self.pl_wordnet = pl_wordnet
        self.xlsx_path = Path(xlsx_path)

        self.__check_paths_and_raise_when_error()

        # Mapping rel.id -> rel.name (comes from an Excel file)
        self.relation_names: Dict[int, str] = {}

        #  Mapping rel.id -> rel.weight (comes from an Excel file)
        self.relation_weights: Dict[int, float] = {}

        self.logger = logging.getLogger(__name__)
        if init_converter:
            self.initialize()

    def initialize(self):
        """
        Initialize the embedder by loading relation weights
        and names from an Excel file.

        This method reads the Excel file specified by xlsx_weights_path
        and validates that it contains all required columns. Then it loads
        relation weights and names into the embedder instance.

        Raises:
            ValueError: If the Excel file is missing, any of the required
            columns are specified in EXCEL_REL_WEIGHTS_COLUMNS.
            FileNotFoundError: If the Excel file at xlsx_weights_path doesn't exist.
            pandas.errors.ParserError: If the Excel file cannot be parsed.

        Note:
            The Excel file must contain all columns specified in
            EXCEL_REL_WEIGHTS_COLUMNS for successful initialization.
        """
        self.logger.info("Initializing WordnetToEmbedderConverter...")

        df = pd.read_excel(self.xlsx_path)
        for column in self.EXCEL_REL_WEIGHTS_COLUMNS:
            if column not in df:
                raise ValueError(
                    f"Excel file must contain ["
                    f"{','.join(self.EXCEL_REL_WEIGHTS_COLUMNS)}"
                    f"] columns"
                )
        self.__load_relation_weights(df=df)
        self.__load_relation_names(df=df)

    def extract_comments_from_relations(
        self, limit: Optional[int] = None
    ) -> Iterator[EmbedderSample]:
        """
        Extracts and yields embedder samples from
        both lexical unit and synset relations.

        This generator method processes relation data from four relation types:
        lexical unit relations, synset relations, "unit and synset" mapped on
        lexical units relations with faked relations between lexical units
        (relation is inherited from the parent) and synonymy.
        Yielding EmbedderSample objects that contain comment data
        suitable for embedding processing.

        Args:
            limit: If specified, only the first `limit` elems are yielded.

        Yields:
            EmbedderSample: Sample objects containing relation comment data from
            lexical unit, synset relation, units and synset and synonymy
        """
        yield from self._embedder_samples_from_relations(
            rel_type=GraphMapperData.G_LU, limit=limit
        )
        yield from self._embedder_samples_from_relations(
            rel_type=GraphMapperData.G_SYN, limit=limit
        )
        yield from self._embedder_samples_from_relations(
            rel_type=GraphMapperData.UNIT_AND_SYNSET, limit=limit
        )
        yield from self._embedder_samples_from_relations(
            rel_type=GraphMapperData.SYNONYM, limit=limit
        )

    def export(
        self,
        output_file: str,
        limit: Optional[int] = None,
        out_type: str = "jsonl",
        low_high_ratio: float = 2.0,
    ) -> bool:
        """
        Export embedder samples to JSONL file.

        Args:
            output_file: Path to output JSONL file
            limit: Optional limit on the number of samples to export
            out_type: Optional output type
            low_high_ratio: Ratio between low and high-weighted samples

        Returns:
            bool: True if export successful, False otherwise

        Raises:
            ValueError: If out_type is not "jsonl"
        """
        if out_type not in ["jsonl"]:
            self.logger.error(
                f"Invalid output type: {out_type} only supports JSONL!"
            )
            raise ValueError(f"Invalid output type: {out_type} only supports JSONL!")

        if low_high_ratio <= 0.0:
            raise ValueError("low_high_ratio must be greater than 0.0")

        self.logger.info("Starting sample extraction and export...")

        w2examples, weights_relations = self.__positive_samples_from_connector(
            limit=limit
        )
        w2examples, weights_relations = self.__align_negatives_samples_to_low(
            w2examples=w2examples,
            weights_relations=weights_relations,
            low_high_ratio=low_high_ratio,
        )

        all_examples = []
        for w, examples in w2examples.items():
            if len(examples):
                all_examples.extend(examples)

        return self.__export_to_out_file(
            output_file=output_file, all_examples=all_examples
        )

    def _embedder_samples_from_relations(
        self, rel_type: str, limit: Optional[int] = None
    ) -> Iterator[EmbedderSample]:
        """
        Generates embedder training samples from relations of a specified type.

        This private method retrieves all relations of the given type
        (synset, lexical unit, "units and synset", synonymy) from the api,
        extracts text data from parent and child elements, and creates embedder
        samples with appropriate relation weights. It processes each relation
        by fetching text content from both parent and child nodes and generating
        training samples that capture the semantic relationship between them.

        Args:
            rel_type (str): The type of relation (synset, lexical unit,
            lexical "unit and synset", synonymy)
            limit: If specified, only the first `limit` elems are yielded.

        Yields:
            EmbedderSample: Training samples containing parent-child
            text pairs with relation metadata and weights

        Note:
            The method skips relations that don't have valid IDs
            or aren't found in the configured relation weights.
            It extracts all available text data from both parent
            and child elements before creating the final embedder samples.
        """
        extract_type = rel_type
        if rel_type == GraphMapperData.G_SYN:
            all_relations = self.pl_wordnet.get_synset_relations(limit=limit)
        elif rel_type == GraphMapperData.G_LU:
            all_relations = self.pl_wordnet.get_lexical_relations(limit=limit)
        elif rel_type == GraphMapperData.UNIT_AND_SYNSET:
            # When unit and synset are used, then data is stored into LU
            extract_type = GraphMapperData.G_LU
            all_relations = list(self.__lu_in_syn_as_fake_relations(limit=limit))
        elif rel_type == GraphMapperData.SYNONYM:
            # When synonymy is used, then data is stored into LU
            extract_type = GraphMapperData.G_LU
            all_relations = list(self.__lu_in_synonymy_relation(limit=limit))
        else:
            return

        r_count = len(all_relations)
        self.logger.info(
            f"{r_count} {rel_type} relations will be used to generate data"
        )

        with tqdm.tqdm(total=r_count, desc=f"Preparing {rel_type} examples") as pbar:
            for relation in all_relations:
                pbar.update(1)

                relation_id = relation.REL_ID
                if relation_id is None or relation_id not in self.relation_weights:
                    self.logger.warning(
                        f"Relation ID {relation_id} not found in {rel_type}"
                    )
                    continue

                parent_id = relation.PARENT_ID
                child_id = relation.CHILD_ID
                relation_weight = self.relation_weights[relation_id]

                parent_texts = []
                p_texts = self.__extract_all_texts_from(
                    elem_type=extract_type, elem_id=parent_id
                )
                if p_texts is not None:
                    parent_texts = list(p_texts)

                child_texts = []
                ch_texts = self.__extract_all_texts_from(
                    elem_type=extract_type, elem_id=child_id
                )
                if ch_texts is not None:
                    child_texts = list(ch_texts)

                yield from self._create_embedder_samples(
                    parent_texts=parent_texts,
                    child_texts=child_texts,
                    relation_id=relation_id,
                    relation_weight=relation_weight,
                    parent_id=parent_id,
                    child_id=child_id,
                )

    def __lu_in_syn_as_fake_relations(
        self, limit: int
    ) -> Iterator[LexicalUnitAndSynsetFakeRelation]:
        """
        Generate fake relations between lexical units based on synset relationships.

        This method creates artificial relations between lexical units
        by leveraging existing synset relationships. For each synset relation,
        it generates relations between all lexical units in the parent synset
        and all lexical units in the child synset, preserving the original
        relation type.

        Args:
            limit: Maximum number of synset relations to process

        Yields:
            LexicalUnitAndSynsetFakeRelation: Fake relation objects connecting
            lexical units through synset relationships
        """
        # Mapping of synset to the list of lexical units
        syn_lu_ids = self.pl_wordnet.get_units_and_synsets(
            limit=limit, return_mapping=True
        )
        syn_relations = self.pl_wordnet.get_synset_relations(limit=limit)
        for sr in syn_relations:
            lus_child = syn_lu_ids.get(sr.CHILD_ID, [])
            lus_parent = syn_lu_ids.get(sr.PARENT_ID, [])
            for lu_child in lus_child:
                for lu_parent in lus_parent:
                    yield LexicalUnitAndSynsetFakeRelation(
                        PARENT_ID=lu_parent,
                        CHILD_ID=lu_child,
                        REL_ID=sr.REL_ID,
                    )

    def __lu_in_synonymy_relation(
        self, limit: int
    ) -> Iterator[LexicalUnitAndSynsetFakeRelation]:
        """
        Generate synonymy relations between lexical units within the same synsets.

        Creates fake relations representing synonymy between all pairs of lexical units
        that belong to the same synset. Each lexical unit is connected to every other
        lexical unit in its synset with a synonymy relation type (REL_ID = 30).

        Args:
            limit: Maximum number of unit-synset relationships to process

        Yields:
            LexicalUnitAndSynsetFakeRelation: Fake relation objects
            with synonymy relation type (REL_ID=30) connecting
            lexical units within the same synsets
        """
        syn_lu_ids = self.pl_wordnet.get_units_and_synsets(
            limit=limit, return_mapping=True
        )
        for synonyms in syn_lu_ids.values():
            for lu_parent in synonyms:
                for lu_child in synonyms:
                    yield LexicalUnitAndSynsetFakeRelation(
                        PARENT_ID=lu_parent,
                        CHILD_ID=lu_child,
                        REL_ID=30,  # 30 - synonymy
                    )

    def __extract_all_texts_from(
        self, elem_type: str, elem_id: int
    ) -> Iterator[NodeTextData]:
        """
        Extracts all available text data from a specified element.

        This private method retrieves either a synset or lexical unit based on the
        element type and ID, then extracts various text components (definitions and
        comments) from the element's comment data. It yields NodeTextData objects
        for each piece of extracted text.

        Args:
            elem_type (str): The type of element to extract from (synset
            or lexical unit)
            elem_id (int): The unique identifier of the element

        Yields:
            NodeTextData: Text data objects containing extracted definitions
            and comments from the specified element

        Note:
            The method handles both synset and lexical unit types by fetching the
            appropriate object and processing its comment data to extract meaningful
            text content for further processing.
        """

        elem_obj = None
        if elem_type == GraphMapperData.G_SYN:
            elem_obj = self.pl_wordnet.get_synset(syn_id=elem_id)
        elif elem_type == GraphMapperData.G_LU:
            elem_obj = self.pl_wordnet.get_lexical_unit(lu_id=elem_id)

        comment = (
            elem_obj.comment.as_dict()
            if elem_obj is not None and elem_obj.comment is not None
            else None
        )
        if comment and len(comment):
            yield from self._extract_definition_text(
                comment=comment, elem_id=elem_id
            )
            yield from self._extract_comment_texts(comment=comment, elem_id=elem_id)

    def _create_embedder_samples(
        self,
        parent_texts: List[NodeTextData],
        child_texts: List[NodeTextData],
        relation_id: int,
        relation_weight: float,
        parent_id: int,
        child_id: int,
    ) -> Iterator[EmbedderSample]:
        """
        Create EmbedderSample instances by combining parent and child texts.

        Args:
            parent_texts: List of texts extracted from parent node
            child_texts: List of texts extracted from child node
            relation_id: ID of the relation
            relation_weight: Weight of the relation
            parent_id: Parent node ID
            child_id: Child node ID

        Yields:
            EmbedderSample: Samples with combinations of parent and child texts
        """
        # If no texts available, skip this edge
        if not parent_texts or not child_texts:
            return

        relation_name = self.relation_names.get(relation_id, "unknown")

        # Create samples by combining each parent text with each child text
        for parent_text_data in parent_texts:
            for child_text_data in child_texts:
                yield EmbedderSample(
                    text_parent=parent_text_data.text,
                    text_child=child_text_data.text,
                    relation_id=relation_id,
                    relation_name=relation_name,
                    relation_weight=relation_weight,
                    source_type_parent=parent_text_data.source_type,
                    source_type_child=child_text_data.source_type,
                    node_id_parent=parent_text_data.node_id,
                    node_id_child=child_text_data.node_id,
                    parent_id=parent_id,
                    child_id=child_id,
                )

    def _extract_comment_texts(
        self, comment, elem_id: int
    ) -> Iterator[NodeTextData]:
        """
        Extract all textual data from a comment-as-dict:
            - usage_examples
            - external_url_descriptions
            - sentiment_annotations

        Args:
            comment: Comment (dict) from LU or Synset node data
            elem_id: ID of the elem (SYN/LU)
        """
        if "usage_examples" not in comment:
            return

        yield from self._extract_usage_examples_texts(
            usage_examples=comment.get("usage_examples"), elem_id=elem_id
        )

        yield from self._extract_external_url_description_texts(
            external_url_description=comment.get("external_url_description"),
            elem_id=elem_id,
        )

        yield from self._extract_sentiment_annotations_texts(
            sentiment_annotations=comment.get("sentiment_annotations"),
            elem_id=elem_id,
        )

    @staticmethod
    def _extract_definition_text(
        comment: dict, elem_id: int
    ) -> Iterator[NodeTextData]:
        """
        Extract definition text from comment-as-dict.

        Args:
            comment: Comment from LU/Synset node
            elem_id: ID of the elem (LU/Synset)

        Yields:
            NodeTextData: Definition extracted from node data
        """
        definition = comment.get("definition", "")
        if definition and len(definition.strip()):
            yield NodeTextData(
                text=definition.strip(), source_type="definition", node_id=elem_id
            )

    @staticmethod
    def _extract_usage_examples_texts(
        usage_examples: Optional[list[dict]], elem_id: int
    ) -> Iterator[NodeTextData]:
        """
        Extract text from usage examples.

        Args:
            usage_examples: Usage examples (dict) extracted from LU/Synset
            elem_id: ID of the elem (LU/Synset)

        Yields:
            NodeTextData: Text samples extracted from usage examples
        """
        if not usage_examples:
            return

        for usage_example in usage_examples:
            if len(usage_example):
                usage_example_txt = usage_example.get("text", "")
                if usage_example_txt and len(usage_example_txt.strip()):
                    yield NodeTextData(
                        text=usage_example_txt.strip(),
                        source_type=f"usage_example_"
                        f"{usage_example.get('source_pattern', '?')}",
                        node_id=elem_id,
                    )

    @staticmethod
    def _extract_external_url_description_texts(
        external_url_description: Optional[dict], elem_id: int
    ) -> Iterator[NodeTextData]:
        """
        Extract text from the external URL description.

        Args:
            external_url_description: External URL description
            elem_id: ID of the elem (LU/Synset)

        Yields:
            NodeTextData: Text samples extracted from external URL
        """
        if not external_url_description:
            return

        content = external_url_description.get("content", "")
        if content and len(content.strip()):
            yield NodeTextData(
                text=content.strip(),
                source_type="external_url_content",
                node_id=elem_id,
            )

    @staticmethod
    def _extract_sentiment_annotations_texts(
        sentiment_annotations: Optional[list[dict]], elem_id: int
    ) -> Iterator[NodeTextData]:
        """
        Extract text from sentiment annotation examples.

        Args:
            sentiment_annotations: Sentiment annotation examples
            elem_id: ID of the elem (LU/Synset)

        Yields:
            NodeTextData: Text samples extracted from sentiment annotation examples
        """
        if not sentiment_annotations:
            return

        for sent_anno in sentiment_annotations:
            example_str = sent_anno.get("example", "")
            if example_str and len(example_str.strip()):
                yield NodeTextData(
                    text=example_str.strip(),
                    source_type="sentiment_example",
                    node_id=elem_id,
                )

    def __check_paths_and_raise_when_error(self):
        if not self.xlsx_path.exists():
            self.logger.error(f"Excel weights file not found: {self.xlsx_path}")
            raise FileNotFoundError(
                f"Excel weights file not found: {self.xlsx_path}"
            )

    def __load_relation_weights(self, df: pd.DataFrame) -> None:
        """
        Load relation weights from an Excel file and create ID to weight mapping.
        Expected columns in Excel:
        - ID: relation identifier
        - embedder_weight_coarse: weight value for the relation

        Args:
            df: Pandas DataFrame with relation weights
        """
        self.logger.info("Loading relation weights...")

        self.relation_weights = dict(zip(df["ID"], df["embedder_weight_coarse"]))
        self.logger.info(f"Loaded {len(self.relation_weights)} relation weights")

    def __load_relation_names(self, df: pd.DataFrame) -> None:
        """
        Load relation names mapping from the Excel file.
        Expected columns in Excel:
        - ID: relation identifier
        - name: relation name

        Args:
            df: Pandas DataFrame with relation names
        """
        self.logger.info("Loading relation names...")
        self.relation_names = dict(zip(df["ID"], df["name"]))
        self.logger.info(f"Loaded {len(self.relation_names)} relation names")

    def __align_relations__low(
        self,
        w2s: Dict[float, int],
        w2examples: Dict[float, List[EmbedderSample]],
        cut_weight: float = 0.5,
        low_to_high_ratio: float = 2.0,
    ) -> Optional[List[EmbedderSample]]:
        """
        Aligns low-weighted relation examples with high-weighted ones
        to create balanced training data.

        This private method balances the dataset by pairing low-weighted examples
        with randomly selected high-weighted examples to generate new training
        samples. The goal is to create a more balanced representation where
        low-weighted relationships are contrasted against high-weighted
        ones with a specified ratio.

        Args:
            w2s (Dict[float, int]): Dictionary mapping weights to sample counts
            w2examples (Dict[float, List[EmbedderSample]]): Dictionary mapping
            weights to example lists
            cut_weight (float, optional): Threshold weight separating low from
             high examples. Defaults to 0.5.
            low_to_high_ratio (float, optional): Ratio of high-weighted examples
            to pair with each low-weighted example. Defaults to 2.0.

        Returns:
            Optional[List[EmbedderSample]]: List of new aligned training samples,
            or None if alignment conditions are not met (insufficient ratios
            or empty example sets).

        Note:
            The method shuffles examples randomly and creates negative
            samples by pairing low-weighted examples with high-weighted ones,
            helping to improve model training through better class balance
            and contrastive learning.
        """

        self.logger.info(
            f"Aligning low-weighted examples with ratio "
            f"{low_to_high_ratio} to high-weighted examples..."
        )

        l_h_ratio, new_examples_count, l_examples, h_examples = (
            self.__low_weighted_ratio(
                w2s=w2s,
                w2examples=w2examples,
                cut_weight=cut_weight,
                low_to_high_ratio=low_to_high_ratio,
            )
        )

        if l_h_ratio < 0.001 or new_examples_count < 0.001:
            self.logger.warning(
                f"Too small l_h_ratio: {l_h_ratio} or "
                f"too small new_examples_count: {new_examples_count}"
            )
            return None

        if not len(l_examples) or not len(h_examples):
            self.logger.warning(
                f"There are no low_examples: {len(l_examples)} "
                f"or not high_examples: {len(h_examples)}"
            )
            return None

        all_h_examples = []
        for samples in h_examples.values():
            all_h_examples.extend(samples)
        random.shuffle(all_h_examples)

        all_l_examples = []
        for samples in l_examples.values():
            all_l_examples.extend(samples)
        random.shuffle(all_l_examples)

        new_examples = []
        for l_e in all_l_examples:
            _h_examples = random.sample(all_h_examples, l_h_ratio)
            _e = self.__merge_to_embedder_sample(
                example=l_e, negatives=_h_examples, low_value=0.05
            )
            if len(_e):
                new_examples.extend(_e)
        return new_examples[:new_examples_count]

    def __merge_to_embedder_sample(
        self,
        example: EmbedderSample,
        negatives: List[EmbedderSample],
        low_value: float = 0.05,
    ) -> List[EmbedderSample]:
        """
        Merges a positive example with negative samples to create
        low-weighted training samples.

        This private method takes a positive embedder sample and combines
        it with a list of negative samples to generate new training examples.
        The resulting samples inherit the parent information from the positive
        example while using the child information from each negative sample.
        All generated samples are assigned a low-related weight to indicate
        they represent negative or weak relationships.

        Args:
            example (EmbedderSample): The positive example that
            provides parent context
            negatives (List[EmbedderSample]): List of negative samples that
            provide child context
            low_value (float, optional): The low weight value assigned
            to negative relationships. Defaults to 0.05.

        Returns:
            List[EmbedderSample]: A list of new embedder samples where each
            combines the parent from the positive example with a child from
            the negative samples, all assigned the specified low weight value.

        Note:
            This method is typically used in contrastive learning
            scenarios where positive examples need to be contrasted
            with negative examples during model training.
        """

        examples = []
        self.logger.debug(
            f"Preparing negative low-weighted embedder "
            f"samples for {example.node_id_parent}..."
        )

        for n in negatives:
            examples.append(
                EmbedderSample(
                    text_parent=example.text_parent,
                    text_child=n.text_child,
                    relation_id=example.relation_id,
                    relation_name=example.relation_name,
                    relation_weight=low_value,
                    source_type_parent=example.source_type_parent,
                    source_type_child=n.source_type_child,
                    node_id_parent=example.node_id_parent,
                    node_id_child=n.node_id_child,
                    parent_id=example.parent_id,
                    child_id=n.child_id,
                )
            )
        return examples

    def __low_weighted_ratio(
        self,
        w2s: Dict[float, int],
        w2examples: Dict[float, List[EmbedderSample]],
        cut_weight: float,
        low_to_high_ratio: float,
    ) -> Tuple[
        int,
        int,
        Dict[float, List[EmbedderSample]],
        Dict[float, List[EmbedderSample]],
    ]:
        """
        Calculate ratio and distribution for balancing
        low-weighted and high-weighted examples.

        This method separates examples into low-weighted and high-weighted
        groups based on the `cut_weight` threshold, then calculates how many
        additional examples are needed to maintain the desired low_to_high_ratio.

        Args:
            w2s: Dictionary mapping weights to sample counts
            w2examples: Mapping weights to lists of EmbedderSample objects
            cut_weight: Threshold to separate low from high-weighted examples
            low_to_high_ratio: Ratio of low-weighted to high-weighted examples

        Returns:
            Tuple containing:
            - l_h_ratio: Ratio of high-weighted examples
            to add per low-weighted example
            - add_examples: Number of additional low-weighted examples to add
            - l_examples: Dictionary of low-weighted examples
            - h_examples: Dictionary of high-weighted examples

        Raises:
            Exception: If the ratio is too small for the number of examples,
            and there's no implementation for removing high-weighted examples.
        """

        h_examples = {}
        l_examples = {}
        for w, res in w2s.items():
            if w >= cut_weight:
                h_examples[w] = w2examples[w]
            else:
                l_examples[w] = w2examples[w]
        h_count = sum(len(e) for e in h_examples.values())
        l_count = sum(len(e) for e in l_examples.values())

        if l_count > h_count * low_to_high_ratio:
            raise Exception(
                "Ratio is too small for the number of examples. "
                "There is no implementation for removing high-weighted examples."
            )

        self.logger.info(f"  - high-weighted examples: {h_count}")
        self.logger.info(f"  - low-weighted examples: {l_count}")

        add_examples = (
            int(low_to_high_ratio * (l_count * (h_count / l_count))) - l_count
        )
        self.logger.info(f"  + {add_examples} low-weighted examples will be added")

        l_h_ratio = round(0.5 + add_examples / l_count)
        if l_h_ratio > 0:
            total_examples = l_count * l_h_ratio
            self.logger.info(
                f"For single low-weighted examples {l_h_ratio} "
                f"high-weighted examples will be added. "
            )

            if total_examples > add_examples:
                self.logger.info(
                    f"After that number of examples will equal {total_examples} "
                    f"and at the end will be cut to {add_examples} examples."
                )

        return l_h_ratio, add_examples, l_examples, h_examples

    def __positive_samples_from_connector(
        self, limit: Optional[int] = None
    ) -> Tuple[Dict, Dict]:
        """
        Extracts positive training samples from relation data via the connector.

        This method processes relation comments to create positive examples
        for training, organizing them by relation weight and tracking unique
        relation IDs. Progress is displayed via a progress bar
        during sample extraction.

        Args:
            limit (Optional[int]): Maximum number of samples
            to extract (None for no limit)

        Returns:
            Tuple[Dict, Dict]: A tuple containing:
                - w2examples: Dictionary mapping relation weights
                to lists of samples
                - weights_relations: Dictionary mapping relation weight
                to sets of relation IDs

        Note:
            Samples are grouped by relation weight to facilitate balanced training
            and prevent duplicate relation processing within each weight category.
        """
        self.logger.debug("Preparing samples from 'positives'")

        sample_count = 0
        w2examples = {}
        weights_relations = {}
        samples_from_edges = list(self.extract_comments_from_relations(limit=limit))
        with tqdm.tqdm(
            total=len(samples_from_edges), desc="Preparing positives"
        ) as pbar:
            for sample in samples_from_edges:
                if limit is not None and sample_count >= limit:
                    break

                if sample.relation_weight not in w2examples:
                    w2examples[sample.relation_weight] = []
                    weights_relations[sample.relation_weight] = set()
                w2examples[sample.relation_weight].append(sample)
                weights_relations[sample.relation_weight].add(sample.relation_id)
                sample_count += 1

                pbar.update(1)
        return w2examples, weights_relations

    def __align_negatives_samples_to_low(
        self,
        w2examples,
        weights_relations,
        low_high_ratio,
    ) -> Tuple[Dict, Dict]:
        """
        Balances negative samples by generating additional
        low-weight examples to match the ratio.

        This method addresses class imbalance in relation to weight distribution
        by creating negative samples for underrepresented low-weight categories.
        It logs the distribution before and after alignment to track
        the balancing process.

        Args:
            w2examples: Dictionary mapping relation weights to example lists
            weights_relations: Dictionary mapping weights to sets of relation IDs
            low_high_ratio: Target ratio of low-weight to high-weight samples

        Returns:
            Tuple[Dict, Dict]: Updated `w2examples` and weights_relations
            dictionaries containing the newly balanced sample distributions

        Note:
            Uses a cut_weight threshold of 0.5 to determine what constitutes
            low-weight relations. Generated samples are added to existing
            collections and logged for verification.
        """
        self.logger.debug("Preparing 'negative' samples based on the 'positives'")

        w2s = {w: len(rels) for w, rels in w2examples.items()}
        self.logger.info(
            f"Weights counts before alignment: "
            f"{json.dumps(w2s, ensure_ascii=False)}"
        )

        w2r = {w: len(rels) for w, rels in weights_relations.items()}
        self.logger.info(
            f"Num of relations in weights: " f"{json.dumps(w2r, ensure_ascii=False)}"
        )
        self.logger.info(weights_relations)

        low_examples = self.__align_relations__low(
            w2s=w2s,
            w2examples=w2examples,
            cut_weight=0.5,
            low_to_high_ratio=low_high_ratio,
        )

        if low_examples is not None:
            for sample in low_examples:
                if sample.relation_weight not in w2examples:
                    w2examples[sample.relation_weight] = []
                    weights_relations[sample.relation_weight] = set()
                w2examples[sample.relation_weight].append(sample)
                weights_relations[sample.relation_weight].add(sample.relation_id)

        w2s = {w: len(rels) for w, rels in w2examples.items()}
        self.logger.info(
            f"Weights counts after alignment: "
            f"{json.dumps(w2s, ensure_ascii=False)}"
        )
        return w2examples, weights_relations

    def __export_to_out_file(
        self, output_file: str, all_examples: List[EmbedderSample]
    ):
        """
        Exports embedder samples to a JSONL output file.

        This private method serializes a list of EmbedderSample objects and
        writes them to a specified output file in JSON Lines format.
        Each sample is converted to a dictionary containing all its attributes
        and written as a separate JSON object on each line. The method ensures
        the output directory exists and handles encoding properly
        for non-ASCII characters.

        Args:
            output_file (str): Path to the output file where samples will be exported
            all_examples (List[EmbedderSample]): List of embedder samples to export

        Returns:
            bool: True if export was successful, False if an error occurred
            during export

        Note:
            The method creates parent directories if they don't exist
            and uses UTF-8 encoding with ensure_ascii=False to preserve
            non-ASCII characters in the exported data. Each sample is written
            as one JSON object per line (JSONL format).
        """
        try:
            sample_count = 0
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for sample in all_examples:
                    sample_dict = {
                        "text_parent": sample.text_parent,
                        "text_child": sample.text_child,
                        "relation_id": sample.relation_id,
                        "relation_name": sample.relation_name,
                        "relation_weight": sample.relation_weight,
                        "source_type_parent": sample.source_type_parent,
                        "source_type_child": sample.source_type_child,
                        "node_id_parent": sample.node_id_parent,
                        "node_id_child": sample.node_id_child,
                        "parent_id": sample.parent_id,
                        "child_id": sample.child_id,
                    }
                    f.write(json.dumps(sample_dict, ensure_ascii=False) + "\n")
                    sample_count += 1

                self.logger.info(
                    f"Successfully exported {sample_count} samples to {output_file}"
                )
            return True
        except Exception as e:
            self.logger.error(f"Error exporting samples to JSONL: {e}")
            return False
