import json
import tqdm
import random
import logging

import pandas as pd
import networkx as nx

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Iterator, List, Tuple

from plwordnet_handler.base.connectors.connector_data import GraphMapperData
from plwordnet_handler.base.connectors.nx.nx_connector import PlWordnetAPINxConnector


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
    Class for converting Polish Wordnet (Słowosieć) to dataset for bi-encoder training.
    This class processes NetworkX graphs containing synsets and lexical units,
    extracts textual information from comments, and applies relation weights
    from an Excel file to prepare training data for bi-encoders.
    """

    EXCEL_REL_WEIGHTS_COLUMNS = ["ID", "name", "embedder_weight_coarse"]

    def __init__(
        self, xlsx_path: str, graph_path: str, init_converter: bool = False
    ):
        """
        Initialize converter with paths and connector.

        Args:
            xlsx_path: Path to an Excel file with relation weights
            graph_path: Path to graphs to load
            init_converter: If true, auto initialization will be performed
        """
        self.logger = logging.getLogger(__name__)

        self.xlsx_path = Path(xlsx_path)
        self.graph_path = Path(graph_path)
        self.__check_paths()

        self.wordnet_connector = PlWordnetAPINxConnector(
            nx_graph_dir=graph_path, autoconnect=True
        )

        # Mappings
        #   rel.id -> rel.name (comes from Excel file)
        self.relation_names: Dict[int, str] = {}
        #   rel.id -> rel.weight (comes from Excel file)
        self.relation_weights: Dict[int, float] = {}

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

    def extract_comments_from_edges(self) -> Iterator[EmbedderSample]:
        """
        Extract textual information from graph edges with relation weights.
        This method iterates through all edges in both synset and lexical graphs,
        extracts textual information from node comments (usage examples,
        external URL descriptions), and applies relation weights based on
        the relation ID of each edge. Creates samples with text from
        both sides of the relation.

        Yields:
            EmbedderSample: Individual samples for bi-encoder training
            with parent and child texts
        """
        for g_type, g in self.wordnet_connector.graphs.items():
            if g_type in [GraphMapperData.G_UAS]:
                self.logger.debug(f"Skipping {g_type} graph while extracting")
                continue
            yield from self._process_graph_edges(graph=g, graph_type=g_type)

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
            limit: Optional limit on number of samples to export
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

        w2examples, weights_relations = self.__positive_samples_from_g(limit=limit)
        w2examples, weights_relations = self.__align_samples_to_low(
            w2examples=w2examples,
            weights_relations=weights_relations,
            low_high_ratio=low_high_ratio,
        )

        all_examples = []
        for w, examples in w2examples.items():
            if len(examples):
                all_examples.extend(examples)

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
                    f"Successfully exported {sample_count} samples to "
                    f"{output_file} without alignment"
                )
            return True
        except Exception as e:
            self.logger.error(f"Error exporting samples to JSONL: {e}")
            return False

    def _process_graph_edges(
        self, graph: nx.MultiDiGraph, graph_type: str
    ) -> Iterator[EmbedderSample]:
        """
        Process edges of a single graph and extract
        textual information from both nodes.

        Args:
            graph: NetworkX MultiDiGraph to process
            graph_type: Type of graph ("synset" or "lexical")

        Yields:
            EmbedderSample: Individual samples extracted from
            graph edges with parent and child texts
        """
        for parent_id, child_id, edge_data in graph.edges(data=True):
            relation_id = edge_data.get("relation_id")

            if relation_id is None or relation_id not in self.relation_weights:
                self.logger.warning(
                    f"Relation ID {relation_id} not found in {graph_type} graph"
                )
                continue

            relation_weight = self.relation_weights[relation_id]
            parent_texts = list(
                self._extract_all_node_texts(graph=graph, node_id=parent_id)
            )
            child_texts = list(
                self._extract_all_node_texts(graph=graph, node_id=child_id)
            )

            yield from self._create_embedder_samples(
                parent_texts=parent_texts,
                child_texts=child_texts,
                relation_id=relation_id,
                relation_weight=relation_weight,
                parent_id=parent_id,
                child_id=child_id,
            )

    def _extract_all_node_texts(
        self, graph: nx.MultiDiGraph, node_id: int
    ) -> Iterator[NodeTextData]:
        """
        Extract all textual data from a single node.

        Args:
            graph: NetworkX graph containing the node
            node_id: ID of the node to process

        Yields:
            NodeTextData: All text samples extracted from node data
        """
        if node_id not in graph.nodes:
            return

        node_data = graph.nodes[node_id].get("data", {})
        comment = node_data.get("comment", {})

        if comment and len(comment):
            yield from self._extract_definition_text(
                comment=comment, node_id=node_id
            )

            yield from self._extract_comment_texts(comment=comment, node_id=node_id)

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
        self, comment, node_id: int
    ) -> Iterator[NodeTextData]:
        """
        Extract all textual data from a comment object:
            - usage_examples
            - external_url_descriptions
            - sentiment_annotations

        Args:
            comment: Comment (dict) from LU or Synset node data
            node_id: ID of the node
        """
        if "usage_examples" not in comment:
            return

        yield from self._extract_usage_examples_texts(
            usage_examples=comment.get("usage_examples"), node_id=node_id
        )

        yield from self._extract_external_url_description_texts(
            external_url_description=comment.get("external_url_description"),
            node_id=node_id,
        )

        yield from self._extract_sentiment_annotations_texts(
            sentiment_annotations=comment.get("sentiment_annotations"),
            node_id=node_id,
        )

    @staticmethod
    def _extract_definition_text(
        comment: dict, node_id: int
    ) -> Iterator[NodeTextData]:
        """
        Extract definition text from node data.

        Args:
            comment: Comment from LU/Synset node
            node_id: ID of the node

        Yields:
            NodeTextData: Definition extracted from node data
        """
        definition = comment.get("definition", "")
        if definition and len(definition.strip()):
            yield NodeTextData(
                text=definition.strip(), source_type="definition", node_id=node_id
            )

    @staticmethod
    def _extract_usage_examples_texts(
        usage_examples: list[dict] or None, node_id: int
    ) -> Iterator[NodeTextData]:
        """
        Extract text from usage examples.

        Args:
            usage_examples: Usage examples (dict) extracted from node data
            node_id: ID of the node

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
                        node_id=node_id,
                    )

    @staticmethod
    def _extract_external_url_description_texts(
        external_url_description: dict or None, node_id: int
    ) -> Iterator[NodeTextData]:
        """
        Extract text from external URL description.

        Args:
            external_url_description: External URL description
            node_id: ID of the node

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
                node_id=node_id,
            )

    @staticmethod
    def _extract_sentiment_annotations_texts(
        sentiment_annotations: list[dict] or None, node_id: int
    ) -> Iterator[NodeTextData]:
        """
        Extract text from sentiment annotation examples.

        Args:
            sentiment_annotations: Sentiment annotation examples
            node_id: ID of the node

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
                    node_id=node_id,
                )

    def __check_paths(self):
        if not self.xlsx_path.exists():
            self.logger.error(f"Excel weights file not found: {self.xlsx_path}")
            raise FileNotFoundError(
                f"Excel weights file not found: {self.xlsx_path}"
            )

        if not self.graph_path.exists():
            self.logger.error(f"Graph directory not found: {self.graph_path}")
            raise FileNotFoundError(f"Graph directory not found: {self.graph_path}")

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
    ) -> None or List[EmbedderSample]:
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

    def __positive_samples_from_g(
        self, limit: Optional[int] = None
    ) -> Tuple[Dict, Dict]:
        self.logger.debug("Preparing samples from 'positives'")
        sample_count = 0
        w2examples = {}
        weights_relations = {}
        samples_from_edges = list(self.extract_comments_from_edges())
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

    def __align_samples_to_low(
        self,
        w2examples,
        weights_relations,
        low_high_ratio,
    ) -> Tuple[Dict, Dict]:
        w2s = {w: len(rels) for w, rels in w2examples.items()}
        self.logger.info(
            f"Weights counts before alignment: "
            f"{json.dumps(w2s, ensure_ascii=False)}"
        )

        w2r = {w: len(rels) for w, rels in weights_relations.items()}
        self.logger.info(
            f"Num of relations in weights: " f"{json.dumps(w2r, ensure_ascii=False)}"
        )

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
