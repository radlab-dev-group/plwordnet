from tqdm import tqdm

SAMPLE_USAGE = """
python convert-plwn-dump-to-dataset.py \
    --jsonl-path resources/dataset/embedder/dataset.jsonl \
    --output-dir output/ \
    --train-ratio 0.85 
    --seed 42
"""

import os
import json
import spacy
import random
import argparse

from typing import List


class EmbedderDatasetConverter:
    """
    Converts PLWN embedder JSONL dump files into
    structured JSON datasets with train/test split.
    """

    def __init__(
        self,
        jsonl_path: str,
        output_dir: str,
        train_ratio: float,
        split_to_sentences: bool,
        seed=None,
        spacy_model_name: str = "pl_core_news_sm",
    ):
        """
        Initialize the dataset converter with configuration parameters.

        Args:
            jsonl_path: Path to the input JSONL file containing the dataset
            output_dir: Directory where the split dataset files will be saved
            train_ratio: Proportion of data to allocate to a training set
            split_to_sentences: Whether to split text data into individual sentences
            seed: Random seed for reproducible data splitting. Defaults to None
            spacy_model_name: Name of the spacy model to use.
            Defaults to "pl_core_news_sm"
        """

        self.nlp = spacy.load(spacy_model_name)

        self.jsonl_path = jsonl_path
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.split_to_sentences = split_to_sentences
        self.seed = seed

    def read_jsonl(self):
        """
        Reads and parses JSONL file line by line, returning a list of JSON objects.
        """
        samples = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                samples.append(obj)
        return samples

    def convert_samples(self, samples):
        """
        Converts raw PLWN samples to standardized dataset
        format with sentence pairs and metadata.
        """

        converted = []
        with tqdm(total=len(samples), desc="Converting samples") as pbar:
            for s in samples:
                _s1 = s["text_parent"]
                _s2 = s["text_child"]
                if self.split_to_sentences:
                    s1_list = self.split_text_to_sentences(text_str=_s1)
                    s2_list = self.split_text_to_sentences(text_str=_s2)
                else:
                    s1_list = [_s1]
                    s2_list = [_s2]

                for s1 in s1_list:
                    for s2 in s2_list:
                        converted.append(
                            {
                                "sentence1": s1,
                                "sentence2": s2,
                                "score": s["relation_weight"],
                                "split": "",  # Assign later
                            }
                        )
                pbar.update(1)
        return converted

    def split_text_to_sentences(self, text_str: str) -> List[str]:
        """
        Splits text into individual sentences using basic punctuation rules.

        Args:
            text_str: Input text string to be split into sentences

        Returns:
            List[str]: List of individual sentences, stripped of leading/trailing whitespace
        """
        if not text_str or not text_str.strip():
            return []

        doc = self.nlp(text_str.strip())
        sentences = [
            sent.text.strip() for sent in doc.sents if len(sent.text.strip())
        ]
        return sentences

    def split_samples(self, samples):
        """
        Randomly splits samples into training and test
        sets based on the specified train ratio.
        """
        if self.seed is not None:
            random.seed(self.seed)

        random.shuffle(samples)
        train_size = int(len(samples) * self.train_ratio)
        train_samples = samples[:train_size]
        test_samples = samples[train_size:]

        for s in train_samples:
            s["split"] = "train"
        for s in test_samples:
            s["split"] = "test"
        return train_samples, test_samples

    def write_json(self, data, filename):
        """
        Writes data to a JSON file in the specified
        output directory with proper UTF-8 encoding.
        """
        with open(
            os.path.join(self.output_dir, filename), "w", encoding="utf-8"
        ) as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def run(self):
        """
        Executes the complete conversion pipeline:
            load,
            convert,
            split,
            and save the dataset.
        """

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        print("Loading jsonl...")
        raw_samples = self.read_jsonl()

        print("Converting samples...")
        converted = self.convert_samples(raw_samples)

        print("Split to train test")
        train, test = self.split_samples(converted)

        print("Writing dataset...")
        self.write_json(train, "train.json")
        self.write_json(test, "test.json")

        print(f"Done. Train: {len(train)} samples, Test: {len(test)} samples.")


def parse_args():
    """
    Parses command line arguments for the PLWN dataset conversion script.
    """
    parser = argparse.ArgumentParser(
        description="Convert PLWN embedder JSONL dump"
        " to JSON dataset (train/test split)."
    )
    parser.add_argument(
        "--jsonl-path",
        dest="jsonl_path",
        required=True,
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        required=True,
        help="Output directory for train.json and test.json",
    )
    parser.add_argument(
        "--train-ratio",
        dest="train_ratio",
        default=0.85,
        type=float,
        help="Fraction of data to use for training (e.g., 0.85)",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility",
    )
    parser.add_argument(
        "--split-to-sentences",
        dest="split_to_sentences",
        action="store_true",
        default=False,
        help="Optional flag to split samples into sentences",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    converter = EmbedderDatasetConverter(
        jsonl_path=args.jsonl_path,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        split_to_sentences=args.split_to_sentences,
    )
    converter.run()
