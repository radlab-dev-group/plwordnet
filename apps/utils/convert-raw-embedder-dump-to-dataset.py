"""
python3 plwordnet_trainer/embedder/apps/convert-plwn-dump-to-dataset.py
    --jsonl-path=/mnt/data2/data/datasets/radlab-semantic-embeddings/20250811/embedding-dump-ratio-1.2-w-synonymy/raw-embedding-dump-ratio-1.2-w-synonymy.jsonl
    --output-dir=/mnt/data2/data/datasets/radlab-semantic-embeddings/20250811/embedding-dump-ratio-1.2-w-synonymy/embedder_sentsplit_train-0.93
    --train-ratio=0.93
    --split-to-sentences
    --n-workers=32
    --batch-size=1000
"""

import os
import json
import spacy
import random
import argparse

from tqdm import tqdm
from functools import partial
from typing import List, Dict, Any
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_sample_batch(
    batch: List[Dict[Any, Any]],
    split_to_sentences: bool,
    spacy_model_name: str = "pl_core_news_sm",
) -> List[Dict[str, Any]]:
    """
    Process a batch of samples in a separate process.

    Args:
        batch: List of samples to process
        split_to_sentences: Whether to split text into sentences
        spacy_model_name: Name of the spacy model to use

    Returns:
        List of converted samples
    """
    # Load spaCy model in each process
    nlp = spacy.load(spacy_model_name)

    def split_text_to_sentences(text_str: str) -> List[str]:
        """Split text into sentences using spaCy."""
        if not text_str or not text_str.strip():
            return []

        doc = nlp(text_str.strip())
        sentences = [
            sent.text.strip() for sent in doc.sents if len(sent.text.strip())
        ]
        return sentences

    converted = []
    for s in batch:
        _s1 = s["text_parent"]
        _s2 = s["text_child"]

        if split_to_sentences:
            s1_list = split_text_to_sentences(_s1)
            s2_list = split_text_to_sentences(_s2)
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

    return converted


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
        n_workers: int = None,
        batch_size: int = None,
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
            n_workers: Number of parallel workers. Defaults to cpu_count()
            batch_size: Size of batches for parallel processing.
            Auto-calculated if None
        """
        self.nlp = spacy.load(spacy_model_name)
        self.jsonl_path = jsonl_path
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.split_to_sentences = split_to_sentences
        self.seed = seed
        self.spacy_model_name = spacy_model_name
        self.n_workers = n_workers or cpu_count()
        self.batch_size = batch_size

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
        Converts raw PLWN samples to standardized dataset format
        with sentence pairs and metadata using parallel processing.
        """
        if not samples:
            return []

        # Auto-calculate batch size if not provided
        if self.batch_size is None:
            # Aim for reasonable batch sizes - not too small (overhead) or too large (memory)
            self.batch_size = max(1, len(samples) // (self.n_workers * 4))
            self.batch_size = min(self.batch_size, 1000)

        batches = self.create_batches(samples)

        print(
            f"Processing {len(samples)} samples in {len(batches)} "
            f"batches using {self.n_workers} workers"
        )

        converted = []
        # Process batches in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Create a partial function with fixed arguments
            process_func = partial(
                process_sample_batch,
                split_to_sentences=self.split_to_sentences,
                spacy_model_name=self.spacy_model_name,
            )

            # Submit all batches for processing
            future_to_batch = {
                executor.submit(process_func, batch): i
                for i, batch in enumerate(batches)
            }

            # Collect results with progress bar
            with tqdm(total=len(batches), desc="Converting sample batches") as pbar:
                for future in as_completed(future_to_batch):
                    try:
                        batch_result = future.result()
                        converted.extend(batch_result)
                        pbar.update(1)
                    except Exception as exc:
                        batch_idx = future_to_batch[future]
                        print(f"Batch {batch_idx} generated an exception: {exc}")
                        pbar.update(1)

        return converted

    def create_batches(
        self, samples: List[Dict[Any, Any]]
    ) -> List[List[Dict[Any, Any]]]:
        """
        Split samples into batches for parallel processing.
        """
        return [
            samples[i : i + self.batch_size]
            for i in range(0, len(samples), self.batch_size)
        ]

    def split_text_to_sentences(self, text_str: str) -> List[str]:
        """
        Splits text into individual sentences using basic punctuation rules.

        Note: This method is kept for backward compatibility but is now
        primarily used in the main process for single-threaded operations.
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
    parser.add_argument(
        "--n-workers",
        dest="n_workers",
        type=int,
        default=None,
        help=f"Number of parallel workers (default: {cpu_count()})",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=None,
        help="Batch size for parallel processing (auto-calculated if not provided)",
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
        n_workers=args.n_workers,
        batch_size=args.batch_size,
    )
    converter.run()
