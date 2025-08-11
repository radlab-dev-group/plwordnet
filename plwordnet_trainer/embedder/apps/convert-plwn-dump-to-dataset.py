SAMPLE_USAGE = """
python convert-plwn-dump-to-dataset.py \
    --jsonl-path resources/dataset/embedder/dataset.jsonl \
    --output-dir output/ \
    --train-ratio 0.85 
    --seed 42
"""

import json
import os
import argparse
import random


class EmbedderDatasetConverter:
    """
    Converts PLWN embedder JSONL dump files into
    structured JSON datasets with train/test split.
    """

    def __init__(self, jsonl_path, output_dir, train_ratio, seed=None):
        self.jsonl_path = jsonl_path
        self.output_dir = output_dir
        self.train_ratio = train_ratio
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

    @staticmethod
    def convert_samples(samples):
        """
        Converts raw PLWN samples to standardized dataset
        format with sentence pairs and metadata.
        """

        converted = []
        for s in samples:
            converted.append(
                {
                    "sentence1": s["text_parent"],
                    "sentence2": s["text_child"],
                    "score": s["relation_weight"],
                    "split": "",  # Assign later
                }
            )
        return converted

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    converter = EmbedderDatasetConverter(
        jsonl_path=args.jsonl_path,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    converter.run()
