import json
import argparse
from pathlib import Path

from typing import Optional, List, Tuple, Dict


class DatasetDeduplicator:
    """
    A class to handle the deduplication of datasets from JSON files.
    """

    def __init__(self, test_path: Path, train_path: Path):
        """
        Initializes the deduplicator with paths to the test and train files.

        Args:
            test_path (Path): The path to the test JSON file.
            train_path (Path): The path to the train JSON file.
        """
        self.test_path = test_path
        self.train_path = train_path
        self.output_test_path = test_path.with_name(
            f"{test_path.stem}_deduplicated.json"
        )
        self.output_train_path = train_path.with_name(
            f"{train_path.stem}_deduplicated.json"
        )

    @staticmethod
    def _load_json_data(file_path: Path) -> Optional[List[Dict]]:
        """
        Loads data from a JSON file.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
            return None
        except json.JSONDecodeError:
            print(f"Error: The file is not a valid JSON file: {file_path}")
            return None

    @staticmethod
    def _save_json_data(file_path: Path, data: List[Dict]):
        """
        Saves data to a JSON file with indentation and UTF-8 encoding.
        """
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Data saved to file: {file_path}")

    @staticmethod
    def _perform_deduplication(
        test_data: Optional[List[Dict]], train_data: Optional[List[Dict]]
    ) -> Tuple[list, list]:
        """
        Deduplicates datasets based on specified rules.

        Rules:
        1. Uniqueness is defined by the pair (sentence1, sentence2).
        2. Deduplicates data within the test set.
        3. Deduplicates data within the train set.
        4. If a pair exists in both sets, it is kept in the test set
           and removed from the train set.
        """
        seen_pairs = set()
        deduplicated_test = []
        deduplicated_train = []

        # Step 1: Process the test dataset first, as it has priority.
        if test_data:
            for item in test_data:
                pair_key = (
                    item.get("sentence1").strip().lower()
                    + item.get("sentence2").strip().lower()
                )

                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    deduplicated_test.append(item)

        # Step 2: Process the training dataset.
        # Add only items with pairs not already seen in the test set.
        if train_data:
            for item in train_data:
                pair_key = (
                    item.get("sentence1").strip().lower()
                    + item.get("sentence2").strip().lower()
                )
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    deduplicated_train.append(item)

        return deduplicated_test, deduplicated_train

    @staticmethod
    def _remove_short_examples(
        test_data: Optional[List[Dict]],
        train_data: Optional[List[Dict]],
        min_text_length: int = 25,
    ) -> Tuple[list, list]:
        """
        Remove examples that contain short sentences.

        This helper filters out any entry whose ``sentence1`` or ``sentence2``
        fields are shorter than ``min_text_length`` characters (default is 25).
        It processes both the test and train datasets independently and
        returns new lists that contain only the examples meeting the length
        requirement.

        Args:
            test_data: A list of dictionaries representing the test set examples.
                Each dictionary should have ``sentence1`` and ``sentence2`` keys.
                If ``None`` is provided, it is treated as an empty list.
            train_data: A list of dictionaries representing the training set
                examples, with the same expected structure as ``test_data``.
                ``None`` is also treated as an empty list.
            min_text_length: Minimum number of characters required for each
                sentence. Sentences shorter than this value are discarded.

        Returns:
            Tuple[List[Dict], List[Dict]]: ``(filtered_test, filtered_train)`` —
            two lists containing the retained examples from the respective
            input datasets.
        """
        clr_test = []
        clr_train = []

        for item in test_data:
            if len(item.get("sentence1", "")) < min_text_length:
                continue
            if len(item.get("sentence2", "")) < min_text_length:
                continue
            clr_test.append(item)

        for item in train_data:
            if len(item.get("sentence1", "")) < min_text_length:
                continue
            if len(item.get("sentence2", "")) < min_text_length:
                continue
            clr_train.append(item)

        return clr_test, clr_train

    def run(self):
        """
        Executes the full deduplication process: loading, processing, and saving.
        """
        print("Loading data...")
        test_data = self._load_json_data(self.test_path)
        train_data = self._load_json_data(self.train_path)

        if test_data is None or train_data is None:
            print("Aborting due to file loading errors.")
            return

        original_test_count = len(test_data)
        original_train_count = len(train_data)
        print(f"Original number of examples in test file: {original_test_count}")
        print(f"Original number of examples in train file: {original_train_count}")
        print("-" * 30)

        print("Performing deduplication...")
        clear_test, clear_train = self._perform_deduplication(
            test_data=test_data, train_data=train_data
        )

        print("Removing short examples...")
        clear_test, clear_train = self._remove_short_examples(
            test_data=clear_test, train_data=clear_train
        )

        final_test_count = len(clear_test)
        final_train_count = len(clear_train)
        print("-" * 30)
        print("Deduplication finished.")
        print(
            f"Examples in test file after deduplication: {final_test_count} "
            f"(removed {original_test_count - final_test_count})"
        )
        print(
            f"Examples in train file after deduplication: {final_train_count} "
            f"(removed {original_train_count - final_train_count})"
        )
        print("-" * 30)

        self._save_json_data(self.output_test_path, clear_test)
        self._save_json_data(self.output_train_path, clear_train)

        print("\nProcess complete!")


def prepare_parser():
    parser = argparse.ArgumentParser(
        description="Deduplicate data in JSON files (test and train)."
    )
    parser.add_argument(
        "--test",
        type=Path,
        required=True,
        help="Path to the converted embedder dataset, test JSON file .",
    )
    parser.add_argument(
        "--train",
        type=Path,
        required=True,
        help="Path to the converted embedder dataset, train JSON file.",
    )

    return parser


def main():
    args = prepare_parser().parse_args()

    deduplicator = DatasetDeduplicator(test_path=args.test, train_path=args.train)
    deduplicator.run()


if __name__ == "__main__":
    main()
