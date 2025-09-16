"""
python3 plwordnet_ml/embedder/apps/convert-plwn-dump-to-dataset.py
    --jsonl-path=/mnt/data2/data/datasets/radlab-semantic-embeddings/20250811/embedding-dump-ratio-1.2-w-synonymy/raw-embedding-dump-ratio-1.2-w-synonymy.jsonl
    --output-dir=/mnt/data2/data/datasets/radlab-semantic-embeddings/20250811/embedding-dump-ratio-1.2-w-synonymy/embedder_sentsplit_train-0.93
    --train-ratio=0.93
    --split-to-sentences
    --n-workers=32
    --batch-size=1000
"""

import os
import json
import datetime

import spacy
import random
import argparse
import threading

from tqdm import tqdm
from functools import partial
from multiprocessing import cpu_count
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from rdl_ml_utils.handlers.prompt_handler import PromptHandler
from rdl_ml_utils.handlers.openapi_handler import OpenAPIClient

GLOBAL_CORRECTED_TEXTS = {}
GLOBAL_LOCK_CORRECTED_TEXTS = threading.Lock()


def correct_if_necessary(
    text_str: str, open_api: OpenAPIClient, prompt_str: str
) -> str | None:
    """
    Return a corrected version of *text_str* using the provided OpenAPI client
    and prompt, caching results to avoid duplicate API calls.

    The function performs the following steps:

    1. **Input validation** – If *text_str* is `None` or an empty string,
       the original value is returned unchanged.
    2. **Cache lookup** – A global, thread‑safe cache (`GLOBAL_CORRECTED_TEXTS`)
       is consulted under `GLOBAL_LOCK_CORRECTED_TEXTS`.  If a corrected
       version of *text_str* is already stored, that cached value is returned
       immediately.
    3. **API generation** – When the text is not cached, the function calls
       `open_api.generate` with *text_str* as the message, *prompt_str* as the
       system prompt, and a `max_tokens` limit set to twice the length of the
       input text.  This yields a corrected string.
    4. **Fallback handling** – If the API returns `None` or an empty/whitespace‑only
       string, the original *text_str* is used as the corrected result.
    5. **Cache insertion** – The (potentially corrected) result is stored in the
       global cache under the original *text_str* key, again protected by the
       lock, so subsequent calls can reuse it.
    6. **Return value** – The corrected (or original) string is returned.  The
       return type is `str`; `None` is only possible when the caller
       explicitly passes `None` as *text_str*.

    Parameters
    ----------
    text_str : str
        The raw text that may need correction.  `None` or an empty string is
        treated as a no‑op and returned unchanged.
    open_api : OpenAPIClient
        An instantiated client capable of generating corrected text via its
        `generate` method.
    prompt_str : str
        The system prompt that guides the correction performed by the OpenAPI
        model.

    Returns
    -------
    str | None
        The corrected text if a correction was performed, otherwise the
        original *text_str*.  `None` is returned only when the input was
        `None`.

    Notes
    -----
    The function relies on two global objects that must be defined elsewhere in
    the module:

    * `GLOBAL_CORRECTED_TEXTS` – a `dict` that maps original texts to their
      corrected versions.
    * `GLOBAL_LOCK_CORRECTED_TEXTS` – a `threading.Lock` (or similar) used
      to protect concurrent access to the cache.

    This design makes the correction step safe for use in multi‑process or
    multi‑threaded environments, avoiding redundant API calls and reducing
    latency.
    """
    if text_str is None or not len(text_str):
        return text_str

    clear_s = None
    with GLOBAL_LOCK_CORRECTED_TEXTS:
        clear_s = GLOBAL_CORRECTED_TEXTS.get(text_str)
    if clear_s is not None:
        return clear_s

    clear_s = open_api.generate(
        message=text_str, system_prompt=prompt_str, max_tokens=len(text_str) * 2
    )
    if clear_s is None or not len(clear_s.strip()):
        clear_s = text_str

    with GLOBAL_LOCK_CORRECTED_TEXTS:
        GLOBAL_CORRECTED_TEXTS[text_str] = clear_s
    return clear_s


def process_sample_batch(
    batch: List[Dict[Any, Any]],
    split_to_sentences: bool,
    spacy_model_name: str = "pl_core_news_sm",
    open_api_config_path: Optional[str] = None,
    correct_text: bool = False,
    prompts_dir: Optional[str] = None,
    prompt_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Process a batch of samples in a separate process.

    Args:
        batch: List of samples to process
        split_to_sentences: Whether to split text into sentences
        spacy_model_name: Name of the spacy model to use (default: pl_core_news_sm)
        open_api_config_path: Path to config file to use for text correct
        (default: None) this option is required when correct_text is True
        correct_text: Option to enable text correction, if enabled, then
        open_api_config_path, prompts_dir, prompt_to_correct must be provided.
        prompts_dir: Directory where prompts files are located.
        prompt_name: Name of prompt used to correct the text

    Returns:
        List of converted samples
    """
    open_api = None
    prompt_str = None
    if correct_text:
        if open_api_config_path is None:
            raise RuntimeError(
                "If correct_text is True, open_api_config_path must be provided"
            )
        if prompts_dir is None:
            raise RuntimeError(
                "If correct_text is True, prompts_dir must be provided"
            )
        if prompt_name is None:
            raise RuntimeError(
                "If correct_text is True, prompt_name must be provided"
            )

        open_api = OpenAPIClient(open_api_config=open_api_config_path)
        try:
            with PromptHandler(base_dir=prompts_dir) as prompt_handler:
                prompt_str = prompt_handler.get_prompt(key=prompt_name)
        except KeyError as e:
            raise RuntimeError(f"Prompt {prompt_name} not found. {e}")

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
        if correct_text:
            print("_s1 before", datetime.datetime.now())
            _s1 = correct_if_necessary(
                text_str=_s1, open_api=open_api, prompt_str=prompt_str
            )
            print("_s1 after", datetime.datetime.now())

            print("_s2 before", datetime.datetime.now())
            _s2 = correct_if_necessary(
                text_str=_s2, open_api=open_api, prompt_str=prompt_str
            )
            print("_s2 after", datetime.datetime.now())

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
        correct_texts: bool = False,
        prompts_dir: Optional[str] = None,
        prompt_name: Optional[str] = None,
        open_api_config_path: Optional[str] = None,
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
            correct_texts: If option is set to `True` then text correction
            will be run before splitting text to sentences. Defaults to False.
            prompts_dir: Directory where prompts files are located (default: None,
            required when `correct_text` is True).
            prompt_name: Name of prompt used to correct the text. Defaults to None,
            required when `correct_text` is True.
            open_api_config_path: Path to config file to use for text correct.
            Defaults to None, required when `correct_text` is True.
        """
        self.seed = seed

        self.jsonl_path = jsonl_path
        self.output_dir = output_dir
        self.train_ratio = train_ratio

        self.prompts_dir = prompts_dir
        self.prompt_name = prompt_name
        self.correct_texts = correct_texts
        self.open_api_config_path = open_api_config_path
        self.split_to_sentences = split_to_sentences

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
                open_api_config_path=self.open_api_config_path,
                correct_text=self.correct_texts,
                prompts_dir=self.prompts_dir,
                prompt_name=self.prompt_name,
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
        "--correct-texts",
        dest="correct_texts",
        action="store_true",
        default=False,
        help="Optional flag to correct text before splitting to sentences",
    )
    parser.add_argument(
        "--prompts-dir",
        dest="prompts_dir",
        type=str,
        default=None,
        help="Path to directory containing prompts files (*.prompt). "
        "This option is required when --correct-texts",
    )
    parser.add_argument(
        "--prompt-name",
        dest="prompt_name",
        type=str,
        default=None,
        help="Prompt name used to prepare correct version of text. "
        "This option is required when --correct-texts",
    )
    parser.add_argument(
        "--openapi-config",
        dest="openapi_config",
        type=str,
        default=None,
        help="Path to config with generative models available with OpenAPI. "
        "This option is required when --correct-texts",
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


def check_args(app_args: argparse.Namespace):
    """
    Validate command‑line arguments related to text correction.

    This function ensures that when the `--correct-texts` flag is set, the required
    `--prompts-dir`, `--prompt-name` and `--openapi-config` arguments
    are also provided.  If either is missing a `ValueError` with a
    clear message is raised.

    Args:
        app_args: `argparse.Namespace` containing the parsed command‑line
            arguments.

    Returns:
        The same `app_args` namespace if validation succeeds.
    """

    if app_args.correct_texts:
        if app_args.prompts_dir is None:
            raise ValueError("--correct-texts requires --prompts-dir")

        if app_args.prompt_name is None:
            raise ValueError("--correct-texts requires --prompts-dir")

        if app_args.openapi_config is None:
            raise ValueError("--correct-texts requires --openapi-config")

    return app_args


if __name__ == "__main__":
    args = check_args(parse_args())

    converter = EmbedderDatasetConverter(
        jsonl_path=args.jsonl_path,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        split_to_sentences=args.split_to_sentences,
        n_workers=args.n_workers,
        batch_size=args.batch_size,
        correct_texts=args.correct_texts,
        prompts_dir=args.prompts_dir,
        prompt_name=args.prompt_name,
        open_api_config_path=args.openapi_config,
    )
    converter.run()
