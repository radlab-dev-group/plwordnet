from datasets import load_dataset
from transformers import AutoTokenizer


class TrainingHandler:
    def __init__(
        self,
        train_dataset_file_path,
        eval_dataset_file_path,
        base_model,
        train_batch_size,
        workdir="./workdir",
    ):
        self.workdir = workdir
        self.base_model = base_model
        self.train_dataset_file_path = train_dataset_file_path
        self.eval_dataset_file_path = eval_dataset_file_path

        self.num_labels = None
        self.uniq_labels = set()
        self.train_dataset = None
        self.valid_dataset = None

        self.eval_dataloader = None
        self.train_dataloader = None

        self.train_batch_size = train_batch_size

        self.__prepare_datasets()

    def __prepare_datasets(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, use_fast=True
        )

        data = load_dataset(
            "json",
            cache_dir="./cache",
            data_files={
                "train": self.train_dataset_file_path,
                "validation": self.eval_dataset_file_path,
            },
        )

        self.train_dataset = data["train"]
        self.eval_dataset = data["validation"]
