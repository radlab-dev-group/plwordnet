from plwordnet_trainer.utils.handler import TrainingHandler
from plwordnet_trainer.utils.wandb import WanDBHandler, WandbConfig
from plwordnet_trainer.utils.argument_parser import (
    BASE_MODEL_REQUIRED,
    OUTPUT_DIR_REQUIRED,
    TRAIN_INPUT_FILE_REQUIRED,
    VALID_INPUT_FILE_REQUIRED,
    WANDB_BOOLEAN_FULL,
    prepare_parser_for_fields,
)

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction


def main(argv=None):
    args = prepare_parser_for_fields(
        fields_list=[
            BASE_MODEL_REQUIRED,
            OUTPUT_DIR_REQUIRED,
            TRAIN_INPUT_FILE_REQUIRED,
            VALID_INPUT_FILE_REQUIRED,
            WANDB_BOOLEAN_FULL,
        ],
        description="plWordnet semantic embeddings Trainer",
    ).parse_args()

    train_batch_size = 4
    training_handler = TrainingHandler(
        train_dataset_file_path=args.train_file,
        eval_dataset_file_path=args.valid_file,
        base_model=args.base_model,
        train_batch_size=train_batch_size,
    )

    report_to = []
    if args.wandb_full:
        report_to = ["wandb"]

    training_arguments = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        # warmup_ratio=0.1,
        warmup_steps=1000,
        # weight_decay=0.001, # 0.0
        load_best_model_at_end=True,
        logging_steps=5,
        evaluation_strategy="steps",
        eval_steps=5000,
        disable_tqdm=False,
        logging_first_step=False,
        learning_rate=0.0000001,  # 5e-1,
        fp16=True,
        bf16=False,
        report_to=report_to,
        save_total_limit=5,
        save_strategy="steps",
        save_steps=10000,
        logging_strategy="steps",
        optim="adamw_torch_fused",
    )

    run_conf_dict = {
        "dataset_path": args.train_file,
        "base_model": args.base_model,
        "output_dir": args.output_dir,
    }

    wnb_config = WandbConfig()
    if args.wandb_full:
        WanDBHandler.init_wandb(
            wandb_config=wnb_config,
            run_config=run_conf_dict,
            training_args=training_arguments,
        )

    model = SentenceTransformer(
        model_name_or_path=args.base_model,
        tokenizer_kwargs={
            "model_max_length": 256,
            "truncation": True,
            "padding": True,
            "max_length": 256
        },
        trust_remote_code=True,
    )

    loss = losses.CosineSimilarityLoss(model=model)

    dev_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=training_handler.eval_dataset["sentence1"],
        sentences2=training_handler.eval_dataset["sentence2"],
        scores=training_handler.eval_dataset["score"],
        main_similarity=SimilarityFunction.COSINE,
        name="sts-dev",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_arguments,
        train_dataset=training_handler.train_dataset,
        eval_dataset=training_handler.eval_dataset,
        tokenizer=training_handler.tokenizer,
        loss=loss,
        evaluator=dev_evaluator,
    )

    trainer.train()

    test_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=training_handler.eval_dataset["sentence1"],
        sentences2=training_handler.eval_dataset["sentence2"],
        scores=training_handler.eval_dataset["score"],
        main_similarity=SimilarityFunction.COSINE,
        name="sts-dev",
    )

    test_evaluator(model)

    if args.wandb_full:
        # WanDBHandler.add_model(name=wnb_config.PROJECT_NAME, local_path=best_m_path)
        WanDBHandler.finish_wand()


if __name__ == "__main__":
    main()
