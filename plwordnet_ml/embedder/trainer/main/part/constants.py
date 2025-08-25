class ConstantsRelGATTrainer:
    class Default:
        EPOCHS = 12
        TRAIN_EVAL_RATIO = 0.9
        TRAIN_BATCH_SIZE = 256

        LOG_EVERY_N_STEPS = 100

        NUM_NEG = 6
        GAT_HEADS = 12
        GAT_NUM_LAYERS = 1
        GAT_DROPOUT = 0.25
        GAT_OUT_DIM = 300

        LR = 2e-4
        # Scheduler: {"linear", "cosine", "constant"}
        LR_SCHEDULER = "linear"
        WARMUP_STEPS = None
        # As default, 10% of training is used to warmup
        DEFAULT_WARMUP_RATIO = 0.1

        # Scorer, one of: {"distmult", "transe"}
        GAT_SCORER = "distmult"

        # Storing model
        OUT_MODEL_NAME = "relgat-model.pt"
        DEFAULT_TRAINER_OUT_DIR = "relgat-out"
        TRAINING_CONFIG_FILE_NAME = "training-config.json"

    from plwordnet_ml.embedder.constants.wandb import WandbConfig as _WANDBConfig

    class WandbConfig(_WANDBConfig):
        PROJECT_NAME = "plWordnet-relgat"
        PROJECT_TAGS = ["relgat", "link-prediction"]
        PREFIX_RUN = "run_"
        BASE_RUN_NAME = "relgat"
