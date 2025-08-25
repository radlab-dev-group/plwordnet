import os.path

import wandb
import socket
import datetime


class WanDBHandler:
    """
    Handler class for Weights & Biases (wandb) integration and logging.

    Provides static methods for initializing wandb runs, logging artifacts,
    and managing experiment tracking for machine learning workflows.
    """

    @staticmethod
    def init_wandb(
        wandb_config, run_config, training_args, run_name: str | None = None
    ):
        """
        Initialize a wandb run with project configuration.

        Args:
            wandb_config: Configuration object containing wandb settings
            run_config: Dictionary containing run-specific configuration
            training_args: Training arguments object to be logged
            run_name: Optional custom name for the run. If None,
            generates a name with a timestamp
        """
        project_run_name = run_name
        if run_name is None or not len(run_name.strip()):
            project_run_name = WanDBHandler.prepare_run_name_with_date(
                wandb_config, run_config
            )

        wandb.init(
            project=wandb_config.PROJECT_NAME,
            name=project_run_name,
            tags=WanDBHandler.prepare_run_tags(wandb_config.PROJECT_TAGS),
            config=WanDBHandler.prepare_run_config(run_config, training_args),
        )

    @staticmethod
    def add_dataset(name, local_path):
        """
        Add a dataset as a wandb artifact.

        Args:
            name: Name identifier for the dataset artifact
            local_path: Local directory path containing the dataset
        """
        artifact = wandb.Artifact(name=name, type="dataset")
        artifact.add_dir(local_path=local_path)
        wandb.log_artifact(artifact)

    @staticmethod
    def add_model(name, local_path):
        """
        Add a model as a wandb artifact.

        Args:
            name: Name identifier for the model artifact
            local_path: Local directory path containing the model files
        """
        artifact = wandb.Artifact(name=name, type="model")
        artifact.add_dir(local_path=local_path)
        wandb.log_artifact(artifact)

    @staticmethod
    def log_metrics(metrics: dict, step: int | None = None):
        """
        Log a dictionary of scalar metrics to wandb.

        Args:
            metrics: Mapping name → value (float, int, …).
            step: Optional global step that will be attached to the log.
        """
        wandb.log(metrics, step=step)

    @staticmethod
    def finish_wand():
        """
        Finish the current wandb run and clean up resources.
        """
        if wandb.run is not None:
            wandb.run.finish()

    @staticmethod
    def prepare_run_tags(run_tags):
        """
        Prepare and enhance run tags with system information.

        Args:
            run_tags: List of initial tags for the run

        Returns:
            List of tags with hostname appended
        """
        run_tags = run_tags if run_tags is not None else []
        run_tags.append(socket.gethostname())
        return run_tags

    @staticmethod
    def prepare_run_name_with_date(wandb_config, run_config):
        """
        Generate a run name with timestamp and model information.

        Args:
            wandb_config: Configuration object containing naming settings
            run_config: Dictionary containing run configuration with base_model

        Returns:
            str: Formatted run name with prefix, model name, and timestamp
        """
        run_prefix = wandb_config.PREFIX_RUN
        base_run_name = wandb_config.BASE_RUN_NAME
        bm_name = os.path.basename(run_config["base_model"])
        date_str = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")

        return f"{run_prefix}{bm_name} {base_run_name} {date_str}"

    @staticmethod
    def prepare_simple_run_name_with_date(wandb_config):
        """
        Generate a simple run name with a timestamp.

        Args:
            wandb_config: Configuration object containing naming settings

        Returns:
            str: Formatted run name with prefix, base name, and timestamp
        """
        run_prefix = wandb_config.PREFIX_RUN
        base_run_name = wandb_config.BASE_RUN_NAME
        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        return f"{run_prefix}{base_run_name}_{date_str}"

    @staticmethod
    def prepare_run_config(run_config, training_args):
        """
        Merge run configuration with training arguments.

        Args:
            run_config: Dictionary containing base run configuration
            training_args: Training arguments object to merge

        Returns:
            Dict: Combined configuration dictionary
        """
        if training_args is not None:
            for k, v in training_args.__dict__.items():
                run_config[k] = v
        return run_config

    @staticmethod
    def plot_confusion_matrix(ground_truth, predictions, class_names, probs=None):
        """
        Log a confusion matrix plot to wandb.

        Args:
            ground_truth: True labels
            predictions: Predicted labels
            class_names: List of class names for labeling
            probs: Optional prediction probabilities
        """
        wandb.log(
            {
                "Confusion matrix": wandb.plot.confusion_matrix(
                    probs=probs,
                    y_true=ground_truth,
                    preds=predictions,
                    class_names=class_names,
                )
            }
        )

    @staticmethod
    def store_prediction_results(texts_str, ground_truth, pred_labels, probs=None):
        """
        Log detailed prediction results as a wandb table.

        Args:
            texts_str: List of input text strings
            ground_truth: List of true labels
            pred_labels: List of predicted labels
            probs: Optional list of prediction probability arrays
        """
        assert len(texts_str) == len(ground_truth) == len(pred_labels)

        table_data = []
        class_header = None
        for txt, cl, pcl, prob in zip(texts_str, ground_truth, pred_labels, probs):
            table_row = [txt, cl, pcl, prob]
            if class_header is None:
                class_header = []
                for idx, _ in enumerate(prob):
                    class_header.append(f"c{idx}")
            for idx, p in enumerate(prob):
                table_row.append(p)

            table_data.append(table_row)

        eval_pred_table = wandb.Table(
            columns=["text", "class", "pred_class", "probs"] + class_header,
            data=table_data,
        )
        wandb.log({"Predictions on text eval": eval_pred_table})
