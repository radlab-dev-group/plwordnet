import os.path

import wandb
import socket
import datetime


class WanDBHandler:
    @staticmethod
    def init_wandb(
        wandb_config, run_config, training_args, run_name: str | None = None
    ):
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
        artifact = wandb.Artifact(name=name, type="dataset")
        artifact.add_dir(local_path=local_path)
        wandb.log_artifact(artifact)

    @staticmethod
    def add_model(name, local_path):
        artifact = wandb.Artifact(name=name, type="model")
        artifact.add_dir(local_path=local_path)
        wandb.log_artifact(artifact)

    @staticmethod
    def finish_wand():
        wandb.run.finish()

    @staticmethod
    def prepare_run_tags(run_tags):
        run_tags = run_tags if run_tags is not None else []
        run_tags.append(socket.gethostname())
        return run_tags

    @staticmethod
    def prepare_run_name_with_date(wandb_config, run_config):
        run_prefix = wandb_config.PREFIX_RUN
        base_run_name = wandb_config.BASE_RUN_NAME
        bm_name = os.path.basename(run_config["base_model"])
        date_str = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")

        return f"{run_prefix}{bm_name} {base_run_name} {date_str}"

    @staticmethod
    def prepare_simple_run_name_with_date(wandb_config):
        run_prefix = wandb_config.PREFIX_RUN
        base_run_name = wandb_config.BASE_RUN_NAME
        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        return f"{run_prefix}{base_run_name}_{date_str}"

    @staticmethod
    def prepare_run_config(run_config, training_args):
        if training_args is not None:
            for k, v in training_args.__dict__.items():
                run_config[k] = v
        return run_config

    @staticmethod
    def plot_confusion_matrix(ground_truth, predictions, class_names, probs=None):
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
