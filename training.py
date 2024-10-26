from transformers import (
    AlbertForSequenceClassification,
    Trainer,
    TrainingArguments,
    AlbertTokenizer,
)
from datasets import load_dataset, load_from_disk
import torch
import os
import sys

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import wandb
import yaml


def preprocess_function(examples):
    """ Tokenises inputs. """
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        padding="max_length",
    )


def compute_metrics(pred) -> dict:
    """Callback function to compute the metrics during validation and testing."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Calculate accuracy
    acc = accuracy_score(labels, preds)

    # Calculate macro-averaged F1 score
    f1 = f1_score(labels, preds, average="macro")

    # Calculate macro-averaged precision and recall
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def generate_unique_logpath(res_dir:str, raw_run_name:str)->str:
    """
    Generate a unique directory name
    Argument:
        res_dir: the prefix directory
        raw_run_name(str): the base name
    Returns:
        log_path: a non-existent path like res_dir/raw_run_name_xxxx
                  where xxxx is an int
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(res_dir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


if __name__ == "__main__":

    # Load YAML file
    config_path = sys.argv[1]

    # Load YAML file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    preprocess_path = config["PREPROCESS_PATH"]

    if os.path.exists(preprocess_path):
        snli = load_from_disk(preprocess_path)
    else:
        snli = load_dataset("snli")
        # Removing sentence pairs with no label (-1)
        snli = snli.filter(lambda example: example["label"] != -1)
        tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
        snli = snli.map(preprocess_function, batched=True)
        snli.save_to_disk(preprocess_path)

    snli.set_format(
        type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"]
    )

    if config["USE_ALL_SAMPLES"]:
        data_train = snli["train"]
        data_val = snli["validation"]
    else:
        data_train = snli["train"].select(range(config["NUM_SAMPLES"]))
        data_val = snli["validation"].select(range(config["NUM_SAMPLES"]))

    data_test = snli["test"]

    ## MODEL
    model = AlbertForSequenceClassification.from_pretrained(
        config["PRETRAINED_MODEL"], num_labels=3
    )

    wandb.init(
        # set the wandb project where this run will be logged
        project="nlp_project",
    )

    res_dir = generate_unique_logpath(
        config["BASE_RES_PATH"], f"{config['PRETRAINED_MODEL']}"
    )
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)
    print(f"Will be logging into {res_dir}")

    ## TRAINING
    training_args = TrainingArguments(
        output_dir=res_dir,
        overwrite_output_dir=True,
        logging_steps=200,
        num_train_epochs=config["N_EPOCHS"],
        per_device_train_batch_size=config["BATCH_SIZE"],
        per_device_eval_batch_size=config["BATCH_SIZE"],
        optim="adamw_torch",
        weight_decay=config["WEIGHT_DECAY"],
        warmup_steps=config["WARMUP_STEPS"],
        report_to="wandb",
        save_safetensors=False,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=5000,
        save_steps=5000,
        load_best_model_at_end=True,
        learning_rate=config["LEARNING_RATE"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_val,
        compute_metrics=compute_metrics
    )

    trainer.train()

    ## Print best checkpoint path at the end of training. 
    print(f"*** Best checkpoint : {trainer.state.best_model_checkpoint}")

    ## TESTING on test set
    print("*** Testing on test set : ")
    trainer.evaluate(data_test)
