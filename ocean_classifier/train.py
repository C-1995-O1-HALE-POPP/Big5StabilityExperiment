import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from loguru import logger
from packaging import version

# ------------------
# Loguru Config
# ------------------
logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO"
)
logger.add("training.log", rotation="10 MB", level="INFO")

# ------------------
# HF Mirror & Imports
# ------------------
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    set_seed
)
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import transformers

# ------------------
# Config
# ------------------
MODEL_NAME = "roberta-large"
DATASET_NAME = "wenkai-li/big5_chat"
TEXT_FIELD = "train_output"

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 3
LR = 1e-5
SEED = 42

LEVEL2FLOAT = {"low": 0.0, "high": 1.0}
set_seed(SEED)

# ------------------
# Device Check
# ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"Using GPU: {gpu_name}")
else:
    logger.warning("No GPU detected, using CPU.")

# ------------------
# Dataset
# ------------------
def load_and_prepare(trait):
    logger.info(f"Loading dataset for {trait}...")
    ds = load_dataset(DATASET_NAME)
    filtered_ds = ds["train"].filter(
        lambda x: x.get("level") in ["low", "high"] and x.get("trait") == trait
    )
    logger.info(f"Total filtered samples for {trait}: {len(filtered_ds)}")
    split_ds = filtered_ds.train_test_split(test_size=0.1, seed=SEED)
    logger.info(f"{trait} - Train: {len(split_ds['train'])}, Validation: {len(split_ds['test'])}")
    return {"train": split_ds["train"], "validation": split_ds["test"]}

# ------------------
# Model
# ------------------
class SingleHeadRoberta(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0])
        logits = self.head(pooled).view(-1)

        loss = None
        if labels is not None:
            bce = nn.BCEWithLogitsLoss()
            loss = bce(logits, labels)
        return {"loss": loss, "logits": logits}

# ------------------
# Dataset Class
# ------------------
class SingleTraitDataset(Dataset):
    def __init__(self, hf_split, tokenizer):
        self.data = hf_split
        logger.info(self.data[0])
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        text = row.get(TEXT_FIELD) or ""
        encoding = self.tokenizer(
            text=text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        label = LEVEL2FLOAT[row["level"]]
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float32)
        }

@dataclass
class DataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features])
        }

# ------------------
# Metrics
# ------------------
def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    labels = labels.astype(int)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = np.nan
    return {"accuracy": acc, "f1": f1, "auc": auc}

# ------------------
# TrainingArguments
# ------------------
def build_training_arguments(trait):
    common_args = dict(
        output_dir=f"./big5_{trait}_classifier",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=SEED
    )
    if torch.cuda.is_available():
        common_args["fp16"] = True
    try:
        args = TrainingArguments(**common_args)
    except TypeError:
        for k in ["report_to", "evaluation_strategy", "save_strategy",
                  "load_best_model_at_end", "metric_for_best_model", "greater_is_better"]:
            common_args.pop(k, None)
        args = TrainingArguments(**common_args)
    return args

# ------------------
# Training Loop
# ------------------
def train_single_trait(trait):
    logger.info(f"===== Training {trait} classifier =====")
    ds = load_and_prepare(trait)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    train_ds = SingleTraitDataset(ds["train"], tokenizer)
    eval_ds = SingleTraitDataset(ds["validation"], tokenizer)

    model = SingleHeadRoberta(MODEL_NAME)
    collator = DataCollator()
    args = build_training_arguments(trait)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn
    )

    # 训练并在每个epoch后自动评估
    trainer.train()

    # 输出最终评估
    metrics = trainer.evaluate()
    logger.info(f"Final Evaluation metrics: {metrics}")

    save_base = f"./big5_{trait}_classifier"
    # 保存最佳模型
    best_model_dir = os.path.join(save_base, "best")
    trainer.save_model(best_model_dir)
    torch.save(model.state_dict(), os.path.join(best_model_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(best_model_dir)
    logger.info(f"Best model saved to {best_model_dir}")

    # # 保存最后一次模型
    # final_model_dir = os.path.join(save_base, "final")
    # os.makedirs(final_model_dir, exist_ok=True)
    # trainer.model.save_pretrained(final_model_dir)
    # tokenizer.save_pretrained(final_model_dir)
    # logger.info(f"Final model saved to {final_model_dir}")

# ------------------
# Main
# ------------------
def main():
    for trait in TRAITS:
        train_single_trait(trait)
    logger.info("All trait classifiers trained successfully.")

if __name__ == "__main__":
    main()
