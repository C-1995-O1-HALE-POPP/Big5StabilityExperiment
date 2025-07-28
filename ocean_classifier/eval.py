import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from loguru import logger

# -----------------
# 配置
# -----------------
TRAIT = "openness"  # 可修改为 5 个特质之一
MODEL_DIR = f"./big5_{TRAIT}_classifier/best"
MODEL_NAME = "roberta-large"
DATASET_NAME = "wenkai-li/big5_chat"
TEXT_FIELD = "train_output"
MAX_LENGTH = 512
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEVEL2FLOAT = {"low": 0.0, "high": 1.0}


# -----------------
# 模型结构（与训练时一致）
# -----------------
class SingleHeadRoberta(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0])
        logits = self.head(pooled).view(-1)
        return logits


# -----------------
# 数据加载
# -----------------
def load_validation_data(trait):
    ds = load_dataset(DATASET_NAME)
    filtered = ds["train"].filter(lambda x: x.get("level") in ["low", "high"] and x.get("trait") == trait)
    split_ds = filtered.train_test_split(test_size=0.1, seed=42)
    return split_ds["test"]


# -----------------
# 评估函数
# -----------------
@torch.no_grad()
def evaluate_model(model, tokenizer, dataset):
    all_logits, all_labels = [], []
    loss_fn = nn.BCEWithLogitsLoss()

    for i in range(0, len(dataset), BATCH_SIZE):
        batch = dataset[i:i + BATCH_SIZE]
        texts = batch[TEXT_FIELD]  # 直接取出字段
        labels = [LEVEL2FLOAT[l] for l in batch["level"]]


        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(DEVICE)

        labels_tensor = torch.tensor(labels, dtype=torch.float32).to(DEVICE)
        logits = model(**encodings)
        all_logits.append(logits.cpu())
        all_labels.append(labels_tensor.cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-all_logits))
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    auc = roc_auc_score(all_labels, probs)

    loss = loss_fn(torch.tensor(all_logits), torch.tensor(all_labels)).item()

    return acc, f1, auc, loss, all_labels, probs, preds


# -----------------
# 可视化函数
# -----------------
def plot_confusion_matrix(labels, preds, trait):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"{trait.capitalize()} Confusion Matrix")
    plt.colorbar()
    classes = ["Low", "High"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{trait}.png")
    logger.info(f"Confusion matrix saved as confusion_matrix_{trait}.png")


def plot_roc_curve(labels, probs, trait):
    RocCurveDisplay.from_predictions(labels, probs)
    plt.title(f"{trait.capitalize()} ROC Curve")
    plt.savefig(f"roc_curve_{trait}.png")
    logger.info(f"ROC curve saved as roc_curve_{trait}.png")


# -----------------
# 主程序
# -----------------
def main():
    logger.info(f"Loading model from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = SingleHeadRoberta(MODEL_NAME)
    state_path = os.path.join(MODEL_DIR, "pytorch_model.bin")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"pytorch_model.bin not found in {MODEL_DIR}, did you save state_dict?")
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.to(DEVICE)
    model.eval()

    logger.info(f"Loading validation data for {TRAIT}")
    val_ds = load_validation_data(TRAIT)

    logger.info("Evaluating model...")
    acc, f1, auc, loss, labels, probs, preds = evaluate_model(model, tokenizer, val_ds)
    logger.info(f"Results: ACC={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, Loss={loss:.4f}")

    logger.info("Plotting metrics...")
    plot_confusion_matrix(labels, preds, TRAIT)
    plot_roc_curve(labels, probs, TRAIT)


if __name__ == "__main__":
    main()
