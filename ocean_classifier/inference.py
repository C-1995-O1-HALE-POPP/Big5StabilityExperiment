import os
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from transformers import AutoTokenizer, AutoModel
import numpy as np
from loguru import logger

# -----------------
# Traits
# -----------------
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
DEFAULT_ENCODER = "roberta-large"


# -----------------
# Model Definition
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
        logits = self.head(pooled).view(-1)  # [B]
        return logits


# -----------------
# Inference Config
# -----------------
@dataclass
class InferenceConfig:
    model_root: str = "."   # 根目录，包含 big5_{trait}_classifier/best
    encoder_name: str = DEFAULT_ENCODER
    max_length: int = 512
    batch_size: int = 16
    threshold: float = 0.5
    chunk_long_text: bool = False
    chunk_stride: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------
# Inferencer
# -----------------
class Big5Inferencer:
    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.tokenizers: Dict[str, AutoTokenizer] = {}
        self.models: Dict[str, SingleHeadRoberta] = {}

        logger.info(f"Loading 5 trait classifiers from root: {cfg.model_root}")
        for trait in TRAITS:
            trait_dir = os.path.join(cfg.model_root, f"big5_{trait}_classifier", "best")
            if not os.path.isdir(trait_dir):
                raise FileNotFoundError(f"Directory not found for trait {trait}: {trait_dir}")

            tokenizer = AutoTokenizer.from_pretrained(trait_dir, use_fast=True)
            self.tokenizers[trait] = tokenizer

            model = SingleHeadRoberta(cfg.encoder_name)
            state_path = os.path.join(trait_dir, "pytorch_model.bin")
            state_dict = torch.load(state_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()
            self.models[trait] = model

        logger.success("All 5 classifiers loaded successfully.")

    @torch.inference_mode()
    def score(self, text: str) -> Dict[str, Dict[str, Any]]:
        return self.score_batch([text])[0]

    @torch.inference_mode()
    def score_batch(self, texts: List[str]) -> List[Dict[str, Dict[str, Any]]]:
        results = []
        for text in texts:
            trait2info = {}
            for trait in TRAITS:
                out = self._predict_one_trait(text, trait)
                trait2info[trait] = out
            results.append(trait2info)
        return results

    def _predict_one_trait(self, text: str, trait: str) -> Dict[str, float]:
        tokenizer = self.tokenizers[trait]
        model = self.models[trait]

        if not self.cfg.chunk_long_text:
            inputs = tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=self.cfg.max_length,
                return_tensors="pt"
            ).to(self.device)

            logits = model(**inputs)  # [B]
            logit_val = logits.mean().item()
            prob_val = torch.sigmoid(logits).mean().item()
            label = "high" if prob_val >= self.cfg.threshold else "low"
            return {"logit": logit_val, "prob": prob_val, "label": label}

        # 长文本切块（可选）
        chunks = _split_text_by_length(text, self.cfg.max_length)
        all_logits = []
        for i in range(0, len(chunks), self.cfg.batch_size):
            batch = chunks[i:i + self.cfg.batch_size]
            batch_inputs = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.cfg.max_length,
                return_tensors="pt"
            ).to(self.device)
            logits = model(**batch_inputs)
            all_logits.append(logits.cpu().numpy())
        all_logits = np.concatenate(all_logits) if all_logits else np.array([0.0])
        avg_logit = float(all_logits.mean())
        prob_val = float(1 / (1 + np.exp(-avg_logit)))
        label = "high" if prob_val >= self.cfg.threshold else "low"
        return {"logit": avg_logit, "prob": prob_val, "label": label}


def _split_text_by_length(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks


# -----------------
# Example Usage
# -----------------
from loguru import logger
class big5_classifier:
    def __init__(self, model_root="./ocean_classifier", encoder_name=DEFAULT_ENCODER, max_length=512,
            batch_size=16, threshold=0.5, chunk_long_text=True):
        self.cfg = InferenceConfig(
            model_root=model_root,  # 修改为你的模型根目录
            encoder_name=encoder_name,
            max_length=max_length,
            batch_size=batch_size,
            threshold=threshold,
            chunk_long_text=chunk_long_text
        )
        self.infer = Big5Inferencer(self.cfg)

    def inference(self, texts: List[str]) -> Dict:
        scores = self.infer.score_batch(texts)
        for t, s in zip(texts, scores):
            logger.debug("TEXT:", t)
            for trait, info in s.items():
                logger.debug(f"  {trait:18s} -> logit={info['logit']:.4f} prob={info['prob']:.4f} label={info['label']}")
            logger.debug("-" * 80)
        return scores
