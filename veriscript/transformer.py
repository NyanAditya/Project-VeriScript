"""Transformer stage: windowed RoBERTa / DeBERTa-v3 inference (Sec. 4.3.4, Eq. 4.6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class TransformerDetector:
    """Fine-tuned encoder; long documents are split into overlapping windows and averaged."""

    name = "transformer"

    def __init__(self, model_dir: str | Path, max_len: int = 512, stride: int = 384, device: str = "cpu", batch_size: int = 16):
        self.model_dir = str(model_dir)
        self.max_len = max_len
        self.stride = stride
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._tok = None

    @property
    def available(self) -> bool:
        return (Path(self.model_dir) / "config.json").exists()

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            if not self.available:
                raise FileNotFoundError(
                    f"No fine-tuned transformer found in {self.model_dir}. Run `python -m train.train_transformer` first."
                )
            self._tok = AutoTokenizer.from_pretrained(self.model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            model.eval().to(self.device)
            if self.device == "cuda":
                model = model.half()
            self._model = model
            self._torch = torch
        return self._model, self._tok

    def windows(self, text: str) -> list[list[int]]:
        """Token-id windows of at most max_len (including special tokens), step = stride."""
        _, tok = self._load()
        ids = tok(text, add_special_tokens=False)["input_ids"]
        body = self.max_len - tok.num_special_tokens_to_add(pair=False)
        if len(ids) <= body:
            return [tok.build_inputs_with_special_tokens(ids)]
        out = []
        for start in range(0, len(ids), self.stride):
            piece = ids[start : start + body]
            out.append(tok.build_inputs_with_special_tokens(piece))
            if start + body >= len(ids):
                break
        return out

    def _score_windows(self, wins: list[list[int]]) -> np.ndarray:
        model, tok = self._load()
        torch = self._torch
        probs = []
        pad = tok.pad_token_id
        with torch.no_grad():
            for i in range(0, len(wins), self.batch_size):
                batch = wins[i : i + self.batch_size]
                L = max(len(w) for w in batch)
                ids = torch.full((len(batch), L), pad, dtype=torch.long)
                mask = torch.zeros((len(batch), L), dtype=torch.long)
                for j, w in enumerate(batch):
                    ids[j, : len(w)] = torch.tensor(w)
                    mask[j, : len(w)] = 1
                logits = model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).logits.float()
                probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(probs) if probs else np.zeros(0)

    def predict_proba(self, text: str) -> float:
        return float(self.predict_with_windows(text)[0])

    def predict_with_windows(self, text: str) -> tuple[float, int]:
        wins = self.windows(text)
        return float(np.mean(self._score_windows(wins))), len(wins)

    def predict_proba_many(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Mini-batched inference over many documents; returns (p_t, n_windows)."""
        all_w, owner = [], []
        for k, t in enumerate(texts):
            w = self.windows(t)
            all_w.extend(w)
            owner.extend([k] * len(w))
        if not all_w:
            return np.zeros(0), np.zeros(0, dtype=int)
        # Sort by length so batches pad little.
        order = np.argsort([len(w) for w in all_w])
        scores_sorted = self._score_windows([all_w[i] for i in order])
        scores = np.empty(len(all_w))
        scores[order] = scores_sorted
        owner = np.asarray(owner)
        p = np.array([scores[owner == k].mean() for k in range(len(texts))])
        n = np.bincount(owner, minlength=len(texts))
        return p, n
