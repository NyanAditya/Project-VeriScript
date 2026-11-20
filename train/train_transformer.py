"""Step 5: fine-tune RoBERTa-base (or DeBERTa-v3-base) as a binary classifier (Sec. 4.3.4, Table 6.3).

    python -m train.train_transformer [--model roberta-base | microsoft/deberta-v3-base]
                                      [--exclude-family llama] [--max-train N]

Training documents are truncated to the first 512 tokens; inference uses sliding windows.
The best checkpoint by validation F1 is saved to models/transformer/<name>.
"""

from __future__ import annotations

import argparse
import inspect
import json

import numpy as np

from veriscript.config import PROJECT_ROOT, load_config
from veriscript.utils import set_seed

from .common import MODELS, REPORTS, load_corpus, write_json


def main(argv=None):
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--out", default=None)
    ap.add_argument("--exclude-family", default=None)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=None, help="default 2e-5 (RoBERTa) / 1.5e-5 (DeBERTa)")
    ap.add_argument("--batch-size", type=int, default=8, help="per-device; gradient accumulation makes the effective size 16")
    ap.add_argument("--max-len", type=int, default=cfg["transformer"]["max_len"])
    ap.add_argument("--max-train", type=int, default=None, help="subsample the training split (quick runs)")
    ap.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    args = ap.parse_args(argv)

    import torch
    from datasets import Dataset
    from sklearn.metrics import f1_score, roc_auc_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              EarlyStoppingCallback, Trainer, TrainingArguments)

    set_seed(args.seed)
    name = args.model.split("/")[-1]
    suffix = f"_no_{args.exclude_family}" if args.exclude_family else ""
    out_dir = MODELS / "transformer" / (args.out or f"{name}{suffix}")
    lr = args.lr or (1.5e-5 if "deberta" in args.model else 2e-5)

    corpus = load_corpus()
    if args.exclude_family:
        corpus = corpus[~((corpus.label == 1) & (corpus.generator_family == args.exclude_family))]
    train_df = corpus[corpus.split == "train"]
    if args.max_train and len(train_df) > args.max_train:
        train_df = train_df.groupby("label", group_keys=False).apply(lambda g: g.sample(args.max_train // 2, random_state=args.seed))
    val_df = corpus[corpus.split == "val"]

    tok = AutoTokenizer.from_pretrained(args.model)

    def encode(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)

    ds_tr = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True)).map(encode, batched=True, remove_columns=["text"])
    ds_va = Dataset.from_pandas(val_df[["text", "label"]].reset_index(drop=True)).map(encode, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    def metrics(pred):
        logits, labels = pred
        p = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)[:, 1].numpy()
        return {"f1": f1_score(labels, p >= 0.5), "auroc": roc_auc_score(labels, p)}

    ta_kwargs = dict(
        output_dir=str(out_dir) + "_ckpt",
        num_train_epochs=args.epochs,
        learning_rate=lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 4,
        gradient_accumulation_steps=max(1, 16 // args.batch_size),
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="linear",
        fp16=torch.cuda.is_available() and "deberta" not in args.model,  # DeBERTa-v3 is unstable in fp16
        bf16=False,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=100,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
    )
    # transformers renamed evaluation_strategy -> eval_strategy
    key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"
    ta_kwargs[key] = "epoch"
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**ta_kwargs),
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        tokenizer=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    trainer.train()
    ev = trainer.evaluate()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    info = {
        "base_model": args.model,
        "exclude_family": args.exclude_family,
        "n_train": len(train_df),
        "lr": lr,
        "epochs": args.epochs,
        "val": {k: float(v) for k, v in ev.items() if isinstance(v, (int, float))},
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "log_history": trainer.state.log_history,
    }
    (out_dir / "veriscript_training.json").write_text(json.dumps(info, indent=2, default=str))
    write_json(info, REPORTS / f"transformer_training_{out_dir.name}.json")
    print(f"saved {out_dir.relative_to(PROJECT_ROOT)}; val {info['val']}")


if __name__ == "__main__":
    main()
