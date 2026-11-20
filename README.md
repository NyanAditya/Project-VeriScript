# VeriScript

**A hybrid heuristic-transformer system for AI-generated text detection.**

VeriScript scores every document with a cheap, interpretable heuristic model and escalates only the
ambiguous ones to a fine-tuned transformer. A calibrated heuristic probability `p_h(d)` is compared
with an ambiguity band `[T_low, T_high]`:

```
p_h < T_low   -> fast path: human-written      (explained by the top features)
p_h > T_high  -> fast path: AI-generated       (explained by the top features)
otherwise     -> escalate: RoBERTa-base on 512-token sliding windows (stride 384), averaged
```

The band is tuned on validation data to give the lowest escalation rate (cost) whose F1 is within
ε of the transformer-only detector, without a larger native / non-native false-positive gap
(Algorithm 4.3 of the report). Everything runs locally on open-source models.

> **Use responsibly.** Scores are probabilistic estimates, not proof of misconduct. Every result
> carries an advisory note and should be reviewed by a person.

## Repository layout

```
configs/default.yaml        model paths, thresholds, operating points
data/download.py            fetch HC3, M4, RAID, MAGE and the Liang et al. fairness essays
data/build_corpus.py        unified, balanced, group-split corpus (Tables 6.5, 6.6)
data/generate_custom.py     optional topic-matched Claude / GPT / Llama generations
veriscript/                 core library (no UI dependencies)
  io.py                     text extraction (.txt, .pdf, .docx)
  preprocess.py             NFKC, markup removal, spaCy sentences, length check
  features.py               GPT-2 token statistics, burstiness, stylometry (38 features)
  heuristic.py              LR / XGBoost head, Platt / isotonic calibration, explanations
  transformer.py            windowed RoBERTa / DeBERTa-v3 inference
  gate.py                   confidence gate, band grid search, trade-off curve
  cascade.py                CascadeDetector facade, batch processing
  evaluate.py               metrics, bootstrap CIs, paired tests, latency, fairness, plots
  cli.py                    `veriscript` command line tool
train/                      feature extraction, training, prediction caching, tuning, evaluation
app/api.py                  FastAPI REST service (Table 5.4)
app/streamlit_app.py        web interface
tests/                      pytest unit, integration and API tests
Dockerfile
```

## Setup

Python 3.11. A CUDA GPU is optional (used for training and for faster inference).

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124   # or /whl/cpu
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m spacy download en_core_web_sm
.venv\Scripts\python -m pytest -q
```

## Reproducing the pipeline

Each step writes to `data/processed/`, `models/` or `reports/` (all git-ignored).

```powershell
python -m data.download                 # ~3 GB: HC3, M4 (English), RAID train sample, MAGE OOD, fairness essays
python -m data.build_corpus             # data/processed/corpus.parquet (~44.6k docs)
python -m train.extract_features        # GPT-2 + spaCy features for every document (~40 min on a GPU)
python -m train.train_heuristic         # LR vs XGBoost, calibration -> models/heuristic/heuristic.joblib
python -m train.train_transformer       # RoBERTa-base fine-tuning -> models/transformer/roberta-base
python -m train.predict                 # cache p_h and p_t on every evaluation split
python -m train.tune_thresholds         # fast / balanced / thorough bands -> models/operating_points.json
python -m train.evaluate_all            # reports/results.md, reports/results.json, reports/figures/
```

Optional, for the generalisation study:

```powershell
# Leave-one-generator-out for the heuristic stage (cheap; repeat per family)
python -m train.train_heuristic --exclude-family gpt --out models/heuristic/heuristic_no_gpt.joblib
python -m train.predict --heuristic models/heuristic/heuristic_no_gpt.joblib --tag no_gpt --skip-transformer --splits test
# Full-cascade LOGO for one family (retrains the transformer, ~2-3 h)
python -m train.train_transformer --exclude-family llama
python -m train.predict --heuristic models/heuristic/heuristic_no_llama.joblib --transformer models/transformer/roberta-base_no_llama --tag no_llama --splits test
# DeBERTa-v3 as the alternative second stage
python -m train.train_transformer --model microsoft/deberta-v3-base
```

Custom generations (API usage is billed by the provider; start small):

```powershell
python -m data.generate_custom --provider claude --n 100       # Anthropic SDK; default model claude-opus-5-5
python -m data.generate_custom --provider openai --n 100       # needs `pip install openai`
python -m data.generate_custom --provider local --model meta-llama/Meta-Llama-3-8B-Instruct --load-in-4bit --n 100
python -m data.build_corpus                                     # picks up data/raw/custom/*.jsonl
```

## Using the detector

```powershell
veriscript detect essay.docx
veriscript detect --text "..." --json --mode thorough
veriscript batch submissions/ --out report.csv --mode balanced
veriscript evaluate --data data/processed/corpus.parquet --systems heuristic,transformer,cascade --limit 500
veriscript retrain-heuristic --data new_labelled.csv      # FR12: retrain + recalibrate on new data
veriscript tune --data new_validation.csv                 # re-tune the operating points
veriscript serve                                          # REST API on http://127.0.0.1:8000/docs
veriscript demo                                           # Streamlit UI on http://localhost:8501
```

Python:

```python
from veriscript import load_detector
det = load_detector(operating_point="balanced")
print(det.detect(open("essay.txt").read()).to_json())
```

### REST API

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/detect` | `{"text": ..., "mode": "balanced"}` → label, probability, route, latency, explanation |
| POST | `/api/v1/detect/file` | multipart upload of a .txt / .pdf / .docx |
| POST | `/api/v1/batch` | CSV (`id,text`) or ZIP of documents → job id (processed in the background) |
| GET | `/api/v1/batch/{id}` | job status and, when complete, per-document results and summary |
| GET | `/api/v1/config` | current operating point and thresholds |
| PUT | `/api/v1/config` | change operating point / thresholds; requires header `X-Admin-Token` = `$VERISCRIPT_ADMIN_TOKEN` |
| GET | `/api/v1/health` | service and model status |

### Docker

After training (so `models/` exists):

```powershell
docker build -t veriscript .
docker run -p 8000:8000 -p 8501:8501 veriscript
```

## Data sources

| Source | Use | Licence / access |
|---|---|---|
| HC3 (Hello-SimpleAI/HC3) | train / cal / val / test | CC-BY-SA, Hugging Face Hub |
| M4 (mbzuai-nlp/M4), English | train / cal / val / test | GitHub release |
| RAID (liamdugan/raid), train split sample | train / cal / val / test; attacked subset for robustness | Hugging Face parquet conversion |
| MAGE OOD (yaful/MAGE) | out-of-distribution test only (plain and paraphrased) | Hugging Face Hub |
| Liang et al. (2023) essays | fairness: TOEFL (non-native) vs Hewlett + college essays (native) | GitHub release |
| Custom generations | optional, topic-matched pairs | `data/generate_custom.py` |

## Results (first full run)

Full tables are in `reports/results.md` (git-ignored, regenerated by `train.evaluate_all`). Headline
numbers on the in-distribution `test` split (4,000 documents):

| System | F1 | AUROC | FPR | Escalation | CPU mean latency | GPU mean latency |
|---|---|---|---|---|---|---|
| Heuristic only (XGBoost + isotonic, 38 features) | 0.946 | 0.988 | 4.7% | 0% | 354 ms | 73 ms |
| Transformer only (RoBERTa-base) | 0.975 | 0.999 | 5.0% | 100% | 175 ms | 10 ms |
| Cascade, thorough [0.19, 0.84] | 0.971 | 0.991 | 2.5% | 8% | 363 ms | 74 ms |

What the run shows:

* **Accuracy (RQ1).** The thorough cascade recovers 88% of the transformer's F1 advantage while
  escalating 8% of documents. On `val`, cascades that escalate 20-30% *exceed* the transformer
  (F1 0.987 vs 0.983): the two stages make different errors.
* **Cost: the report's premise does not hold as specified.** The heuristic stage needs a full GPT-2
  small pass (124M parameters plus a 50k-word vocabulary projection), which costs *more* than the
  RoBERTa-base classifier (also 124M parameters, no vocabulary projection) on a 512-token window. On
  GPU the heuristic stage is bound by spaCy. So the cascade is slower than the transformer alone, and
  the "≥50% faster" success criterion (P3) is not met with GPT-2 small as the reference model.
* **Robustness.** Out of distribution the transformer collapses: on MAGE OOD its FPR is 69% (F1 0.73),
  while the heuristic keeps F1 0.86. On student essays (fairness set) the transformer flags 52% of
  native and 64% of non-native human essays; the heuristic flags 2.5% and 17%.
* **Fairness (RQ4).** Because the second stage is itself biased, the report's disparity-only constraint
  (Δ-FPR ≤ transformer's) is met by escalating *everything*, which raises native FPR to ~50%. Tuning
  therefore uses a worst-group constraint by default (non-native FPR ≤ transformer's); the disparity
  form is available with `--fairness disparity`, and both are recorded in `reports/threshold_tuning.json`.
* **Generalisation (RQ3).** With Llama removed from training, the full cascade (F1 0.754) beats the
  transformer (0.698) on Llama texts. The heuristic's leave-one-out recall drops sharply only for BLOOMZ
  (0.94 → 0.32).
* **Calibration (RQ2).** Isotonic calibration gives a well-calibrated `p_h` (ECE 0.015 on test) but does
  not improve cascade F1 at matched escalation rates on this data.

## Implementation notes and deviations from the report

* **Held-out generator family** defaults to **Mistral** (RAID `mistral`, `mistral-chat`): it is removed
  from training and calibration and appears only in the `heldout_generator` split. Public benchmarks
  contain no Claude texts; Claude, GPT-4 and Llama 3 generations can be added with `generate_custom.py`.
* **Fairness set** is small (91 TOEFL, 158 native essays) and about a third of the TOEFL essays are under
  100 words. For the audit all essays are scored regardless of length; the deployed system would report
  short ones as "insufficient text". The set is split 50/50 into `fair_val` (used by threshold tuning for
  the Δ-FPR constraint) and `fair_test` (reported).
* **Corpus balance**: human and AI counts are equal within every (source, domain, length bucket)
  stratum, and the AI side is spread across generator families as evenly as availability allows.
* **HC3 de-tokenisation**: parts of HC3 are PTB-tokenised (`do n't`, ` .`); this is undone before use so
  that spacing cannot leak the label.
* **POS bigrams**: the 20 most informative bigrams (mutual information on the training split) are learned
  with the heuristic model, giving 18 + 20 = 38 features per document.
* **Fine-tuning batch**: 4 per step × 4 accumulation steps (effective 16) to fit an 8 GB GPU.
* **Claude generations** use the model's default decoding; current Claude models do not accept
  temperature / top-p.
