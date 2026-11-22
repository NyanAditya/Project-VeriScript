"""Fetch the trained VeriScript models and the GPT-2 reference model at image build time."""

import os
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(os.environ.get("VERISCRIPT_HOME", Path(__file__).resolve().parent))
models = ROOT / "models"

if (models / "heuristic" / "heuristic.joblib").exists():
    print("models already present, skipping download")
else:
    repo = os.environ["MODEL_REPO"]
    snapshot_download(repo_id=repo, local_dir=str(models),
                      allow_patterns=["heuristic/*", "transformer/roberta-base/*", "operating_points.json"])
    print(f"downloaded {repo} -> {models}")

# Reference language model of the heuristic stage (cached in HF_HOME).
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

AutoTokenizer.from_pretrained("gpt2")
AutoModelForCausalLM.from_pretrained("gpt2")
print("gpt2 cached")
