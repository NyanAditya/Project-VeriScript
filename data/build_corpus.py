"""Build the unified, balanced working corpus (Steps 1-2; Sec. 4.3.1, 6.2; Tables 6.5, 6.6).

    python -m data.build_corpus [--scale 1.0] [--heldout-family mistral]

Writes data/processed/corpus.parquet with one row per document and the attributes of Table 6.6:
id, text, label, source_dataset, domain, generator, generator_family, decoding, attack, group_id,
n_words, native, split.

Splits
  train / cal / val / test   balanced human vs AI within each (domain, length bucket) stratum,
                             AI side balanced across generator families; held-out family excluded
  heldout_generator          AI texts of the held-out family + human texts, test groups only
  raid_attack                attacked RAID generations (AI only), robustness
  mage_ood                   MAGE out-of-distribution test set
  fair_val / fair_test       native vs non-native human essays (Liang et al.), split 50/50
Splitting is by group_id (question / source document) so paired texts never cross splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from veriscript.preprocess import normalise, remove_boilerplate, text_hash, word_count

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "processed"

MIN_WORDS = 100
MAX_WORDS = 1500
LENGTH_BUCKETS = [(100, 250), (250, 500), (500, 10**9)]
TARGETS = {"train": 14000, "cal": 2000, "val": 2000, "test": 2000}  # per class (Table 6.5)
EXTRA_TARGETS = {"heldout_generator": 1000, "raid_attack": 2000, "mage_ood": 1000}


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------
def generator_family(gen: str) -> str:
    g = gen.lower()
    if g == "human":
        return "human"
    for fam, keys in [
        ("gpt", ("gpt", "davinci", "openai", "text-")),
        ("llama", ("llama",)),
        ("mistral", ("mistral", "mixtral")),
        ("mpt", ("mpt",)),
        ("cohere", ("cohere", "command")),
        ("claude", ("claude",)),
        ("bloomz", ("bloom",)),
        ("dolly", ("dolly",)),
        ("flant5", ("flan", "t5")),
    ]:
        if any(k in g for k in keys):
            return fam
    return g


def split_of(group_id: str) -> str:
    h = int(hashlib.md5(group_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 70 else "cal" if h < 80 else "val" if h < 90 else "test"


def length_bucket(n: int) -> int:
    for i, (lo, hi) in enumerate(LENGTH_BUCKETS):
        if lo <= n < hi:
            return i
    return -1


_DETOK = [
    (re.compile(r" ([.,!?;:%)\]}])"), r"\1"),
    (re.compile(r"([(\[{$]) "), r"\1"),
    (re.compile(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", re.I), r"\1"),
    (re.compile(r"\s+' s\b"), "'s"),
    (re.compile(r'" ([^"]*?) "'), r'"\1"'),
]


def detokenize(text: str) -> str:
    """Undo the PTB-style tokenisation in parts of HC3 (it would otherwise leak the label)."""
    for pat, rep in _DETOK:
        text = pat.sub(rep, text)
    return text


def clean(text: str, label: int) -> str:
    t = normalise(text)
    if label == 1:
        t = remove_boilerplate(t)
    return t


def rec(text, label, source, domain, generator, group_id, decoding="unknown", attack="none", native=None):
    return {
        "text": text,
        "label": int(label),
        "source_dataset": source,
        "domain": domain,
        "generator": generator,
        "decoding": decoding,
        "attack": attack,
        "group_id": group_id,
        "native": native,
    }


# ---------------------------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------------------------
def load_hc3() -> list[dict]:
    path = RAW / "hc3" / "all.jsonl"
    out = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            d = json.loads(line)
            src = d.get("source", "hc3")
            gid = f"hc3:{i}"
            for t in d.get("human_answers", []) or []:
                out.append(rec(detokenize(t), 0, "hc3", src, "human", gid))
            for t in d.get("chatgpt_answers", []) or []:
                out.append(rec(detokenize(t), 1, "hc3", src, "chatgpt", gid))
    return out


def _as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def load_m4() -> list[dict]:
    out = []
    for path in sorted((RAW / "m4").glob("*.jsonl")):
        domain, gen = path.stem.split("_", 1)
        gen = gen.lower()
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Field names differ between M4 files (e.g. the BLOOMZ files use text / abstract / *_reviews).
                humans = _as_list(d.get("human_text") or d.get("human_reviews") or d.get("text") or d.get("abstract"))
                machines = _as_list(d.get("machine_text") or d.get("bloom_reviews"))
                key = humans[0] if humans and isinstance(humans[0], str) else json.dumps(d.get("prompt", ""))[:200]
                gid = "m4:" + text_hash(key)[:16]
                for t in humans:
                    if isinstance(t, str):
                        out.append(rec(t, 0, "m4", domain, "human", gid))
                for t in machines:
                    if isinstance(t, str):
                        out.append(rec(t, 1, "m4", domain, gen, gid))
    return out


def load_raid(ai_keep: float = 0.08, attack_keep: float = 0.004, seed: int = 42) -> list[dict]:
    """RAID rows: all human texts; a random fraction of clean and attacked generations (memory)."""
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    out = []
    cols = ["id", "source_id", "adv_source_id", "model", "decoding", "repetition_penalty", "attack", "domain", "generation"]
    for path in sorted((RAW / "raid").glob("*.parquet")):
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            df = pf.read_row_group(rg, columns=cols).to_pandas()
            is_h = df["model"] == "human"
            clean_ai = (~is_h) & (df["attack"] == "none")
            attacked_ai = (~is_h) & (df["attack"] != "none")
            r = rng.random(len(df))
            keep = (is_h & (df["attack"] == "none")) | (clean_ai & (r < ai_keep)) | (attacked_ai & (r < attack_keep))
            for row in df[keep].itertuples(index=False):
                label = 0 if row.model == "human" else 1
                gid = f"raid:{row.source_id if label else row.id}"
                dec = f"{row.decoding}" + ("+rep" if str(row.repetition_penalty) == "yes" else "")
                out.append(rec(row.generation, label, "raid", row.domain, row.model, gid, dec, row.attack))
        print(f"  raid {path.name}: {len(out)} rows kept so far")
    return out


def load_mage() -> list[dict]:
    out = []
    for path in sorted((RAW / "mage").glob("test_ood_set_gpt*.csv")):
        df = pd.read_csv(path)
        tag = "mage_para" if "para" in path.stem else "mage"
        for i, row in enumerate(df.itertuples(index=False)):
            src = str(row.src)
            # MAGE: label 1 = human-written, 0 = machine-generated; double-check with the src tag.
            is_human = "human" in src.lower() if ("human" in src.lower() or "machine" in src.lower()) else int(row.label) == 1
            label = 0 if is_human else 1
            gen = "human" if label == 0 else "gpt-4"  # the OOD testbed's machine texts come from GPT-4
            out.append(rec(row.text, label, tag, src.split("_")[0], gen, f"{tag}:{i}",
                           attack="paraphrase" if tag == "mage_para" else "none"))
    return out


def load_fairness() -> list[dict]:
    out = []
    for key, native in [("toefl", False), ("hewlett", True), ("college", True)]:
        path = RAW / "fairness" / f"{key}.json"
        if not path.exists():
            continue
        for i, d in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            t = d["document"] if isinstance(d, dict) else d
            out.append(rec(t, 0, "fairness", key, "human", f"fair:{key}:{i}", native=native))
    return out


def load_custom() -> list[dict]:
    """Custom generations from data/generate_custom.py (optional)."""
    out = []
    for path in sorted((RAW / "custom").glob("*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                out.append(rec(d["text"], int(d.get("label", 1)), "custom", d.get("domain", "custom"),
                               d.get("generator", "unknown"), d.get("group_id", "custom:" + text_hash(d["text"])[:16]),
                               d.get("decoding", "unknown")))
    return out


# ---------------------------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------------------------
def family_balanced(df: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    """Sample n rows spreading the quota as evenly as possible over generator families."""
    if len(df) <= n:
        return df
    groups = {f: g.sample(frac=1.0, random_state=int(rng.integers(1 << 31))) for f, g in df.groupby("generator_family")}
    quota = {f: 0 for f in groups}
    remaining = n
    active = [f for f in groups]
    while remaining > 0 and active:
        share = max(remaining // len(active), 1)
        for f in list(active):
            take = min(share, len(groups[f]) - quota[f], remaining)
            quota[f] += take
            remaining -= take
            if quota[f] >= len(groups[f]):
                active.remove(f)
            if remaining == 0:
                break
    return pd.concat([groups[f].iloc[:q] for f, q in quota.items() if q > 0])


def balanced_split(pool: pd.DataFrame, per_class: int, rng) -> pd.DataFrame:
    """Equal human / AI counts in every (domain, length bucket) stratum, scaled to the target."""
    strata = []
    for key, g in pool.groupby(["source_dataset", "domain", "len_bucket"]):
        h, a = g[g.label == 0], g[g.label == 1]
        strata.append((key, h, a, min(len(h), len(a))))
    total = sum(c for *_, c in strata)
    if total == 0:
        return pool.iloc[:0]
    scale = min(1.0, per_class / total)
    parts = []
    for _, h, a, c in strata:
        k = int(round(c * scale))
        if k == 0:
            continue
        parts.append(h.sample(k, random_state=int(rng.integers(1 << 31))))
        parts.append(family_balanced(a, k, rng))
    return pd.concat(parts)


def build(scale: float = 1.0, heldout_family: str = "mistral", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    print("loading sources ...")
    rows = load_hc3() + load_m4() + load_custom()
    print(f"  hc3+m4+custom: {len(rows)}")
    rows += load_raid(seed=seed)
    main = pd.DataFrame(rows)
    other = pd.DataFrame(load_mage() + load_fairness())
    print(f"  total raw rows: {len(main)} main, {len(other)} test-only")

    def prep(df: pd.DataFrame, min_words: int) -> pd.DataFrame:
        df = df.copy()
        df["text"] = [clean(t, l) for t, l in zip(df["text"].astype(str), df["label"])]
        df["n_words"] = df["text"].map(word_count)
        df = df[(df.n_words >= min_words) & (df.n_words <= MAX_WORDS)]
        df["hash"] = df["text"].map(text_hash)
        df = df.drop_duplicates("hash")
        df["generator_family"] = df["generator"].map(generator_family)
        df["len_bucket"] = df["n_words"].map(length_bucket)
        return df

    main = prep(main, MIN_WORDS)
    print(f"  after cleaning / length filter / dedup: {len(main)}")
    main["group_split"] = main["group_id"].map(split_of)

    parts = []
    clean_main = main[main.attack == "none"]
    in_family = clean_main[(clean_main.label == 0) | (clean_main.generator_family != heldout_family)]
    for split, n in TARGETS.items():
        s = balanced_split(in_family[in_family.group_split == split], int(n * scale), rng)
        s = s.assign(split=split)
        parts.append(s)
        print(f"  {split}: {len(s)} ({(s.label == 1).sum()} AI)")
    used = set(pd.concat(parts)["hash"])

    # Held-out generator family: test groups only, plus unused human texts from test groups.
    test_pool = clean_main[(clean_main.group_split == "test") & ~clean_main["hash"].isin(used)]
    held_ai = test_pool[(test_pool.label == 1) & (test_pool.generator_family == heldout_family)]
    n_h = min(int(EXTRA_TARGETS["heldout_generator"] * scale), len(held_ai))
    held_ai = held_ai.sample(n_h, random_state=seed)
    held_h = test_pool[test_pool.label == 0]
    held_h = held_h.sample(min(n_h, len(held_h)), random_state=seed)
    parts.append(pd.concat([held_ai, held_h]).assign(split="heldout_generator"))
    print(f"  heldout_generator ({heldout_family}): {len(held_ai)} AI + {len(held_h)} human")

    attacked = main[(main.attack != "none") & (main.group_split == "test")]
    att = attacked.groupby("attack", group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(1, int(EXTRA_TARGETS["raid_attack"] * scale) // attacked.attack.nunique())), random_state=seed)
    )
    parts.append(att.assign(split="raid_attack"))
    print(f"  raid_attack: {len(att)} over {att.attack.nunique()} attacks")

    # Test-only sets.
    if len(other):
        mage = prep(other[other.source_dataset.str.startswith("mage")], MIN_WORDS)
        for tag in ("mage", "mage_para"):
            m = mage[mage.source_dataset == tag]
            k = int(EXTRA_TARGETS["mage_ood"] * scale)
            m = pd.concat([g.sample(min(len(g), k), random_state=seed) for _, g in m.groupby("label")]) if len(m) else m
            parts.append(m.assign(split="mage_ood" if tag == "mage" else "mage_ood_para"))
            print(f"  {tag}: {len(m)}")
        fair = prep(other[other.source_dataset == "fairness"], 1)  # fairness essays are kept regardless of length
        fair["split"] = [("fair_val" if int(hashlib.md5(g.encode()).hexdigest()[:8], 16) % 2 == 0 else "fair_test") for g in fair.group_id]
        parts.append(fair)
        print(f"  fairness: {len(fair)} ({int(fair.native.sum())} native)")

    corpus = pd.concat(parts, ignore_index=True)
    corpus["id"] = [f"{s}-{i:06d}" for i, s in enumerate(corpus["split"])]
    cols = ["id", "text", "label", "source_dataset", "domain", "generator", "generator_family", "decoding", "attack",
            "group_id", "n_words", "native", "split"]
    corpus = corpus[cols]
    # Sanity check: no group crosses main splits.
    main_rows = corpus[corpus.split.isin(TARGETS)]
    assert main_rows.groupby("group_id").split.nunique().max() <= 1, "group leakage between splits"
    return corpus


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the target sizes (e.g. 0.1 for a quick run)")
    ap.add_argument("--heldout-family", default="mistral")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT / "corpus.parquet"))
    args = ap.parse_args(argv)
    corpus = build(args.scale, args.heldout_family, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}: {len(corpus)} documents")
    summary = corpus.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(summary)
    fam = corpus[corpus.split.isin(TARGETS)].groupby(["generator_family"]).size()
    print(fam)
    meta = {
        "heldout_family": args.heldout_family,
        "seed": args.seed,
        "scale": args.scale,
        "counts": {f"{s}|{l}": int(v) for (s, l), v in corpus.groupby(["split", "label"]).size().items()},
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
