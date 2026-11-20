"""Custom generations for the three named generator families (Sec. 6.2.1).

Topic-matched: each prompt is derived from a human text in the corpus (its opening sentence is the
topic, its length the target), and the generation shares that text's group_id so the pair always
falls in the same split. Two temperatures (0.7, 1.0) with nucleus sampling p = 0.95 where the
provider supports sampling parameters.

    python -m data.generate_custom --provider claude --n 300
    python -m data.generate_custom --provider openai --model gpt-4o --n 300
    python -m data.generate_custom --provider local --model meta-llama/Meta-Llama-3-8B-Instruct --load-in-4bit --n 300

Output: data/raw/custom/<provider>_<model>.jsonl (appends; resumable). Then rebuild the corpus with
`python -m data.build_corpus`. API calls cost money: start with a small --n.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from veriscript.preprocess import text_hash

ROOT = Path(__file__).resolve().parent
DOMAIN_STYLE = {
    "academic": "an academic essay",
    "news": "a news-style article",
    "creative": "a short piece of creative writing",
}
# corpus domains -> target style
DOMAIN_MAP = {
    "arxiv": "academic", "abstracts": "academic", "wiki_csai": "academic", "wikipedia": "academic", "wiki": "academic",
    "peerread": "academic", "news": "news", "reddit": "creative", "books": "creative", "reddit_eli5": "academic",
    "poetry": "creative", "reviews": "creative", "wikihow": "news", "open_qa": "academic", "finance": "news", "medicine": "academic",
}


def topic_of(text: str) -> str:
    first = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    return " ".join(first.split()[:30])


def build_prompt(text: str, style_key: str, n_words: int) -> str:
    target = int(min(max(n_words, 150), 800) // 10 * 10)
    return (f"Write {DOMAIN_STYLE[style_key]} of about {target} words on the following topic. "
            f"Output only the text itself, with no title, preamble or closing remarks.\n\nTopic: {topic_of(text)}")


def make_generator(args):
    if args.provider == "claude":
        from .generators.anthropic_gen import DEFAULT_MODEL, ClaudeGenerator

        return ClaudeGenerator(args.model or DEFAULT_MODEL)
    if args.provider == "openai":
        from .generators.openai_gen import OpenAIGenerator

        return OpenAIGenerator(args.model or "gpt-4o")
    from .generators.local_gen import LocalGenerator

    return LocalGenerator(args.model or "meta-llama/Meta-Llama-3-8B-Instruct", args.load_in_4bit)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["claude", "openai", "local"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=100, help="number of human seed texts (two generations each where sampling is supported)")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    corpus = pd.read_parquet(ROOT / "processed" / "corpus.parquet")
    seeds = corpus[(corpus.label == 0) & corpus.split.isin(["train", "cal", "val", "test"])]
    seeds = seeds[seeds.domain.isin(DOMAIN_MAP)].sample(frac=1.0, random_state=args.seed)

    gen = make_generator(args)
    model_name = getattr(gen, "model", None) or getattr(gen, "model_name", "model")
    out = ROOT / "raw" / "custom" / f"{args.provider}_{str(model_name).split('/')[-1]}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["seed_id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    temps = [None] if args.provider == "claude" else [0.7, 1.0]

    n = 0
    with open(out, "a", encoding="utf-8") as fh:
        for row in seeds.itertuples():
            if n >= args.n:
                break
            if row.id in done:
                n += 1
                continue
            style = DOMAIN_MAP[row.domain]
            prompt = build_prompt(row.text, style, row.n_words)
            for t in temps:
                text, decoding = gen.generate(prompt, temperature=t) if t is not None else gen.generate(prompt)
                if not text:
                    print(f"  [refused/empty] {row.id}")
                    continue
                fh.write(json.dumps({
                    "text": text, "label": 1, "generator": str(model_name), "domain": row.domain, "style": style,
                    "decoding": decoding, "group_id": row.group_id, "seed_id": row.id, "prompt": prompt,
                }) + "\n")
                fh.flush()
            n += 1
            print(f"  {n}/{args.n}", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
