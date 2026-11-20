"""Download the public data sources of Table 6.4 into data/raw/.

    python -m data.download                 # everything
    python -m data.download --only hc3 m4   # a subset

Sources
  hc3       Hello-SimpleAI/HC3 (Hugging Face Hub), all.jsonl                        ~70 MB
  m4        mbzuai-nlp/M4 (GitHub), English files only                              ~650 MB
  raid      liamdugan/raid (Hugging Face parquet conversion of train.csv)            ~2.3 GB
  mage      yaful/MAGE (Hugging Face Hub), out-of-distribution test sets            ~6 MB
  fairness  Weixin-Liang/ChatGPT-Detector-Bias (GitHub), human essay sets           <1 MB
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

RAW = Path(__file__).resolve().parent / "raw"

M4_URL = "https://raw.githubusercontent.com/mbzuai-nlp/M4/main/data/{name}.jsonl"
M4_ENGLISH = [
    "arxiv_bloomz", "arxiv_chatGPT", "arxiv_cohere", "arxiv_davinci", "arxiv_dolly", "arxiv_flant5",
    "peerread_bloomz", "peerread_chatgpt", "peerread_cohere", "peerread_davinci", "peerread_dolly", "peerread_llama",
    "reddit_bloomz", "reddit_chatGPT", "reddit_cohere", "reddit_davinci", "reddit_dolly", "reddit_flant5",
    "wikihow_bloomz", "wikihow_chatGPT", "wikihow_cohere", "wikihow_davinci", "wikihow_dolly2",
    "wikipedia_bloomz", "wikipedia_chatgpt", "wikipedia_cohere", "wikipedia_davinci", "wikipedia_dolly",
]
HF = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
RAID_SHARDS = [f"raid/partial-train/{i:04d}.parquet" for i in range(10)]
MAGE_FILES = ["test_ood_set_gpt.csv", "test_ood_set_gpt_para.csv"]
LIANG_URL = "https://raw.githubusercontent.com/Weixin-Liang/ChatGPT-Detector-Bias/main/Data_and_Results/Human_Data/{d}/data.json"
LIANG_SETS = {"toefl": "TOEFL_real_91", "hewlett": "HewlettStudentEssay_real_88", "college": "CollegeEssay_real_70"}


def fetch(url: str, dest: Path, force: bool = False) -> None:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"  [skip] {dest.relative_to(RAW)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for block in r.iter_content(chunk_size=1 << 20):  # decodes gzip transfer encoding
                fh.write(block)
    tmp.replace(dest)
    print(f"  [ok]   {dest.relative_to(RAW)}  ({dest.stat().st_size / 2**20:.1f} MB)")


def download_hc3(force=False):
    fetch(HF.format(repo="Hello-SimpleAI/HC3", rev="main", path="all.jsonl"), RAW / "hc3" / "all.jsonl", force)


def download_m4(force=False):
    for name in M4_ENGLISH:
        fetch(M4_URL.format(name=name), RAW / "m4" / f"{name}.jsonl", force)


def download_raid(force=False):
    for path in RAID_SHARDS:
        url = HF.format(repo="liamdugan/raid", rev="refs%2Fconvert%2Fparquet", path=path)
        fetch(url, RAW / "raid" / Path(path).name, force)


def download_mage(force=False):
    for f in MAGE_FILES:
        fetch(HF.format(repo="yaful/MAGE", rev="main", path=f), RAW / "mage" / f, force)


def download_fairness(force=False):
    for key, d in LIANG_SETS.items():
        fetch(LIANG_URL.format(d=d), RAW / "fairness" / f"{key}.json", force)


SOURCES = {"hc3": download_hc3, "m4": download_m4, "raid": download_raid, "mage": download_mage, "fairness": download_fairness}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(SOURCES), help="download only these sources")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args(argv)
    for name in args.only or SOURCES:
        print(f"== {name}")
        SOURCES[name](args.force)


if __name__ == "__main__":
    main()
