"""Print top cosine per query so KB_MIN_COSINE can be set from real data."""

from __future__ import annotations

import argparse

from backend import config
from backend.chat.retriever import search

RELEVANT = [
    "which waste classes does the model support",
    "why is glass hard to classify",
    "what is the maximum upload size",
    "how many frames are sampled from a video",
    "which header carries the api key",
    "what happens when the api key is missing",
    "where does training run",
    "how is a training run evaluated",
]

IRRELEVANT = [
    "how do I bake sourdough bread",
    "what is the capital of Portugal",
    "best exercises for lower back pain",
    "how do I renew a German residence permit",
]


def _report(label: str, queries: list[str]) -> list[float]:
    """Print the top cosine and best-matching chunk for each query."""
    tops: list[float] = []
    print(f"\n{label}")
    for q in queries:
        hits = search(q)
        top = max((h["cosine"] for h in hits), default=0.0)
        tops.append(top)
        where = f"{hits[0]['source']} / {hits[0]['section']}" if hits else "-"
        print(f"  {top:6.3f}  {q[:44]:44s}  {where}")
    return tops


def main() -> None:
    """Compare relevant and irrelevant queries and suggest a cosine floor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", help="extra query to probe")
    args = parser.parse_args()

    if not config.kb_is_built():
        raise SystemExit("kb index missing; run python -m backend.tools.build_kb")

    print(f"model: {config.EMBED_MODEL}")
    good = _report("relevant", RELEVANT)
    bad = _report("irrelevant", IRRELEVANT)
    if args.query:
        _report("custom", args.query)

    lo, hi = min(good), max(bad)
    print(f"\nlowest relevant : {lo:.3f}")
    print(f"highest noise   : {hi:.3f}")
    print(f"current setting : {config.KB_MIN_COSINE:.3f}")
    if lo > hi:
        print(f"suggested       : {(lo + hi) / 2:.3f}")
    else:
        print("no clean separation; fix the KB or the embedding model first")


if __name__ == "__main__":
    main()
