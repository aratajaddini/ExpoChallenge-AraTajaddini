"""Build the chatbot knowledge-base index from Markdown files in KB_DIR."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend import config

# Three backticks, built at runtime so the source never contains a literal fence.
_FENCE = "`" * 3
_FENCE_RE = re.compile(rf"^\s*{re.escape(_FENCE)}")
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")


@dataclass
class Chunk:
    """One retrievable unit of knowledge."""

    source: str
    section: str
    text: str


def split_sections(md: str) -> list[tuple[str, str]]:
    """Split Markdown into (section_title, body) pairs, ignoring headings inside code fences."""
    sections: list[tuple[str, list[str]]] = []
    title = "Intro"
    body: list[str] = []
    in_fence = False

    for line in md.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            body.append(line)
            continue
        m = None if in_fence else _HEADING_RE.match(line)
        if m:
            if any(s.strip() for s in body):
                sections.append((title, body))
            title = m.group(2)
            body = []
        else:
            body.append(line)

    if any(s.strip() for s in body):
        sections.append((title, body))
    return [(t, "\n".join(b).strip()) for t, b in sections]


def pack_words(text: str, max_words: int, overlap: int) -> list[str]:
    """Split text into word windows of max_words with a fixed overlap."""
    words = text.split()
    if len(words) <= max_words:
        return [" ".join(words)] if words else []

    step = max(1, max_words - overlap)
    out: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + max_words]
        if not window:
            break
        out.append(" ".join(window))
        if start + max_words >= len(words):
            break
    return out


def build_chunks(kb_dir: Path, max_words: int, overlap: int) -> list[Chunk]:
    """Read every .md file in kb_dir and turn it into chunks."""
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        md = path.read_text(encoding="utf-8")
        for section, body in split_sections(md):
            for piece in pack_words(body, max_words, overlap):
                chunks.append(Chunk(source=path.name, section=section, text=piece))
    return chunks


def embed(texts: list[str], model_name: str) -> np.ndarray:
    """Encode texts into L2-normalised float32 vectors."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vecs = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return vecs.astype("float32")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Build the chatbot KB index.")
    ap.add_argument("--kb-dir", type=Path, default=config.KB_DIR)
    ap.add_argument("--out", type=Path, default=config.KB_INDEX)
    ap.add_argument("--model", default=config.EMBED_MODEL)
    ap.add_argument("--max-words", type=int, default=config.KB_CHUNK_WORDS)
    ap.add_argument("--overlap", type=int, default=config.KB_CHUNK_OVERLAP)
    ap.add_argument("--dry-run", action="store_true", help="Chunk only, no embeddings.")
    args = ap.parse_args()

    if args.overlap >= args.max_words:
        raise SystemExit("--overlap must be smaller than --max-words")

    chunks = build_chunks(args.kb_dir, args.max_words, args.overlap)
    if not chunks:
        raise SystemExit(f"No Markdown chunks found in {args.kb_dir}")

    for i, c in enumerate(chunks):
        print(f"[{i:3d}] {c.source} :: {c.section} ({len(c.text.split())}w)")
    print(
        f"\n{len(chunks)} chunks, max {max(len(c.text.split()) for c in chunks)} words"
    )

    if args.dry_run:
        return

    vectors = embed([f"{c.section}\n{c.text}" for c in chunks], args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        vectors=vectors,
        chunks=np.array(
            json.dumps([c.__dict__ for c in chunks], ensure_ascii=False), dtype=object
        ),
        model=np.array(args.model, dtype=object),
    )
    print(f"wrote {args.out} ({vectors.shape[0]}x{vectors.shape[1]})")


if __name__ == "__main__":
    main()
