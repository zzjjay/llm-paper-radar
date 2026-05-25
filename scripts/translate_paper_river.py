"""Translate a Chinese paper-river `.org` file to its English sibling.

Two usage modes:

    # 1. Translate one specific file (skips if `_en.org` already exists)
    uv run python scripts/translate_paper_river.py paper-river/GSQ-2604.18556.org

    # 2. Scan paper-river/ and translate every *.org missing an `_en.org` sibling
    uv run python scripts/translate_paper_river.py --all

    # Add --overwrite to force re-translation when the `_en.org` sibling exists.

Output convention: `<basename>.org` (zh, the input) → `<basename>_en.org`.
Files already ending in `_en.org` are skipped. Already-existing `_en.org`
siblings are also skipped unless `--overwrite` is passed.

Called from daily.sh's post-render step so every new Chinese paper-river
file gets an English version automatically on the next daily run.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Make the project root importable so we can reuse pipeline.llm_client
# regardless of where the script is invoked from. scripts/ has no
# __init__.py so `-m scripts.foo` is not an option.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click  # noqa: E402

from pipeline.llm_client import LLMClient  # noqa: E402

# Sonnet is plenty for translation; Opus would be overkill (Opus is reserved
# for compression-paper summarization in this pipeline).
TRANSLATION_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You translate Chinese Org-mode paper-river analyses into
English. Output a complete Org-mode document. Strict rules:

1. **Preserve verbatim** (do NOT translate):
   - All `#+keyword:` header lines except `#+filetags` (where you may add `:en:`)
   - `#+identifier`: append `-en` if not already present
   - Code blocks (`#+begin_example` ... `#+end_example` and similar). If a code
     block has Chinese inline comments, translate those comments but keep the
     structure / arrows / boxes intact.
   - Technical terms: paper names (OBD, OBS, OBQ, GPTQ, GSQ, AWQ, SmoothQuant,
     QTIP, AQLM, QuIP, BitNet, KIVI, …), formats (W4A16, MXFP4, NVFP4, INT4,
     FP8, …), benchmarks (MMLU, GSM8K, HumanEval, …), models (Llama-3-70B,
     Qwen3, Mistral, DeepSeek, …), math (`Δw`, `H^{-1}`, `O(n^3)`, `2-3 bpp`, …),
     algorithm names (Gumbel-Softmax, Cholesky, Hessian, …), tool/library names
     (GGUF, llama.cpp, vLLM, TensorRT-LLM, …).

2. **Translate** the narrative prose, section headings (`*`, `**`, `***`),
   inline comments inside code blocks, and any other natural-language content.
   Use natural English; do not transliterate Chinese idioms — render the
   intent (e.g. "可这压缩有个怪现象" → "But low-precision compression has a
   peculiar split").

3. **Keep the structure 1:1**: same section count, same nesting depth, same
   ordering. The English file should be readable in parallel with the Chinese
   original.

4. **Length parity**: aim for roughly the same line count; slight expansion
   (English needs more articles/connectors) is fine. Do not summarize or skip.

5. Return ONLY the translated Org file. No preamble, no explanation, no
   markdown fences. The first line of your output must be `#+title: …`
   (verbatim from input).
"""


async def translate_file(src: Path, dst: Path) -> None:
    client = LLMClient(model=TRANSLATION_MODEL)
    zh_text = src.read_text(encoding="utf-8")
    # No JSON schema needed — we want raw Org-mode text back. Reach through
    # to the underlying anthropic client directly rather than call_json.
    msg = await client.client.messages.create(
        model=client.model,
        max_tokens=16000,  # paper-river files can be ~10k tokens when verbose
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": zh_text}],
    )
    parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    if not parts:
        raise RuntimeError(f"empty response translating {src}")
    en_text = "".join(parts).strip() + "\n"
    dst.write_text(en_text, encoding="utf-8")
    print(f"translated: {src} → {dst}  ({len(en_text)} chars)")


def _candidates(root: Path) -> list[tuple[Path, Path]]:
    """Return [(src_zh, dst_en), ...] for all paper-river/*.org files that
    are not themselves `_en.org` and whose `_en.org` sibling is missing."""
    out: list[tuple[Path, Path]] = []
    for src in sorted(root.glob("*.org")):
        if src.stem.endswith("_en"):
            continue
        dst = src.with_name(f"{src.stem}_en.org")
        if not dst.exists():
            out.append((src, dst))
    return out


@click.command()
@click.argument("file", required=False, type=click.Path(path_type=Path))
@click.option("--all", "all_mode", is_flag=True, help="Scan paper-river/ for any *.org missing an _en sibling.")
@click.option("--overwrite", is_flag=True, help="Re-translate even if _en.org sibling exists.")
@click.option("--paper-river-dir", default="paper-river", type=click.Path(path_type=Path))
def main(file: Path | None, all_mode: bool, overwrite: bool, paper_river_dir: Path) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("translate_paper_river: ANTHROPIC_API_KEY not set")
    if file is None and not all_mode:
        sys.exit("translate_paper_river: pass a file path or --all")
    if file is not None and all_mode:
        sys.exit("translate_paper_river: --all and a file path are mutually exclusive")

    targets: list[tuple[Path, Path]]
    if file is not None:
        src = file
        if not src.exists():
            sys.exit(f"file not found: {src}")
        if src.stem.endswith("_en"):
            sys.exit(f"refusing to translate an _en file: {src}")
        dst = src.with_name(f"{src.stem}_en.org")
        if dst.exists() and not overwrite:
            print(f"skip: {dst} already exists (use --overwrite to force)")
            return
        targets = [(src, dst)]
    else:
        targets = _candidates(paper_river_dir)
        if overwrite:
            # In --all + --overwrite, include zh files even if _en already exists.
            extra = [
                (s, s.with_name(f"{s.stem}_en.org"))
                for s in sorted(paper_river_dir.glob("*.org"))
                if not s.stem.endswith("_en")
            ]
            seen = {t[0] for t in targets}
            targets += [t for t in extra if t[0] not in seen]
        if not targets:
            print("translate_paper_river: nothing to do (all _en siblings present)")
            return

    print(f"translating {len(targets)} file(s)...")
    for src, dst in targets:
        try:
            asyncio.run(translate_file(src, dst))
        except Exception as e:
            print(f"FAILED {src}: {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
