You are writing a paper summary for an LLM inference optimization researcher (who is familiar with quantization / compression / systems optimization).

Requirements:
1. summary: 3-5 sentence English summary covering (a) what problem it solves (b) core method (c) main results.
   No filler phrases like "this paper proposes a novel method...". Get directly to the technical content.
2. highlights: 2-4 bullet points, each ≤25 words. Must include concrete numbers (compression ratio / speedup / accuracy delta, etc.) when present in the abstract.
   Use emoji prefixes: 🎯 Method / 📊 Result / 💡 Innovation / ⚠️ Limitation / 🔧 Engineering
3. related_methods: up to 3 prior methods that this paper most directly compares against, builds on, or supersedes, drawn from the abstract.
   - `name`: the canonical short name as it appears in the abstract (e.g. "GPTQ", "AWQ", "SmoothQuant", "MLA", "EAGLE-2"). Preserve casing.
   - `relation`: ≤12 words describing the relationship, e.g. "baseline outperformed by 1.2 PPL at W4", "extends with rotation-aware calibration", "supersedes; same KV-cache budget".
   - `arxiv_id`: the arXiv ID if you are certain (e.g. "2210.17323"); otherwise null. Do NOT guess.
   - If the abstract names no comparable prior methods, return [].
4. Preserve technical English terms verbatim (e.g. GPTQ, FP8, KV cache). Do not paraphrase well-known names.
5. If the abstract lacks information to support a point, omit it. Do not fabricate.

Return JSON only: {"summary": str, "highlights": list[str], "related_methods": list[{"name": str, "relation": str, "arxiv_id": str | null}]}
