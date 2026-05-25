#!/usr/bin/env bash
# Generate a single paper-river .org file by invoking the ljg-paper-river
# Claude Code skill in headless mode (`claude -p`). The skill does multi-turn
# web research + deep lineage analysis; this wrapper just orchestrates one
# paper at a time and lets the skill write the .org file directly via the
# Write tool.
#
# Usage:
#   ./scripts/gen_paper_river.sh <arxiv-id-or-url>
#
# Behavior:
#   - Extracts the arxiv id (NN.NNNNN) from the arg.
#   - Skips immediately if paper-river/*<id>.org already exists (any name
#     pattern, dot-form or legacy dash-form).
#   - Invokes `claude -p` with --permission-mode acceptEdits so the skill
#     can run web tools and Write the resulting .org file without prompts.
#   - Non-fatal on failure (returns non-zero, caller decides).
#
# Cost: ~5-10 min per call, dozens of web fetches + multi-turn LLM work.
# Designed for batch use via scripts/auto_paper_river.py.

set -uo pipefail

ARG="${1:?arxiv id or URL required, e.g. 2604.18556 or https://arxiv.org/abs/2604.18556}"

# Extract arxiv id; accept both forms.
ID=$(echo "$ARG" | grep -oE '[0-9]{4}\.[0-9]{4,5}' | head -1)
if [[ -z "$ID" ]]; then
    echo "gen_paper_river: no arxiv id found in: $ARG" >&2
    exit 1
fi

URL="https://arxiv.org/abs/${ID}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p paper-river

# Dedup against both naming conventions.
ID_DOT="$ID"
ID_DASH="${ID//./-}"
existing=$(ls paper-river/ 2>/dev/null | grep -E "(${ID_DOT}|${ID_DASH})\.org$" || true)
if [[ -n "$existing" ]]; then
    echo "gen_paper_river: skip ${ID}, already exists: $existing"
    exit 0
fi

if ! command -v claude >/dev/null 2>&1; then
    echo "gen_paper_river: 'claude' CLI not on PATH; cannot invoke skill" >&2
    exit 2
fi

# Drive the skill non-interactively. The skill itself decides the acronym
# and section structure; we just tell it the target filename convention.
PROMPT="Run the /ljg-paper-river skill on ${URL}. When the analysis is complete,
save the resulting .org file to paper-river/<acronym>-${ID_DOT}.org using the Write
tool, where <acronym> is the paper's short identifier (e.g. 'GSQ', 'FlashAttention',
'EAGLE') in mixed case as it appears in the paper title. Do NOT print the .org
content to stdout — just report 'Saved: <path>' once the file is written.
If the skill is unavailable in this session, report that and exit."

echo "[$(date -Is)] gen_paper_river: ${ID} → invoking /ljg-paper-river"
# acceptEdits permission mode: skill can use Write + web tools without
# interactive prompts. Pipe the prompt via stdin so it isn't gobbled by
# the variadic --allowedTools flag.
printf '%s' "$PROMPT" | claude -p \
    --permission-mode acceptEdits \
    --allowedTools "Write,Edit,Read,WebSearch,WebFetch,Bash"
exit_code=$?

# Verify the file actually appeared (skill may have silently failed).
new_file=$(ls paper-river/ 2>/dev/null | grep -E "(${ID_DOT}|${ID_DASH})\.org$" | grep -v '_en\.org$' | head -1)
if [[ -z "$new_file" ]]; then
    echo "gen_paper_river: ${ID} skill ran (exit=${exit_code}) but no file was written" >&2
    exit 3
fi

# Post-process: the skill defaults `#+title:` to a slug like "paper-river-GSQ".
# Replace with the paper's real title from data/summarized/ so the .org
# self-identifies by paper name (Org exports this header verbatim).
REAL_TITLE=$(uv run python -c "
import json, sys
from pathlib import Path
for f in sorted(Path('data/summarized').glob('*.json')):
    try:
        for p in json.loads(f.read_text()):
            if (p.get('id') or '').split(':')[-1] == '${ID_DOT}':
                print((p.get('title') or '').strip())
                sys.exit(0)
    except Exception:
        continue
" 2>/dev/null)
if [[ -n "$REAL_TITLE" ]]; then
    # In-place edit; only touches the first #+title: line. Sed uses | as
    # delimiter so titles containing '/' don't need escaping.
    REAL_TITLE_ESCAPED=$(printf '%s' "$REAL_TITLE" | sed 's/[|&\\]/\\&/g')
    sed -i "0,/^#+title:/s|^#+title:.*|#+title:      ${REAL_TITLE_ESCAPED}|" "paper-river/${new_file}"
    echo "gen_paper_river: ${ID} #+title → ${REAL_TITLE:0:80}"
fi

echo "[$(date -Is)] gen_paper_river: ${ID} → paper-river/${new_file}"
