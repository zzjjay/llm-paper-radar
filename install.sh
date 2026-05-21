#!/usr/bin/env bash
# Symlink the in-repo skills into ~/.claude/skills/ so Claude Code can
# discover them. Run once after cloning; safe to re-run (replaces stale
# symlinks, leaves real directories alone).
#
# Mirrors the quark-agent-assets install pattern so muscle memory carries
# over: skill source lives in this repo under `skills/`, the runtime
# location is a symlink into `~/.claude/skills/`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLS_SRC="$REPO_ROOT/skills"
SKILLS_DST="$HOME/.claude/skills"

if [[ ! -d "$SKILLS_SRC" ]]; then
    echo "no skills/ directory in $REPO_ROOT — nothing to install" >&2
    exit 1
fi

mkdir -p "$SKILLS_DST"

installed=0
for src in "$SKILLS_SRC"/*/; do
    [[ -d "$src" ]] || continue
    name="$(basename "$src")"
    dst="$SKILLS_DST/$name"

    if [[ -L "$dst" ]]; then
        # Existing symlink — point it at this repo (replaces any prior install
        # from a different clone path).
        ln -sfn "$src" "$dst"
        echo "↻ relinked $name → $src"
    elif [[ -e "$dst" ]]; then
        # Real directory or file at the destination — refuse to clobber.
        echo "⚠ $dst exists and is NOT a symlink; remove it manually then re-run" >&2
        continue
    else
        ln -s "$src" "$dst"
        echo "✓ installed $name → $src"
    fi
    installed=$((installed + 1))
done

echo
echo "$installed skill(s) installed under $SKILLS_DST"
echo
echo "Next: start a fresh Claude Code session in this repo so the skill is"
echo "picked up. Trigger with phrases like \"scout compression papers\"."
