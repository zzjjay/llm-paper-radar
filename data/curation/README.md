# Curation log

Append-only JSONL of the user's accept/reject decisions on papers surfaced
by the daily digest. The point is **active learning**: over time, the
patterns in these files inform tuning of `seeds.yaml`, `config.yaml`
prefilter weights, and `prompts/relevance.md` few-shot anchors.

Both files are tracked in git on purpose — they are the compounding asset
of this radar. Losing them resets months of curation judgement.

## `accepted.jsonl`

One line per `scripts/seed_add.py` success. Schema:

```jsonc
{
  "ts": "2026-05-21T08:42:00+00:00",  // when the accept happened
  "arxiv_id": "2605.19561",
  "name": "TORQ",                      // short name written into seeds.yaml
  "bucket": "ptq",                     // one of the six valid buckets
  "bucket_source": "scored_cache",     // how we picked the bucket
                                       //   "scored_cache" = reused Haiku's prior call
                                       //   "haiku"        = fresh Haiku judgement
                                       //   "cli"          = --bucket flag override
  "action": "seed_add",
  "note": "..."                        // optional inline comment
}
```

## `rejected.jsonl`

One line per `scripts/seed_reject.py` success. Schema:

```jsonc
{
  "ts": "2026-05-21T08:43:00+00:00",
  "arxiv_id": "2605.XXXXX",
  "title": "...",                      // best-effort lookup from local cache
  "reason": "actually a survey",       // free text, <=200 chars
  "bucket_when_surfaced": "ptq",       // what bucket the LLM put it in (for debugging
                                       //   the rubric vs the routing)
  "action": "reject",
  "blacklist_added": ["survey paper"]  // any prefilter blacklist phrases
                                       //   added in the same call (optional)
}
```

## Reviewing the data

```bash
# Recent rejects, freshest first
tail -20 data/curation/rejected.jsonl | jq -r '.ts + "  " + .reason + "  — " + (.title // "?")'

# Count rejects by bucket — which bucket is the rubric over-permissive in?
jq -r '.bucket_when_surfaced // "?"' data/curation/rejected.jsonl | sort | uniq -c | sort -rn

# All blacklist phrases added through rejects (audit before next prompt tuning)
jq -r '.blacklist_added // [] | .[]' data/curation/rejected.jsonl | sort -u
```
