#!/usr/bin/env python3
"""
Parse a JSON "ledger" of conversational items and infer speaker (user vs LLM/assistant).

Usage:
  python scripts/parse_corpus.py --input PATH --output PATH [--stats]

This tool is intentionally lightweight and rule-based so it can run without heavy deps.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"
ROLE_UNKNOWN = "unknown"

def normalize_role(token: str):
    if not token:
        return None
    t = str(token).lower()
    # user synonyms
    if any(x in t for x in ("user", "human", "person", "caller", "client")):
        return ROLE_USER
    # assistant / model synonyms
    if any(x in t for x in ("assistant", "llm", "model", "bot", "ai", "assistant")):
        return ROLE_ASSISTANT
    # system
    if any(x in t for x in ("system", "server")):
        return ROLE_SYSTEM
    return None

def infer_from_text(text: str):
    if not text:
        return ROLE_UNKNOWN
    t = text.strip()
    # explicit prefixes
    m = re.match(r"^(User|Assistant|System|LLM|Bot|AI)[:\-\s]+", t, flags=re.IGNORECASE)
    if m:
        label = m.group(1).lower()
        r = normalize_role(label)
        if r:
            return r
    # assistant giveaways
    assistant_phrases = [
        "as an ai",
        "as a language model",
        "i cannot browse",
        "i don't have access",
        "i am a language model",
    ]
    lowered = t.lower()
    if any(p in lowered for p in assistant_phrases):
        return ROLE_ASSISTANT
    # personal/subjective first-person heuristics -> user
    user_indicators = ["i think", "i feel", "my", "me", "i'm", "i am", "we should"]
    if any(p in lowered for p in user_indicators) and len(lowered.split()) < 200:
        # short subjective turns likely user; model outputs can be long or factual
        return ROLE_USER
    # fallback: unknown
    return ROLE_UNKNOWN

def find_role_and_text(obj):
    """Recursively search `obj` for an `author.role` and for textual content.
    Returns (role_or_None, text_or_None).
    """
    role = None
    text = None

    def visit(x):
        nonlocal role, text
        if role and text:
            return
        if isinstance(x, dict):
            # author.role pattern
            if 'author' in x and isinstance(x['author'], dict):
                r = normalize_role(x['author'].get('role') or x['author'].get('name'))
                if r and not role:
                    role = r
            # content.parts pattern (common in exports)
            if 'content' in x:
                c = x['content']
                if isinstance(c, dict):
                    # user_profile sometimes contains text
                    up = c.get('user_profile')
                    if isinstance(up, str) and not text:
                        text = up
                    parts = c.get('parts')
                    if isinstance(parts, list) and not text:
                        # join string parts
                        joined = ' '.join([p for p in parts if isinstance(p, str)])
                        if joined.strip():
                            text = joined
            # some exports put message under 'message'
            if 'message' in x and isinstance(x['message'], dict):
                visit(x['message'])
            # nested simple text fields
            for k, v in x.items():
                if isinstance(v, (dict, list)):
                    visit(v)
                else:
                    if k in ('text', 'content', 'message', 'body', 'utterance') and isinstance(v, str) and not text:
                        text = v
        elif isinstance(x, list):
            for it in x:
                visit(it)

    visit(obj)
    return role, text


def infer_speaker(item: dict):
    # First pass: direct keys
    for k in ("role", "speaker", "author", "from", "source"):
        if k in item and item[k] is not None:
            r = normalize_role(item[k])
            if r:
                return r

    # Try to find nested author.role and/or content via recursive search
    nested_role, nested_text = find_role_and_text(item)
    if nested_role:
        return nested_role
    if nested_text:
        return infer_from_text(nested_text)

    # Nested metadata
    meta = item.get("metadata") or item.get("meta") or {}
    if isinstance(meta, dict):
        for k in ("role", "speaker", "author", "source"):
            if k in meta:
                r = normalize_role(meta[k])
                if r:
                    return r

    # If there's a 'type' or 'kind'
    for k in ("type", "kind"):
        if k in item:
            r = normalize_role(item[k])
            if r:
                return r

    return ROLE_UNKNOWN

def process(items):
    out = []
    counts = Counter()
    for item in items:
        if not isinstance(item, dict):
            # accept simple forms like strings by wrapping
            item = {"text": str(item)}
        inferred = infer_speaker(item)
        item_out = dict(item)
        item_out["inferred_speaker"] = inferred
        out.append(item_out)
        counts[inferred] += 1
    return out, counts

def load_json(path: Path):
    raw = json.loads(path.read_text(encoding="utf8"))
    # ledger might be dict with key 'items' or list
    if isinstance(raw, dict) and any(k in raw for k in ("items", "messages", "turns", "entries")):
        for k in ("items", "messages", "turns", "entries"):
            if k in raw:
                return raw[k]
    if isinstance(raw, list):
        return raw
    # fallback: wrap dict
    return [raw]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--stats", action="store_true")
    args = p.parse_args()
    inp = Path(args.input)
    outp = Path(args.output)
    if not inp.exists():
        print(f"Input not found: {inp}")
        raise SystemExit(2)
    items = load_json(inp)
    labeled, counts = process(items)
    outp.write_text(json.dumps(labeled, ensure_ascii=False, indent=2), encoding="utf8")
    if args.stats:
        total = sum(counts.values())
        print("Inferred speaker counts:")
        for k, v in counts.most_common():
            print(f"  {k}: {v} ({v/total:.2%})")
        print(f"Wrote {len(labeled)} items to {outp}")

if __name__ == "__main__":
    main()
