#!/usr/bin/env python3
"""
Flatten an exported chat archive (folder) into a simple JSON ledger (list of message dicts).

Designed to work with the folder you attached (contains `conversations.json`, `shared_conversations.json`, `chat.html`, etc.).

Usage:
  python scripts/flatten_chat_export.py --src "C:\path\to\chat_export_folder" --out data/flattened_ledger.json

The script will attempt to locate common files (`conversations.json`, `shared_conversations.json`, `conversations.html`) and
extract message-like records. The output is a JSON list suitable as input to `scripts/parse_corpus.py` and `scripts/tag_with_silhouette.py`.
"""
import argparse
import json
from pathlib import Path
from typing import List


def load_json_file(p: Path):
    try:
        return json.loads(p.read_text(encoding='utf8'))
    except Exception:
        return None


def extract_from_conversations_json(obj) -> List[dict]:
    out = []
    # heuristic: obj may be dict with 'conversations' or list of conversations
    if isinstance(obj, dict):
        for k in ('conversations', 'items', 'entries', 'messages'):
            if k in obj and isinstance(obj[k], list):
                obj = obj[k]
                break

    if isinstance(obj, list):
        for conv in obj:
            # conversation may have 'messages' or 'items'
            msgs = None
            for mk in ('messages', 'items', 'turns'):
                if mk in conv and isinstance(conv[mk], list):
                    msgs = conv[mk]
                    break
            if msgs is None and isinstance(conv, dict):
                # maybe conv is already a message
                # try common message fields
                if any(k in conv for k in ('text','content','message','body','utterance')):
                    out.append(conv)
                continue
            if msgs:
                for m in msgs:
                    out.append(m)
    return out


def flatten_folder(src: Path) -> List[dict]:
    files = {p.name.lower(): p for p in src.iterdir() if p.is_file()}
    candidates = []
    for name in ('conversations.json', 'shared_conversations.json', 'conversations.html', 'chat.html', 'conversations.jsonl'):
        if name in files:
            candidates.append(files[name])

    # fallback: any .json files
    if not candidates:
        candidates = [p for p in src.iterdir() if p.suffix.lower() == '.json']

    ledger = []
    for f in candidates:
        data = load_json_file(f)
        if data is None:
            continue
        extracted = extract_from_conversations_json(data)
        if extracted:
            ledger.extend(extracted)
        else:
            # if top-level was a dict/list of messages, try to add
            if isinstance(data, list):
                ledger.extend(data)
            elif isinstance(data, dict):
                # try to find nested lists
                for v in data.values():
                    if isinstance(v, list):
                        ledger.extend(v)
    # de-duplicate by stringified content to avoid duplicates
    seen = set()
    deduped = []
    for item in ledger:
        s = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if s in seen:
            continue
        seen.add(s)
        deduped.append(item)
    return deduped


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='Source folder of exported chat archive')
    p.add_argument('--out', required=True, help='Output JSON ledger path')
    args = p.parse_args()
    src = Path(args.src)
    out = Path(args.out)
    if not src.exists() or not src.is_dir():
        print('Source folder not found or not a directory:', src)
        raise SystemExit(2)
    ledger = flatten_folder(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding='utf8')
    print(f'Wrote {len(ledger)} ledger items to {out}')


if __name__ == '__main__':
    main()
