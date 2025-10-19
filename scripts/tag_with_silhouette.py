#!/usr/bin/env python3
"""
Build a lightweight "architectural silhouette" from a JSON ledger (silhouette corpus) and tag a target chatlog with the best-matching silhouette cluster.

This is intentionally dependency-light (standard library only) and uses simple token Jaccard overlap and greedy clustering to form silhouette clusters.

Usage:
  python scripts/tag_with_silhouette.py --silhouette PATH --target PATH --output PATH

Outputs a JSON array of target items with added keys: `silhouette_cluster_id`, `silhouette_score`, `silhouette_keywords`.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Set

STOPWORDS = {
    "the","a","an","and","or","but","if","then","is","are","to","of","in","on","for",
    "with","that","this","it","as","i","you","we","they","he","she","my","your","our",
}

WORD_RE = re.compile(r"\b[\w']{2,}\b", flags=re.IGNORECASE)

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    toks = [m.group(0).lower() for m in WORD_RE.finditer(text)]
    return [t for t in toks if t not in STOPWORDS]

def tokens_set_of_item(item: dict) -> Set[str]:
    for k in ("text","content","message","body","utterance"):
        if k in item and isinstance(item[k], str):
            return set(tokenize(item[k]))
    # try other fields
    for k in ("raw","data"):
        if k in item and isinstance(item[k], str):
            return set(tokenize(item[k]))
    # fallback to stringified item
    return set(tokenize(json.dumps(item)))

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def build_silhouette(items: List[dict], min_jaccard: float = 0.25):
    clusters = []  # each: {'tokens': set, 'members': [idx], 'labels': Counter}
    for idx, item in enumerate(items):
        toks = tokens_set_of_item(item)
        placed = False
        # greedy assign to first cluster with enough overlap
        for c in clusters:
            score = jaccard(toks, c['tokens'])
            if score >= min_jaccard:
                c['members'].append(idx)
                c['tokens'] |= toks
                # collect speaker labels if present
                for k in ('speaker','role','author','from'):
                    if k in item:
                        c['labels'][str(item[k]).lower()] += 1
                placed = True
                break
        if not placed:
            labels = Counter()
            for k in ('speaker','role','author','from'):
                if k in item:
                    labels[str(item[k]).lower()] += 1
            clusters.append({'tokens': set(toks), 'members':[idx], 'labels': labels})
    # compute representative keywords per cluster
    for i, c in enumerate(clusters):
        kw_counter = Counter()
        for m_idx in c['members']:
            toks = tokens_set_of_item(items[m_idx])
            kw_counter.update(toks)
        # top keywords
        c['keywords'] = [k for k, _ in kw_counter.most_common(10)]
        c['id'] = i
        # cluster size
        c['size'] = len(c['members'])
    return clusters

def tag_target(target_items: List[dict], clusters: List[dict]):
    out = []
    counts = Counter()
    for item in target_items:
        toks = tokens_set_of_item(item)
        best_id = None
        best_score = 0.0
        for c in clusters:
            score = jaccard(toks, c['tokens'])
            if score > best_score:
                best_score = score
                best_id = c['id']
        item_out = dict(item)
        item_out['silhouette_cluster_id'] = best_id
        item_out['silhouette_score'] = round(float(best_score), 4)
        if best_id is not None:
            item_out['silhouette_keywords'] = clusters[best_id]['keywords']
            counts[best_id] += 1
        else:
            item_out['silhouette_keywords'] = []
        out.append(item_out)
    return out, counts

def load_json(path: Path):
    raw = json.loads(path.read_text(encoding='utf8'))
    if isinstance(raw, dict) and any(k in raw for k in ('items','messages','turns','entries')):
        for k in ('items','messages','turns','entries'):
            if k in raw:
                return raw[k]
    if isinstance(raw, list):
        return raw
    return [raw]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--silhouette', '-s', required=True, help='JSON ledger to build silhouette from')
    p.add_argument('--target', '-t', required=True, help='Target chatlog JSON to tag')
    p.add_argument('--output', '-o', required=True, help='Output JSON path')
    p.add_argument('--min_jaccard', type=float, default=0.25)
    args = p.parse_args()
    s_path = Path(args.silhouette)
    t_path = Path(args.target)
    out_path = Path(args.output)
    if not s_path.exists():
        print('Silhouette not found:', s_path)
        raise SystemExit(2)
    if not t_path.exists():
        print('Target not found:', t_path)
        raise SystemExit(2)
    s_items = load_json(s_path)
    t_items = load_json(t_path)
    clusters = build_silhouette(s_items, min_jaccard=args.min_jaccard)
    tagged, counts = tag_target(t_items, clusters)
    out_path.write_text(json.dumps(tagged, ensure_ascii=False, indent=2), encoding='utf8')
    print(f'Built {len(clusters)} silhouette clusters (min_jaccard={args.min_jaccard})')
    print('Top clusters by tagged count:')
    for cid, cnt in counts.most_common(10):
        kws = clusters[cid]['keywords'][:6]
        print(f'  cluster {cid}: size={clusters[cid]["size"]}, tagged={cnt}, keywords={kws}')

if __name__ == '__main__':
    main()
