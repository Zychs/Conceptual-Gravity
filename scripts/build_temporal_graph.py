#!/usr/bin/env python3
"""
Build a temporal + spatial graph from a labeled ledger JSON.

Outputs CSVs: nodes.csv (id, timestamp, speaker, text, embedding...) and edges.csv (source,target,type,weight)

By default this script will generate random embeddings (deterministic seed) so you can inspect graph structure
without heavy ML dependencies. If `sentence_transformers` is installed, it will use a small model to encode texts.
"""
import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import List

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST_MODEL = True
except Exception:
    HAS_ST_MODEL = False

def load_ledger(path: Path):
    raw = json.loads(path.read_text(encoding='utf8'))
    if isinstance(raw, dict) and any(k in raw for k in ('items','messages','turns','entries')):
        for k in ('items','messages','turns','entries'):
            if k in raw:
                return raw[k]
    if isinstance(raw, list):
        return raw
    return [raw]

def extract_text(item):
    for k in ('text','content','message','body','utterance'):
        if k in item and isinstance(item[k], str) and item[k].strip():
            return item[k].strip()
    # nested content.parts
    content = item.get('content') or item.get('message')
    if isinstance(content, dict):
        parts = content.get('parts')
        if isinstance(parts, list):
            joined = ' '.join([p for p in parts if isinstance(p, str)])
            if joined.strip():
                return joined
        up = content.get('user_profile')
        if isinstance(up, str) and up.strip():
            return up
    return ''

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', default='data/labeled_flat.json')
    p.add_argument('--outdir', '-o', default='data/temporal_graph')
    p.add_argument('--k', type=int, default=8, help='k for spatial k-NN')
    p.add_argument('--dim', type=int, default=64, help='embedding dim for random embeddings')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    inp = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ledger = load_ledger(inp)
    # sort by timestamp if present
    def ts(it):
        t = None
        for k in ('create_time','timestamp_','update_time'):
            if k in it:
                try:
                    t = float(it[k])
                    break
                except Exception:
                    pass
        return t if t is not None else 0.0

    items = sorted(ledger, key=ts)
    texts = [extract_text(it) for it in items]
    speakers = [ (it.get('inferred_speaker') or it.get('author',{}).get('role') or it.get('speaker') or '') for it in items]

    # embeddings
    if HAS_ST_MODEL:
        print('Using sentence-transformers to encode texts')
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embs = np.array(model.encode(texts, show_progress_bar=False))
    else:
        print('sentence_transformers not available, using deterministic random embeddings')
        rnd = random.Random(args.seed)
        rng = np.random.RandomState(args.seed)
        # deterministic random Gaussian
        embs = rng.randn(len(texts), args.dim).astype(float)

    # write nodes.csv
    nodes_path = outdir / 'nodes.csv'
    with nodes_path.open('w', newline='', encoding='utf8') as fh:
        writer = csv.writer(fh)
        header = ['id','index','timestamp','speaker','text'] + [f'emb_{i}' for i in range(embs.shape[1])]
        writer.writerow(header)
        for idx, it in enumerate(items):
            row = [it.get('id') or f'node-{idx}', idx, ts(it), speakers[idx], texts[idx]] + embs[idx].tolist()
            writer.writerow(row)

    # build spatial k-NN (brute force for now)
    n = len(embs)
    edges = []  # tuples (src_idx, dst_idx, type, weight)
    # compute pairwise distances
    dists = np.linalg.norm(embs[:,None,:] - embs[None,:,:], axis=2)
    for i in range(n):
        # exclude self
        idxs = np.argsort(dists[i])
        k = args.k + 1
        chosen = [j for j in idxs if j != i][:args.k]
        for j in chosen:
            edges.append((i, j, 'spatial', float(dists[i,j])))

    # temporal edges (evenly link sequential turns)
    for i in range(n-1):
        edges.append((i, i+1, 'temporal', 1.0))

    # write edges.csv
    edges_path = outdir / 'edges.csv'
    with edges_path.open('w', newline='', encoding='utf8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['source','target','type','weight'])
        for s,t,typ,w in edges:
            writer.writerow([s,t,typ,w])

    print(f'Wrote nodes to {nodes_path} and edges to {edges_path} (n={n}, edges={len(edges)})')

if __name__ == '__main__':
    main()
