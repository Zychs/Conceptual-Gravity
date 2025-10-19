#!/usr/bin/env python3
"""
Compute sliding-window token entropy across a chatlog to visualize "thought pressure" diffusion.

Outputs a CSV of windowed entropy values and (optionally) a PNG plot if matplotlib is installed.

Usage:
  python scripts/entropy_diffusion.py -i data/sample_tagged.json -o data/entropy_windows.csv --window 3 --step 1

Output files:
  - CSV at the path given with columns: start_idx,end_idx,entropy,normalized_entropy,token_count,unique_tokens
  - If matplotlib is available: PNG at same path with .png extension.
"""
from __future__ import annotations
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import List, Set

# reuse tokenization from tag_with_silhouette.py
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


def tokens_set_of_item(item: dict) -> List[str]:
    for k in ("text","content","message","body","utterance"):
        if k in item and isinstance(item[k], str):
            return tokenize(item[k])
    for k in ("raw","data"):
        if k in item and isinstance(item[k], str):
            return tokenize(item[k])
    return tokenize(json.dumps(item))


def shannon_entropy_from_counts(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counter.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def sliding_window_entropy(items: List[dict], window: int, step: int):
    n = len(items)
    rows = []
    for start in range(0, max(1, n - window + 1), step):
        end = start + window
        window_items = items[start:end]
        token_counter = Counter()
        for it in window_items:
            token_counter.update(tokens_set_of_item(it))
        entropy = shannon_entropy_from_counts(token_counter)
        uniq = len(token_counter)
        if uniq > 1:
            norm = entropy / math.log2(uniq)
        else:
            norm = 0.0
        rows.append({
            'start_idx': start,
            'end_idx': end - 1,
            'entropy': round(entropy, 6),
            'normalized_entropy': round(norm, 6),
            'token_count': sum(token_counter.values()),
            'unique_tokens': uniq,
        })
    # handle short logs: also include final window if n < window
    if n > 0 and n < window:
        token_counter = Counter()
        for it in items:
            token_counter.update(tokens_set_of_item(it))
        entropy = shannon_entropy_from_counts(token_counter)
        uniq = len(token_counter)
        norm = entropy / math.log2(uniq) if uniq > 1 else 0.0
        rows = [{
            'start_idx': 0,
            'end_idx': n - 1,
            'entropy': round(entropy, 6),
            'normalized_entropy': round(norm, 6),
            'token_count': sum(token_counter.values()),
            'unique_tokens': uniq,
        }]
    return rows


def moving_average(values, window_size: int):
    if window_size <= 1:
        return list(values)
    out = []
    n = len(values)
    for i in range(n):
        start = max(0, i - (window_size - 1))
        window_vals = values[start:i+1]
        out.append(sum(window_vals) / len(window_vals))
    return out


def derivative(values):
    # simple discrete derivative (forward difference), length = len(values)
    n = len(values)
    if n == 0:
        return []
    out = [0.0] * n
    for i in range(n - 1):
        out[i] = values[i+1] - values[i]
    out[-1] = 0.0
    return out


def ascii_plot(xs, ys, height=10, width=60, label=None):
    # normalize ys to [0,1]
    if not ys:
        print('(no data to plot)')
        return
    ymin = min(ys)
    ymax = max(ys)
    if ymax - ymin < 1e-9:
        ys_n = [0.5 for _ in ys]
    else:
        ys_n = [(y - ymin) / (ymax - ymin) for y in ys]
    # downsample or map to width
    n = len(ys_n)
    if n <= width:
        idxs = list(range(n))
    else:
        idxs = [ int(i * n / width) for i in range(width) ]

    grid = [[' ' for _ in range(len(idxs))] for _ in range(height)]
    for col, i in enumerate(idxs):
        v = ys_n[i]
        row = int((height - 1) - v * (height - 1))
        grid[row][col] = '*'
    # print
    if label:
        print(label)
    for r in range(height):
        print('|' + ''.join(grid[r]) + '|')
    print('+' + '-' * len(idxs) + '+')
    # x ticks
    if idxs:
        left = xs[idxs[0]]
        right = xs[idxs[-1]]
        print(f'  {left:.1f}'.ljust(10) + f'{right:.1f}'.rjust(len(idxs)))


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
    p.add_argument('-i','--input', required=True, help='Input chat JSON')
    p.add_argument('-o','--output', required=True, help='Output CSV path (and PNG if plotting)')
    p.add_argument('--window', type=int, default=5, help='Sliding window size (messages)')
    p.add_argument('--step', type=int, default=1, help='Window step size')
    args = p.parse_args()

    inp = Path(args.input)
    out_csv = Path(args.output)
    if not inp.exists():
        print('Input not found:', inp)
        raise SystemExit(2)

    items = load_json(inp)
    rows = sliding_window_entropy(items, window=args.window, step=args.step)

    # write CSV
    from csv import DictWriter
    with out_csv.open('w', encoding='utf8', newline='') as fh:
        dw = DictWriter(fh, fieldnames=['start_idx','end_idx','entropy','normalized_entropy','token_count','unique_tokens'])
        dw.writeheader()
        for r in rows:
            dw.writerow(r)
    print(f'Wrote {len(rows)} rows to {out_csv}')

    # try plotting if matplotlib available
    try:
        import matplotlib.pyplot as plt
        xs = [ (r['start_idx'] + r['end_idx'])/2 for r in rows ]
        ys = [ r['normalized_entropy'] for r in rows ]
        plt.figure(figsize=(8,3))
        plt.plot(xs, ys, marker='o')
        plt.ylim(0,1.05)
        plt.xlabel('Message index (window center)')
        plt.ylabel('Normalized entropy')
        plt.title('Entropy diffusion (normalized)')
        png_path = out_csv.with_suffix('.png')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(png_path))
        print('Wrote plot to', png_path)
    except Exception as e:
        print('Skipping plot (matplotlib not available or failed):', e)
    # compute moving average and derivative of normalized_entropy and ASCII plot
    norm_vals = [r['normalized_entropy'] for r in rows]
    centers = [ (r['start_idx'] + r['end_idx'])/2 for r in rows ]
    ma = moving_average(norm_vals, max(1, min(3, len(norm_vals))))
    deriv = derivative(norm_vals)
    print('\nEntropy summary:')
    for i, r in enumerate(rows):
        print(f"window {i}: center={centers[i]:.1f} normalized={r['normalized_entropy']:.3f} ma={ma[i]:.3f} d={deriv[i]:.3f} uniq={r['unique_tokens']} tokens={r['token_count']}")
    print('\nASCII plot of normalized entropy:')
    ascii_plot(centers, norm_vals, height=8, width=60, label='Normalized entropy (0-1)')


if __name__ == '__main__':
    main()
