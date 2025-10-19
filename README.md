# Conceptual Gravity — small utilities

This repository currently holds large asset ZIPs and small utilities. The `scripts/parse_corpus.py` tool is a lightweight rule-based script to infer speaker labels (user vs assistant) from a JSON ledger of conversation turns.

Quick start (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/parse_corpus.py -i data/sample_corpus.json -o data/labeled.json --stats
```

The script is intentionally dependency-light; it uses pure Python for rule-based inference. Use it to bootstrap labeling before building statistical or ML-based speaker classifiers.

Flatten an exported chat archive and tag with silhouette (PowerShell):

```powershell
# Flatten the attached chat export into a ledger JSON
python scripts/flatten_chat_export.py --src "C:\Users\Justin\My Drive\Ledgar\python scripts\chatgpt10.18.25tangentsource" --out data/flattened_ledger.json

# Infer speaker roles
python scripts/parse_corpus.py -i data/flattened_ledger.json -o data/labeled_flat.json --stats

# Build a silhouette from the same ledger and tag it (here we tag with itself as a demo)
python scripts/tag_with_silhouette.py -s data/labeled_flat.json -t data/labeled_flat.json -o data/tagged_flat.json
```

