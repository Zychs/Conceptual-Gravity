<!-- GitHub Copilot-specific instructions for agents working in this workspace -->
# Copilot instructions — Conceptual Gravity

This repository contains experimental assets for a project named "Conceptual Gravity." There is no conventional source tree (no package.json, pyproject.toml, or README). Use these notes to be immediately productive when making changes.

Key facts and layout
- Top-level folders:
  - `Conceptual Gravity/Conceptual Gravity/` — main workspace folder.
  - `Conceptual Gravity/Conceptual Gravity/addvar-0.0.1-dev-env/` — contains large ZIP assets (model/data):
    - `Hey-ADHD_en_android_v3_0_0.zip`
    - `vosk-model-en-us-0.42-gigaspeech.zip`
  - `Conceptual Gravity/Conceptual Gravity/addvar creation/` — currently empty.
- No language/runtime manifests detected. Treat the repo as an asset collection rather than a runnable app unless the user provides more files.

What the agent should assume and prioritize
- Assume this project stores and versions large data/model artifacts (zip files). Do not modify those binary assets unless asked.
- Prefer non-invasive edits: add documentation, repository metadata, CI hints, or small helper scripts that help the human maintainer.
- If adding code, ask (via a PR description comment) what runtime, package manager, and target environment the human prefers.

Practical guidance and examples
- When documenting, mention the exact paths to assets. Example: "Model ZIPs live in `Conceptual Gravity/addvar-0.0.1-dev-env/` and should not be re-compressed." 
- When adding CI or build files, provide safe defaults and document required human steps to opt into running them (for example: "this repo has no package.json — add one only after confirming language/runtime").
- If a change requires installing dependencies or running code, show commands but do not execute them. Example PowerShell snippet for listing large files:

  ```powershell
  Get-ChildItem -Path 'Conceptual Gravity' -Recurse -File | Sort-Object Length -Descending | Select-Object -First 20 Name,Length,FullName
  ```

Patterns and conventions discovered
- The repository currently behaves like an asset store (zip models) rather than a source code project. Expect sparse text files and a minimal Git footprint (see `.gitattributes`).
- Keep changes minimal and reversible. Use small commits with clear messages referencing the asset paths.

When to ask the human
- Before creating language-specific source files (package.json, pyproject.toml, setup.py).
- Before modifying or removing any ZIP or binary asset.
- When you need credentials, external services, or to run heavy jobs (model extraction, training, etc.).

Reference files
- `.gitattributes` — repo-level attributes present at repository root.
- `Conceptual Gravity/addvar-0.0.1-dev-env/` — contains model/data zip files.

If anything here is out of date or you need deeper code context, create a short PR and ask the maintainer to attach the missing source files or a README describing expected runtimes and workflows.
