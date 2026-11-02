"""Best-effort provenance snapshot for archived experiment suites.

Writes ``provenance.json`` into a results directory before a suite starts,
recording the exact code state (sha256 of every relevant .py file), the full
argparse namespace, interpreter/torch versions, and the git commit when one
exists, so any archived result can be tied to the code that produced it.
Collection is wrapped in try/except at every step: a provenance failure must
never abort a training run.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _git_commit(root: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(root),
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _code_hashes(root: Path) -> dict:
    hashes = {}
    for sub in ("Model", "dataloader", "scripts", "scripts/attribution", "utils"):
        folder = root / sub
        if not folder.is_dir():
            continue
        for py in sorted(folder.glob("*.py")):
            digest = _sha256(py)
            if digest:
                hashes[str(py.relative_to(root)).replace("\\", "/")] = digest
    for py in sorted(root.glob("*.py")):
        digest = _sha256(py)
        if digest:
            hashes[py.name] = digest
    return hashes


def _namespace(args: Optional[Any]) -> dict:
    if args is None:
        return {}
    if isinstance(args, dict):
        return {key: repr(value) for key, value in args.items()}
    return {key: repr(value) for key, value in vars(args).items()}


def record_provenance(out_dir, args=None, extra=None) -> None:
    """Write the snapshot; report and return on any failure."""
    try:
        out_dir = Path(out_dir)
        torch_version = None
        try:
            import torch
            torch_version = torch.__version__
        except Exception:
            pass
        payload = {
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "project_root": str(PROJECT_ROOT),
            "git_commit": _git_commit(PROJECT_ROOT),
            "python_version": sys.version.replace("\n", " "),
            "torch_version": torch_version,
            "args": _namespace(args),
            "code_sha256": _code_hashes(PROJECT_ROOT),
        }
        if extra:
            payload["extra"] = {key: repr(value) for key, value in extra.items()}
        target = out_dir / "provenance.json"
        target.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  provenance -> {target}")
    except Exception as exc:
        print(f"  [warn] provenance recording failed: {exc}")
