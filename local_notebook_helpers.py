"""Data reuse and saved-run helpers for the readable local experiment notebook."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from prepare_reviewed_training import prepare_reviewed_manifest
from longitudinal_ext.experiments import run_experiment


def load_reviewed_data(base_csv, reviews_csv, split_csv, reports_root, output_csv):
    """Reuse a reviewed manifest only when its source files match the saved audit."""
    output_csv = Path(output_csv)
    audit_path = output_csv.with_name("manual_review_merge.json")
    if output_csv.exists() and audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        current = {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                   for p in (base_csv, reviews_csv, split_csv)}
        if audit["source_sha256"] != current:
            raise ValueError("Source data changed. Set RESUME_RUN=False to create a new experiment run.")
    else:
        _, audit = prepare_reviewed_manifest(base_csv, reviews_csv, split_csv, reports_root, output_csv,
                                             verbose=False)
    return pd.read_csv(output_csv), audit


def completed_run(run_root, name, seed=42, expected=None):
    """Return the latest completed run whose recorded settings match the request."""
    for folder in sorted(Path(run_root).glob(f"ext_{name}_s{seed}_*"), reverse=True):
        if not all((folder / filename).is_file() for filename in ("metrics.csv", "settings.json")):
            continue
        saved = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
        if all(saved.get(k) == v for k, v in (expected or {}).items()):
            return folder
    return None


def run_or_load(name, *, reuse=True, **kwargs):
    """Load matching saved metrics or train a new experiment with the given settings."""
    seed = kwargs.get("seed", 42)
    use_images = kwargs.get("use_images", True)
    frozen = kwargs.get("frozen_only", False)
    expected = {"encoder": kwargs.get("encoder", "densenet121_imagenet") if use_images else None,
                "seed": seed}
    if not frozen:
        expected.update(
            experiment=name, epochs=kwargs.get("epochs", 20), batch_size=kwargs.get("batch_size", 8),
            use_images=use_images, use_tab=kwargs.get("tab_csv") is not None,
            tab_control=kwargs.get("tab_control"),
            diagnosis_checkpoint=str(kwargs["diagnosis_checkpoint"]) if kwargs.get("diagnosis_checkpoint") else None,
            backbone_lr=kwargs.get("backbone_lr", 1e-5), head_lr=kwargs.get("head_lr", 1e-4),
            final_lr_ratio=kwargs.get("final_lr_ratio", 0.1),
        )
    saved = completed_run(kwargs["out_root"], name, seed, expected) if reuse else None
    if saved is not None:
        print(f"Loaded {name}, seed {seed}: {saved.name}")
        return {"out_dir": saved}
    print(f"Training {name}, seed {seed}", flush=True)
    return run_experiment(name, **kwargs)
