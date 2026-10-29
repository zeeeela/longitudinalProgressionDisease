import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import CLASS_NAMES, DISEASES


def collect(runs):
    """runs: {experiment_name: [run_dir, ...]} (several dirs = several seeds). -> long DataFrame."""
    parts = []
    for name, dirs in runs.items():
        for i, d in enumerate(dirs if isinstance(dirs, (list, tuple)) else [dirs]):
            t = pd.read_csv(Path(d) / "metrics.csv")
            t.insert(0, "run", i)
            t.insert(0, "experiment", name)
            parts.append(t)
    return pd.concat(parts, ignore_index=True)


def summary(all_metrics, split="test", metric="macro_f1"):
    """Mean over seeds (and sd) of `metric` per experiment x disease, plus the mean over diseases."""
    d = all_metrics[all_metrics["split"] == split]
    per_run = d.groupby(["experiment", "run"])[metric].mean().rename("MEAN_over_diseases").reset_index()
    wide = d.pivot_table(index="experiment", columns="disease", values=metric, aggfunc="mean")[
        [x for x in DISEASES if x in d["disease"].unique()]]
    agg = per_run.groupby("experiment")["MEAN_over_diseases"].agg(["mean", "std"]).rename(
        columns={"mean": "mean_over_diseases", "std": "sd_over_seeds"})
    return wide.join(agg).round(3)


def plot_confusions(run_dir, split="test"):
    run_dir = Path(run_dir)
    conf = json.loads((run_dir / "confusion_matrices.json").read_text())[split]["confusion"]
    fig, axes = plt.subplots(1, len(DISEASES), figsize=(20, 4), layout="constrained")
    for ax, name, m in zip(axes, DISEASES, conf):
        m = np.asarray(m)
        sup = m.sum(1, keepdims=True)
        fr = np.divide(m, sup, out=np.zeros(m.shape, float), where=sup != 0)
        ax.imshow(fr, vmin=0, vmax=1, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{m[i, j]}\n{fr[i, j]:.0%}" if sup[i, 0] else "0\nN/A", ha="center", va="center",
                        color="white" if fr[i, j] > .5 else "black", fontsize=9)
        ax.set_xticks(range(3), CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticks(range(3), CLASS_NAMES)
        ax.set(title=f"{name}\nn = {int(m.sum())}", xlabel="Predicted", ylabel="Actual")
    fig.suptitle(f"{run_dir.name} | {split}")
    plt.show()
