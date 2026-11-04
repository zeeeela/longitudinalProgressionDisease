"""Download and check all encoders before a long local notebook execution."""
import gc
import json
import os
import shutil
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
cache = root / "extensions_outputs/model_cache"
os.environ["HF_HOME"] = str(cache / "huggingface")
os.environ["TORCH_HOME"] = str(cache / "torch")
os.environ["XRV_CACHE_DIR"] = str(cache / "xrv")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
weights = Path.home() / ".cache/torch/hub/checkpoints/densenet121-a639ec97.pth"
local = cache / "torch/hub/checkpoints" / weights.name
if weights.exists() and not local.exists():
    local.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, local)

from longitudinal_ext.encoders import build_encoder

if __name__ == "__main__":
    results = []
    for name in ("densenet121_imagenet", "xrv:densenet121-res224-chex",
                 "xrv:densenet121-res224-nih", "hf:microsoft/rad-dino",
                 "biomedclip", "timm:convnext_small.fb_in22k_ft_in1k"):
        print("Preparing", name, flush=True)
        encoder = build_encoder(name)
        results.append({"encoder": name, "feature_dimension": encoder.out_dim,
                        "image_size": encoder.image_size})
        print(results[-1], flush=True)
        del encoder
        gc.collect()
    (cache / "encoder_preflight.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("All six encoders are ready.", flush=True)
