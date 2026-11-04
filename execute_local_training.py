"""Execute the local notebook, saving outputs after every cell and logging progress."""
import argparse
import json
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


class LoggingClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        if msg["msg_type"] == "stream":
            print(msg["content"].get("text", ""), end="", flush=True)
        elif msg["msg_type"] == "error":
            print("\n".join(msg["content"].get("traceback", [])), flush=True)
        return super().process_message(msg, cell, cell_index)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default="main_notebook_clean_ext_multigpu_local.ipynb")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--extensions-only", action="store_true",
                        help="Resume A–F using this run's prepared data and diagnosis checkpoint")
    args = parser.parse_args()
    notebook = Path(args.notebook).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["BN5212_RUN_DIR"] = str(run_dir)
    os.environ["BN5212_PROJECT_DIR"] = str(notebook.parent)
    os.environ["BN5212_EXTENSIONS_ONLY"] = "1" if args.extensions_only else "0"
    os.environ["PYTHONUTF8"] = "1"
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    # Keep model downloads in the project; the existing ImageNet weight is copied once.
    cache = notebook.parent / "extensions_outputs" / "model_cache"
    os.environ["HF_HOME"] = str(cache / "huggingface")
    os.environ["TORCH_HOME"] = str(cache / "torch")
    os.environ["XRV_CACHE_DIR"] = str(cache / "xrv")
    if (cache / "encoder_preflight.json").exists():
        os.environ["HF_HUB_OFFLINE"] = "1"
    weights = Path.home() / ".cache/torch/hub/checkpoints/densenet121-a639ec97.pth"
    local_weights = cache / "torch/hub/checkpoints" / weights.name
    if weights.is_file() and not local_weights.exists():
        local_weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(weights, local_weights)
    backup_name = "notebook_before_extensions.ipynb" if args.extensions_only else "notebook_before_execution.ipynb"
    shutil.copy2(notebook, run_dir / backup_name)
    nb = nbformat.read(notebook, as_version=4)
    tagged_extensions = {i for i, cell in enumerate(nb.cells)
                         if "extensions" in cell.get("metadata", {}).get("tags", [])}
    if args.extensions_only:
        if not tagged_extensions:
            raise ValueError("No cells are tagged 'extensions' in the selected notebook.")
        selected = tagged_extensions
    else:
        selected = set(range(len(nb.cells)))
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and index in selected:
            cell.outputs = []
            cell.execution_count = None
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = LoggingClient(nb, km=km, timeout=None, allow_errors=False,
                           resources={"metadata": {"path": str(notebook.parent)}})
    status = {"pid": os.getpid(), "started": datetime.now().isoformat(),
              "notebook": str(notebook), "run_dir": str(run_dir), "state": "running",
              "scope": "extensions" if args.extensions_only else "all"}

    def save():
        nbformat.write(nb, run_dir / "executed.ipynb")
        temporary = notebook.with_suffix(".executing.tmp")
        nbformat.write(nb, temporary)
        try:
            temporary.replace(notebook)
        except PermissionError:
            # Windows editors can briefly prevent renaming an open notebook.
            nbformat.write(nb, notebook)
            temporary.unlink(missing_ok=True)
        (run_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    save()
    try:
        with client.setup_kernel(cwd=str(notebook.parent)):
            count = 0
            for index, cell in enumerate(nb.cells):
                if index not in selected or cell.cell_type != "code" or not cell.source.strip():
                    continue
                count += 1
                status.update(cell_index=index, execution_count=count)
                save()
                print(f"\n=== Cell {index} / execution {count}: {cell.source.splitlines()[0]} ===", flush=True)
                client.execute_cell(cell, index, execution_count=count)
                save()
            status.update(state="completed", finished=datetime.now().isoformat())
            save()
    except BaseException as error:
        status.update(state="failed", error=str(error), finished=datetime.now().isoformat())
        save()
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
