"""Run one A-F progression experiment per process (local or PBS job array)."""
import argparse
import csv
import json
from pathlib import Path

EXPERIMENTS = ('A_base_img', 'A2_base_diag_img', 'B_base_img_tab', 'C_new_img',
               'D_new_img_tab', 'E_new_img_tab_shuffled', 'F_tab_nn')
TASKS = [(name, seed) for name in EXPERIMENTS for seed in (42, 43, 44)]


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    choice = p.add_mutually_exclusive_group()
    choice.add_argument('--experiment', choices=EXPERIMENTS, default='A_base_img')
    choice.add_argument('--task-index', type=int, choices=range(len(TASKS)))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--list-tasks', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Validate paths and config; no model loading/training')
    p.add_argument('--manifest', type=Path)
    p.add_argument('--tab-csv', type=Path)
    p.add_argument('--checkpoint', type=Path, default=root / 'training_outputs/backbone_20261003_150057/best_backbone.pt')
    p.add_argument('--encoder', default='hf:microsoft/rad-dino')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--out-root', type=Path, default=root / 'training_outputs')
    args = p.parse_args()
    if args.list_tasks:
        for i, (name, seed) in enumerate(TASKS):
            print(f'{i:2d}  {name:25s} seed={seed}')
        return
    name, seed = TASKS[args.task_index] if args.task_index is not None else (args.experiment, args.seed)
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        p.error('epochs/batch-size must be positive; workers must be nonnegative')
    manifest = args.manifest or root / 'extensions_outputs/all_disease_pairs_local.csv'
    if args.manifest is None and not manifest.exists():
        manifest = root / 'labelling/prognosis_labels_hybrid/all_disease_pairs.csv'
    tab = args.tab_csv or root / 'extensions_outputs/local_feature_audit/pair_tabular_features.csv'
    if args.tab_csv is None and not tab.exists():
        tab = root / 'extensions_outputs/pair_tabular_features.csv'
    use_tab = name in ('B_base_img_tab', 'D_new_img_tab', 'E_new_img_tab_shuffled', 'F_tab_nn')
    required = [manifest] + ([tab] if use_tab else []) + ([args.checkpoint] if name == 'A2_base_diag_img' else [])
    for path in required:
        if not path.is_file():
            p.error(f'Missing {path}; prepare data or specify the path explicitly')
    with manifest.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    patients = {}
    pairs = set()
    for row in rows:
        subject, split = row['subject_id'], row['split']
        if split not in ('train', 'val', 'test') or patients.setdefault(subject, split) != split:
            p.error('Invalid split or patient overlap in manifest')
        key = tuple(row[k] for k in ('subject_id', 'input_study_id', 'target_study_id'))
        if key in pairs:
            p.error('Duplicate study pair in manifest')
        pairs.add(key)
        for col in ('prior_dicom_path', 'current_dicom_path'):
            if not Path(row[col]).is_file():
                p.error(f'Image path unavailable: {row[col]}. Run notebook path remapping on this machine first.')
    if set(patients.values()) != {'train', 'val', 'test'}:
        p.error('Manifest must contain all three patient splits')
    config = dict(manifest_csv=manifest, out_root=args.out_root, seed=seed, epochs=args.epochs,
                  batch_size=args.batch_size, num_workers=args.workers, device='cuda',
                  backbone_lr=1e-5, head_lr=1e-4, final_lr_ratio=0.1,
                  encoder=args.encoder if name.startswith(('C_', 'D_', 'E_')) else 'densenet121_imagenet')
    if use_tab:
        config['tab_csv'] = tab
    if name == 'A2_base_diag_img':
        config['diagnosis_checkpoint'] = args.checkpoint
    if name == 'E_new_img_tab_shuffled':
        config['tab_control'] = 'shuffle'
    if name == 'F_tab_nn':
        config['use_images'] = False
    print(json.dumps(dict(experiment=name, **config), default=str, indent=2), flush=True)
    if args.dry_run:
        print(f'Validated {len(rows)} pairs; no training started.')
        return
    import torch
    from longitudinal_ext.experiments import run_experiment
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: use the GPU environment inside an allocated GPU job.')
    print('Visible GPUs:', torch.cuda.device_count(), '| using:', torch.cuda.get_device_name(0), flush=True)
    # Each process gets its own ImageStore memory cache. No concurrent disk-cache writes.
    result = run_experiment(name, **config)
    print('Completed:', result['out_dir'], flush=True)


if __name__ == '__main__':
    main()
