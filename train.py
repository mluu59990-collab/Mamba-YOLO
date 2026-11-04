"""Run from this repository, including after cloning it on Kaggle."""
import argparse
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['baseline', 'mamba'], default='mamba')
    p.add_argument('--task', choices=['train', 'val', 'predict'], default='train')
    p.add_argument('--data', help='Absolute path to YOLO detection dataset YAML')
    p.add_argument('--weights', help='Checkpoint from this repository for val/predict')
    p.add_argument('--resume', help='Resume this repository checkpoint, including optimizer/scheduler')
    p.add_argument('--source', help='Image/video source for predict')
    p.add_argument('--profile', choices=['kaggle', 'author'], default='author')
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch', type=int)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='0', help='One GPU (0), or cpu for small debugging only')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--project', default=str(ROOT / 'runs'))
    p.add_argument('--name')
    a = p.parse_args()
    if ',' in a.device:
        p.error('Use one GPU per run; multi-GPU has not been validated for these custom blocks.')
    if a.task in ('train', 'val') and not a.resume and not a.data:
        p.error('--data is required')
    if a.data and not Path(a.data).is_file():
        p.error(f'Dataset YAML not found: {a.data}')
    if a.task != 'train' and not a.weights:
        p.error('--weights is required for val/predict; do not evaluate random initialization')
    if a.task == 'predict' and not a.source:
        p.error('--source is required for predict')
    if a.task == 'train' and a.weights:
        p.error('Use --resume for your hybrid checkpoint. Baseline weights have different layer indices.')
    import torch
    import ultralytics
    from ultralytics import YOLO
    if ROOT not in Path(ultralytics.__file__).resolve().parents:
        raise RuntimeError('Wrong ultralytics imported; install this repository with pip install -e .')
    if a.device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('Enable Kaggle GPU accelerator or use --device cpu for debugging.')
    if a.device != 'cpu' and (a.variant != 'baseline' or a.weights or a.resume):
        import selective_scan_cuda_core  # fail before expensive model construction
    if a.resume:
        if a.task != 'train' or not Path(a.resume).is_file():
            p.error('--resume requires train and an existing checkpoint')
        resume_options = dict(resume=True, device=a.device, workers=a.workers)
        if a.data:
            resume_options['data'] = str(Path(a.data).resolve())
        YOLO(a.resume).train(**resume_options)
        return
    model = YOLO(a.weights or ROOT / f'configs/yolov12s-{a.variant}.yaml', task='detect')
    common = dict(device=a.device, imgsz=a.imgsz, project=a.project, name=a.name or a.variant)
    if a.task == 'val':
        model.val(data=str(Path(a.data).resolve()), batch=a.batch or 4, workers=a.workers, **common)
    elif a.task == 'predict':
        model.predict(source=a.source, save=True, **common)
    else:
        # Supplied Mamba-YOLO training script/default.yaml: 300 epochs, SGD, linear LR decay.
        # Batch 4 is a Kaggle adaptation; original launcher defaults to batch 512.
        options = dict(data=str(Path(a.data).resolve()), epochs=a.epochs or (300 if a.profile == 'author' else 100),
                       batch=a.batch or 4, workers=a.workers,
                       optimizer='SGD', lr0=.01, lrf=.01, cos_lr=False, momentum=.937,
                       weight_decay=.0005, warmup_epochs=3., warmup_bias_lr=.1, warmup_momentum=.8,
                       box=7.5, cls=.5, dfl=1.5, mosaic=1., hsv_h=.015, hsv_s=.7, hsv_v=.4,
                       mixup=0., copy_paste=0., close_mosaic=10, patience=0, seed=a.seed,
                       deterministic=False, amp=a.amp, cache=False, pretrained=False, **common)
        def record(trainer):
            metadata = dict(arguments=vars(a), torch=torch.__version__, cuda=torch.version.cuda,
                            python=platform.python_version(), ultralytics=ultralytics.__version__,
                            source=str(Path(ultralytics.__file__).resolve()),
                            gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')
            (trainer.save_dir / 'environment.json').write_text(json.dumps(metadata, indent=2))
        model.add_callback('on_pretrain_routine_end', record)
        model.train(**options)


if __name__ == '__main__':
    main()
