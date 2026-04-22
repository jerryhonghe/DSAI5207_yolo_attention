"""
Unified training entry point.

Usage:
    python train.py --config configs/baseline.yaml
    python train.py --config configs/relation_distill.yaml
    python train.py --config configs/feature_distill_d2.yaml
    python train.py --config configs/relation_distill.yaml --resume
"""
import argparse
import yaml

from ultralytics import YOLO


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_baseline(cfg):
    """E1: Standard YOLOv8m training."""
    model = YOLO(cfg['model'])
    model.train(
        data=cfg['data'],
        epochs=cfg['epochs'],
        batch=cfg['batch'],
        imgsz=cfg['imgsz'],
        optimizer=cfg.get('optimizer', 'SGD'),
        lr0=cfg.get('lr0', 0.01),
        lrf=cfg.get('lrf', 0.01),
        momentum=cfg.get('momentum', 0.937),
        weight_decay=cfg.get('weight_decay', 0.0005),
        warmup_epochs=cfg.get('warmup_epochs', 3),
        device=cfg.get('device', 0),
        project=cfg.get('project', 'runs'),
        name=cfg.get('name', 'baseline'),
        val=True,
    )


def train_relation_distill(cfg, resume=False):
    """E2: Attention Relation Distillation."""
    from trainers.relation_trainer import RelationDistillTrainer

    overrides = {
        'model': cfg['model'],
        'data': cfg['data'],
        'epochs': cfg['epochs'],
        'batch': cfg['batch'],
        'imgsz': cfg['imgsz'],
        'optimizer': cfg.get('optimizer', 'SGD'),
        'lr0': cfg.get('lr0', 0.01),
        'lrf': cfg.get('lrf', 0.01),
        'momentum': cfg.get('momentum', 0.937),
        'weight_decay': cfg.get('weight_decay', 0.0005),
        'warmup_epochs': cfg.get('warmup_epochs', 3),
        'device': cfg.get('device', 0),
        'project': cfg.get('project', 'runs'),
        'name': cfg.get('name', 'relation_distill'),
        'val': True,
        # Custom distillation params
        'teacher_weights': cfg.get('teacher_weights'),
        'lambda_distill': cfg.get('lambda_distill', 1.0),
        'temperature': cfg.get('temperature', 0.07),
        'max_relation_tokens': cfg.get('max_relation_tokens', 1600),
        'teacher_layers': cfg.get('teacher_layers'),
        'level_weights': cfg.get('level_weights'),
        'val_interval': cfg.get('val_interval', 10),
    }

    if resume:
        overrides['resume'] = True

    trainer = RelationDistillTrainer(overrides=overrides)
    trainer.train()


def train_d2_distill(cfg, resume=False):
    """E3: Feature Distillation (d2 reproduction)."""
    from trainers.d2_trainer import D2DistillTrainer

    overrides = {
        'model': cfg['model'],
        'data': cfg['data'],
        'epochs': cfg['epochs'],
        'batch': cfg['batch'],
        'imgsz': cfg['imgsz'],
        'optimizer': cfg.get('optimizer', 'SGD'),
        'lr0': cfg.get('lr0', 0.01),
        'lrf': cfg.get('lrf', 0.01),
        'momentum': cfg.get('momentum', 0.937),
        'weight_decay': cfg.get('weight_decay', 0.0005),
        'warmup_epochs': cfg.get('warmup_epochs', 3),
        'device': cfg.get('device', 0),
        'project': cfg.get('project', 'runs'),
        'name': cfg.get('name', 'feature_distill_d2'),
        'val': True,
        # Custom distillation params
        'teacher_weights': cfg.get('teacher_weights'),
        'lambda_distill': cfg.get('lambda_distill', 1.0),
        'dinov2_dim': cfg.get('dinov2_dim', 1024),
        'val_interval': cfg.get('val_interval', 10),
    }

    if resume:
        overrides['resume'] = True

    trainer = D2DistillTrainer(overrides=overrides)
    trainer.train()


EXPERIMENT_MAP = {
    'baseline': train_baseline,
    'relation_distill': train_relation_distill,
    'feature_distill_d2': train_d2_distill,
}


def main():
    parser = argparse.ArgumentParser(description='YOLO Attention Distillation Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    cfg = load_config(args.config)
    experiment = cfg.get('experiment', 'baseline')

    if experiment not in EXPERIMENT_MAP:
        raise ValueError(f"Unknown experiment: {experiment}. Choose from {list(EXPERIMENT_MAP.keys())}")

    print(f"\n{'=' * 60}")
    print(f"  Experiment: {experiment}")
    print(f"  Config: {args.config}")
    print(f"  Resume: {args.resume}")
    print(f"{'=' * 60}\n")

    train_fn = EXPERIMENT_MAP[experiment]

    if experiment == 'baseline':
        train_fn(cfg)
    else:
        train_fn(cfg, resume=args.resume)


if __name__ == '__main__':
    main()
