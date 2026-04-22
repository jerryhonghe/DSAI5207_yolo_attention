"""
mAP comparison across experiments.

Usage:
    from evaluation.compare import compare_experiments
    compare_experiments(
        model_paths={
            'E1 Baseline': 'runs/E1_baseline/weights/best.pt',
            'E2 Ours': 'runs/E2_relation_distill/weights/best.pt',
            'E3 d2': 'runs/E3_feature_distill_d2/weights/best.pt',
        },
        data_yaml='configs/data.yaml',
        output_dir='results/compare/'
    )
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


def compare_experiments(model_paths, data_yaml, output_dir, imgsz=640):
    """Run validation on multiple models and produce comparison.

    Args:
        model_paths: dict {experiment_name: weight_path}
        data_yaml: dataset config path
        output_dir: output directory

    Returns:
        dict of results per experiment
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for name, weight_path in model_paths.items():
        print(f"\nEvaluating: {name}")
        print(f"  Weights: {weight_path}")

        model = YOLO(weight_path)
        metrics = model.val(data=data_yaml, imgsz=imgsz, verbose=False)

        results[name] = {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
        }

        if hasattr(metrics.box, 'maps'):
            results[name]['per_class_ap50'] = [float(x) for x in metrics.box.maps]

        print(f"  mAP50: {results[name]['mAP50']:.4f}")
        print(f"  mAP50-95: {results[name]['mAP50-95']:.4f}")

    # Save results JSON
    results_path = os.path.join(output_dir, 'comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Generate comparison chart
    _plot_comparison(results, output_dir)
    _print_table(results)

    return results


def _plot_comparison(results, output_dir):
    """Generate mAP comparison bar chart."""
    names = list(results.keys())
    metrics = ['mAP50', 'mAP50-95', 'precision', 'recall']
    colors = ['#2196F3', '#4CAF50', '#FF9800'][:len(names)]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))

    for i, metric in enumerate(metrics):
        values = [results[name].get(metric, 0) for name in names]
        bars = axes[i].bar(range(len(names)), values, color=colors)

        axes[i].set_title(metric, fontsize=14, fontweight='bold')
        axes[i].set_xticks(range(len(names)))
        axes[i].set_xticklabels(names, rotation=30, ha='right', fontsize=10)
        axes[i].set_ylim(0, 1)

        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle('Experiment Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'comparison_chart.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved to: {save_path}")


def _print_table(results):
    """Print a formatted comparison table."""
    print(f"\n{'=' * 70}")
    print(f"{'Experiment':<20} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}")
    print(f"{'-' * 70}")
    for name, r in results.items():
        print(f"{name:<20} {r['mAP50']:>8.4f} {r['mAP50-95']:>10.4f} "
              f"{r['precision']:>10.4f} {r['recall']:>8.4f}")
    print(f"{'=' * 70}")
