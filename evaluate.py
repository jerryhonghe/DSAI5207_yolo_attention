"""
Unified evaluation entry point.

Usage:
    python evaluate.py --mode compare
    python evaluate.py --mode gradcam --images img1.jpg img2.jpg
    python evaluate.py --mode relation --images img1.jpg
    python evaluate.py --mode all --images img1.jpg img2.jpg
"""
import argparse
import os


DEFAULT_MODELS = {
    'E1 Baseline': 'runs/E1_baseline/weights/best.pt',
    'E2 Ours': 'runs/E2_relation_distill/weights/best.pt',
    'E3 d2': 'runs/E3_feature_distill_d2/weights/best.pt',
}

DEFAULT_TEACHER = 'weights/dinov2_vitl14_reg4_pretrain.pth'
DEFAULT_DATA = 'configs/data.yaml'
DEFAULT_OUTPUT = 'results'


def run_compare(model_paths, data_yaml, output_dir):
    from evaluation.compare import compare_experiments
    compare_experiments(model_paths, data_yaml, os.path.join(output_dir, 'compare'))


def run_gradcam(image_paths, model_paths, output_dir):
    from evaluation.gradcam import generate_gradcam_comparison
    generate_gradcam_comparison(image_paths, model_paths, os.path.join(output_dir, 'gradcam'))


def run_relation_vis(image_paths, student_model_path, teacher_weights, output_dir):
    from evaluation.relation_vis import visualize_relation_matrices
    for img_path in image_paths:
        visualize_relation_matrices(
            img_path, student_model_path, teacher_weights,
            os.path.join(output_dir, 'relation_vis')
        )


def main():
    parser = argparse.ArgumentParser(description='Evaluation Tools')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['compare', 'gradcam', 'relation', 'all'])
    parser.add_argument('--images', nargs='+', help='Image paths for visualization')
    parser.add_argument('--data', type=str, default=DEFAULT_DATA)
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT)
    parser.add_argument('--teacher', type=str, default=DEFAULT_TEACHER)
    args = parser.parse_args()

    # Check which models exist
    model_paths = {}
    for name, path in DEFAULT_MODELS.items():
        if os.path.exists(path):
            model_paths[name] = path
        else:
            print(f"Warning: {name} model not found at {path}, skipping")

    if not model_paths:
        print("Error: No model weights found. Train models first.")
        return

    if args.mode in ('compare', 'all'):
        print("\n=== mAP Comparison ===")
        run_compare(model_paths, args.data, args.output)

    if args.mode in ('gradcam', 'all'):
        if not args.images:
            print("Error: --images required for GradCAM mode")
            return
        print("\n=== GradCAM Heatmaps ===")
        run_gradcam(args.images, model_paths, args.output)

    if args.mode in ('relation', 'all'):
        if not args.images:
            print("Error: --images required for relation visualization mode")
            return
        ours_path = model_paths.get('E2 Ours')
        if not ours_path:
            print("Error: E2 Ours model required for relation visualization")
            return
        print("\n=== Relation Matrix Visualization ===")
        run_relation_vis(args.images, ours_path, args.teacher, args.output)

    print(f"\nAll results saved to: {args.output}/")


if __name__ == '__main__':
    main()
