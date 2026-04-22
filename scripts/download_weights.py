"""
Download all required model weights on a machine with internet access.

Usage:
    python scripts/download_weights.py --output weights/
"""
import argparse
import os
import urllib.request


WEIGHTS = {
    'yolov8m.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt',
    'dinov2_vitl14_reg4_pretrain.pth': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth',
}


def download(url, output_path):
    if os.path.exists(output_path):
        print(f"  Already exists: {output_path}")
        return
    print(f"  Downloading: {url}")
    print(f"  To: {output_path}")
    urllib.request.urlretrieve(url, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Done! ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description='Download model weights')
    parser.add_argument('--output', type=str, default='weights/')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Downloading model weights...\n")
    for filename, url in WEIGHTS.items():
        output_path = os.path.join(args.output, filename)
        download(url, output_path)
        print()

    print("All weights downloaded.")
    print(f"Upload the {args.output} directory to your server.")


if __name__ == '__main__':
    main()
