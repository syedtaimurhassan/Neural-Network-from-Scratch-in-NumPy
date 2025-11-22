import argparse
import pathlib
import tarfile
import urllib.request
import shutil
import time


FASHION_PRIMARY = "https://fashion-mnist.s3-website.eu-central-1.amazonaws.com"
FASHION_FALLBACK = "https://storage.googleapis.com/tensorflow/tf-keras-datasets"
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


FASHION_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_file(url: str, dest: pathlib.Path, retries: int = 2, sleep: float = 1.0):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest.name} already exists")
        return True

    attempt = 0
    while attempt <= retries:
        try:
            print(f"[download] {url}")
            with urllib.request.urlopen(url, timeout=15) as resp, open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)
            return True
        except Exception as exc:
            attempt += 1
            print(f"[warn] failed ({attempt}/{retries + 1}): {exc}")
            time.sleep(sleep)
    return False


def download_fashion_mnist(data_dir: pathlib.Path):
    ok = True
    for fname in FASHION_FILES.values():
        dest = data_dir / "fashion-mnist" / fname
        # Try primary first, then fallback.
        primary = f"{FASHION_PRIMARY}/{fname}"
        fallback = f"{FASHION_FALLBACK}/{fname}"
        if not download_file(primary, dest):
            print(f"[info] retrying from fallback mirror: {fallback}")
            ok = download_file(fallback, dest) and ok
    return ok


def download_cifar10(data_dir: pathlib.Path):
    dest = data_dir / "cifar-10" / "cifar-10-python.tar.gz"
    if not download_file(CIFAR10_URL, dest):
        return False
    extract_dir = data_dir / "cifar-10"
    if dest.exists():
        with tarfile.open(dest, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        print(f"[extracted] {dest} -> {extract_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download Fashion-MNIST and CIFAR-10 datasets.")
    parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("data"))
    parser.add_argument("--dataset", choices=["fashion", "cifar10", "all"], default="all")
    args = parser.parse_args()

    success = True
    if args.dataset in ("fashion", "all"):
        success = download_fashion_mnist(args.data_dir) and success
    if args.dataset in ("cifar10", "all"):
        success = download_cifar10(args.data_dir) and success

    if not success:
        raise SystemExit("One or more downloads failed.")


if __name__ == "__main__":
    main()
