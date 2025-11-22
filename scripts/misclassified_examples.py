import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from nn.engine import build_mlp  # noqa: E402
from nn.losses import CrossEntropyLoss  # noqa: E402
from nn.optimizers import get_optimizer  # noqa: E402
from data_loader import load_fashion_mnist, load_cifar10, make_dataloader  # noqa: E402
from preprocessing import flatten_images, normalize_images, standardize_per_channel, train_val_split  # noqa: E402
from utils.checkpoint import load_model  # noqa: E402


def prepare_data(dataset, data_dir, val_ratio, seed):
    if dataset == "fashion":
        X_train, y_train = load_fashion_mnist(data_dir, split="train")
        X_test, y_test = load_fashion_mnist(data_dir, split="test")
        X_train, y_train, X_val, y_val = train_val_split(X_train, y_train, val_ratio=val_ratio, seed=seed)
        X_split = flatten_images(normalize_images(X_test))
        y_split = y_test
        input_dim = 28 * 28
        num_classes = 10
    elif dataset == "cifar10":
        X_train, y_train = load_cifar10(data_dir, split="train")
        X_test, y_test = load_cifar10(data_dir, split="test")
        X_train, y_train, X_val, y_val = train_val_split(X_train, y_train, val_ratio=val_ratio, seed=seed)
        X_train = normalize_images(X_train)
        X_test = normalize_images(X_test)
        X_train, mean, std = standardize_per_channel(X_train)
        X_test, _, _ = standardize_per_channel(X_test, mean, std)
        X_split = flatten_images(X_test)
        y_split = y_test
        input_dim = 32 * 32 * 3
        num_classes = 10
    else:
        raise ValueError("Unsupported dataset")
    return X_split, y_split, input_dim, num_classes


def collect_misclassified(model, X: np.ndarray, y: np.ndarray, batch_size: int, limit: int):
    wrong = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        xb = X[start:end]
        yb = y[start:end]
        logits = model.forward(xb)
        preds = np.argmax(logits, axis=1)
        mism = preds != yb
        for xi, yi, pi in zip(xb[mism], yb[mism], preds[mism]):
            wrong.append((xi, yi, pi))
            if len(wrong) >= limit:
                return wrong
    return wrong


def plot_misclassified(dataset, samples, out_path: pathlib.Path):
    n = len(samples)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.array(axes).reshape(rows, cols)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (x, y_true, y_pred) in zip(axes.flat, samples):
        if dataset == "fashion":
            img = x.reshape(28, 28)
            ax.imshow(img, cmap="gray")
        else:
            img = x.reshape(3, 32, 32).transpose(1, 2, 0)
            img = (img - img.min()) / max(img.max() - img.min(), 1e-8)
            ax.imshow(img)
        ax.set_title(f"T:{y_true} P:{y_pred}", fontsize=8)
        ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Save misclassified examples grid from checkpoint.")
    parser.add_argument("--dataset", choices=["fashion", "cifar10"], required=True)
    parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("data"))
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--hidden-layers", type=str, default="256,128")
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--init", type=str, default="he")
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-norm", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("outputs/plots/misclassified.png"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hidden_layers = [int(x) for x in args.hidden_layers.split(",") if x]
    X_split, y_split, input_dim, num_classes = prepare_data(args.dataset, args.data_dir, val_ratio=0.1, seed=args.seed)

    model = build_mlp(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        output_dim=num_classes,
        activation=args.activation,
        init=args.init,
        l2_coeff=args.l2,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
    )
    load_model(model, args.checkpoint)
    samples = collect_misclassified(model, X_split, y_split, args.batch_size, args.limit)
    plot_misclassified(args.dataset, samples, args.out)
    print(f"[{args.dataset}] saved {len(samples)} misclassified samples to {args.out}")


if __name__ == "__main__":
    main()
