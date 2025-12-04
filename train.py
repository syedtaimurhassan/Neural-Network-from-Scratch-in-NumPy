import argparse
import pathlib
import sys

import numpy as np

# Allow imports from src/.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from nn.engine import Trainer, accuracy, build_mlp  # noqa: E402
from nn.losses import CrossEntropyLoss  # noqa: E402
from nn.optimizers import get_optimizer  # noqa: E402
from data_loader import load_cifar10, load_fashion_mnist, make_dataloader  # noqa: E402
from preprocessing import flatten_images, normalize_images, standardize_per_channel, train_val_split  # noqa: E402
from utils.logging import CSVLogger  # noqa: E402
from utils.checkpoint import save_model, load_model  # noqa: E402
from utils.wandb_utils import init_wandb, log_metrics, log_histograms, finish  # noqa: E402


def _decay_lr(optimizer, gamma: float):
    if hasattr(optimizer, "lr"):
        optimizer.lr *= gamma


def run_xor(config):
    """Simple XOR sanity training."""
    np.random.seed(config["seed"])
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.int64)

    model = build_mlp(
        input_dim=2,
        hidden_layers=config["hidden_layers"],
        output_dim=2,
        activation=config["activation"],
        init=config["init"],
        l2_coeff=config["l2"],
        dropout=config["dropout"],
        batch_norm=config["batch_norm"],
    )
    loss_fn = CrossEntropyLoss()
    optimizer = get_optimizer(config["optimizer"], lr=config["lr"], weight_decay=config["weight_decay"])
    trainer = Trainer(model, loss_fn, optimizer, metrics_fn=accuracy, grad_clip_norm=config["clip_grad_norm"])

    for epoch in range(1, config["epochs"] + 1):
        dataloader = make_dataloader(X, y, batch_size=config["batch_size"], shuffle=True)
        train_loss, train_acc, grad_norm = trainer.train_epoch(dataloader)
        print(f"[XOR] epoch {epoch:02d} | loss={train_loss:.4f} | acc={train_acc:.3f} | grad_norm={grad_norm:.3f}")


def prepare_fashion(data_dir, val_ratio, seed):
    """Load and preprocess Fashion-MNIST."""
    X_train, y_train = load_fashion_mnist(data_dir, split="train")
    X_test, y_test = load_fashion_mnist(data_dir, split="test")
    X_train, y_train, X_val, y_val = train_val_split(X_train, y_train, val_ratio=val_ratio, seed=seed)
    X_train = flatten_images(normalize_images(X_train))
    X_val = flatten_images(normalize_images(X_val))
    X_test = flatten_images(normalize_images(X_test))
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), 28 * 28, 10


def prepare_cifar10(data_dir, val_ratio, seed):
    """Load and preprocess CIFAR-10."""
    X_train, y_train = load_cifar10(data_dir, split="train")
    X_test, y_test = load_cifar10(data_dir, split="test")
    X_train, y_train, X_val, y_val = train_val_split(X_train, y_train, val_ratio=val_ratio, seed=seed)
    X_train = normalize_images(X_train)
    X_val = normalize_images(X_val)
    X_test = normalize_images(X_test)
    # Standardize per channel using train stats
    X_train, mean, std = standardize_per_channel(X_train)
    X_val, _, _ = standardize_per_channel(X_val, mean, std)
    X_test, _, _ = standardize_per_channel(X_test, mean, std)
    X_train = flatten_images(X_train)
    X_val = flatten_images(X_val)
    X_test = flatten_images(X_test)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), 32 * 32 * 3, 10


def run_classification(config):
    """Train/eval classification model on chosen dataset."""
    np.random.seed(config["seed"])
    if config["dataset"] == "fashion":
        (X_train, y_train), (X_val, y_val), (X_test, y_test), input_dim, num_classes = prepare_fashion(
            config["data_dir"], config["val_ratio"], config["seed"]
        )
    elif config["dataset"] == "cifar10":
        (X_train, y_train), (X_val, y_val), (X_test, y_test), input_dim, num_classes = prepare_cifar10(
            config["data_dir"], config["val_ratio"], config["seed"]
        )
    else:
        raise ValueError("Unsupported dataset")

    model = build_mlp(
        input_dim=input_dim,
        hidden_layers=config["hidden_layers"],
        output_dim=num_classes,
        activation=config["activation"],
        init=config["init"],
        l2_coeff=config["l2"],
        dropout=config["dropout"],
        batch_norm=config["batch_norm"],
    )
    loss_fn = CrossEntropyLoss()
    optimizer = get_optimizer(config["optimizer"], lr=config["lr"], weight_decay=config["weight_decay"])
    trainer = Trainer(model, loss_fn, optimizer, metrics_fn=accuracy, grad_clip_norm=config["clip_grad_norm"])

    if config["checkpoint"] is not None and config["eval_only"]:
        load_model(model, config["checkpoint"])
        test_loader = make_dataloader(X_test, y_test, batch_size=config["batch_size"], shuffle=False)
        test_loss, test_acc = trainer.evaluate(test_loader)
        print(f"[{config['dataset']}] eval-only checkpoint={config['checkpoint']} | test_loss={test_loss:.4f} acc={test_acc:.3f}")
        return

    logger = None
    if config["log_dir"] is not None:
        log_path = pathlib.Path(config["log_dir"]) / f"{config['dataset']}_metrics.csv"
        logger = CSVLogger(
            log_path,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "grad_norm"],
        )
    wandb_run = init_wandb(config, project=config["wandb_project"], run_name=config["run_name"], mode=config["wandb_mode"])

    best_val = float("inf")
    patience_counter = 0
    for epoch in range(1, config["epochs"] + 1):
        train_loader = make_dataloader(X_train, y_train, batch_size=config["batch_size"], shuffle=True)
        val_loader = make_dataloader(X_val, y_val, batch_size=config["batch_size"], shuffle=False)
        train_loss, train_acc, grad_norm = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.evaluate(val_loader)
        lr_val = getattr(optimizer, "lr", None)
        print(
            f"[{config['dataset']}] epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.3f} | grad_norm={grad_norm:.3f} | lr={lr_val}"
        )
        metrics_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": lr_val,
            "grad_norm": grad_norm,
        }
        if logger:
            logger.log(metrics_row)
        log_metrics(wandb_run, metrics_row, step=epoch)
        log_histograms(wandb_run, model.parameters(), step=epoch)
        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            patience_counter = 0
            if config["checkpoint_dir"] is not None:
                ckpt_path = pathlib.Path(config["checkpoint_dir"]) / f"{config['dataset']}_best.npz"
                save_model(model, ckpt_path)
        else:
            patience_counter += 1
        if config["scheduler"] == "step" and epoch % config["step_size"] == 0:
            _decay_lr(optimizer, config["gamma"])
        if config["scheduler"] == "plateau" and patience_counter > 0 and patience_counter % config["patience"] == 0:
            _decay_lr(optimizer, config["gamma"])
        if config["patience"] is not None and patience_counter >= config["patience"]:
            print(f"[early stopping] no val improvement for {config['patience']} epochs, stopping.")
            break
    if logger:
        logger.close()
    finish(wandb_run)


def str2bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.lower() in ("yes", "true", "t", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "0"):
            return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    """Parse CLI arguments allowing both --arg=value and --arg value."""
    parser = argparse.ArgumentParser(description="NumPy MLP trainer.")

    # Core dataset/training options
    parser.add_argument("--dataset", type=str, default="xor", choices=["xor", "fashion", "cifar10"])
    parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("data"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "momentum", "adam", "nadam"])
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh", "gelu", "selu"])
    parser.add_argument("--init", type=str, default="he", choices=["xavier", "he"])
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--hidden-layers", type=str, default="256,128")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-norm", type=str2bool, default=False)
    parser.add_argument("--clip-grad-norm", type=float, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "step", "plateau"])
    parser.add_argument("--step-size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.5)

    # Logging / checkpointing
    parser.add_argument("--log-dir", type=pathlib.Path, default=pathlib.Path("logs"))
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, default=pathlib.Path("checkpoints"))
    parser.add_argument("--eval-only", type=str2bool, default=False)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="disabled", choices=["disabled", "online", "offline"])
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--name", type=str, default=None, help="Alias for run-name (for sweep compatibility)")

    return parser.parse_args()


def main():
    args = parse_args()
    hidden_layers = [int(x) for x in args.hidden_layers.split(",") if x]
    config = {
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "activation": args.activation,
        "init": args.init,
        "weight_decay": args.weight_decay,
        "l2": args.l2,
        "hidden_layers": hidden_layers,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "dropout": args.dropout,
        "batch_norm": args.batch_norm,
        "clip_grad_norm": args.clip_grad_norm,
        "patience": args.patience,
        "scheduler": args.scheduler,
        "step_size": args.step_size,
        "gamma": args.gamma,
        "log_dir": args.log_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "eval_only": args.eval_only,
        "checkpoint": args.checkpoint,
        "wandb_project": args.wandb_project,
        "wandb_mode": args.wandb_mode,
        "run_name": args.run_name or args.name,
    }
    if config["dataset"] == "xor":
        run_xor(config)
    else:
        run_classification(config)


if __name__ == "__main__":
    main()
