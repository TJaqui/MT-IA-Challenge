from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from .mt_model import (
        ATTEMPTS,
        attempt_by_name,
        available_attempts,
        build_model,
        compliance_report,
        load_database_subset,
        mse_complex,
        mt_forward_numpy,
        mt_forward_torch,
        observation_features,
        physics_loss,
        regularization_loss,
        save_json,
        set_reproducible_seed,
        standardize,
        target_log_resistivity,
    )
except ImportError:
    from mt_model import (
        ATTEMPTS,
        attempt_by_name,
        available_attempts,
        build_model,
        compliance_report,
        load_database_subset,
        mse_complex,
        mt_forward_numpy,
        mt_forward_torch,
        observation_features,
        physics_loss,
        regularization_loss,
        save_json,
        set_reproducible_seed,
        standardize,
        target_log_resistivity,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train physics-guided MT inversion neural network.")
    parser.add_argument("--database-dir", type=Path, default=Path("database"))
    parser.add_argument("--attempt", choices=[a.name for a in ATTEMPTS], required=True)
    parser.add_argument("--train-count", type=int, default=20000)
    parser.add_argument("--val-count", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--list-attempts", action="store_true")
    return parser.parse_args()


def make_loader(
    rho: np.ndarray,
    zxy: np.ndarray,
    frequencies: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    features = standardize(observation_features(zxy, frequencies), feature_mean, feature_std)
    target = target_log_resistivity(rho)
    dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(target.astype(np.float32)),
        torch.from_numpy(zxy.astype(np.complex64)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    thicknesses: torch.Tensor,
    frequencies: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    supervised_losses: list[float] = []
    physics_losses: list[float] = []
    pred_rhos: list[np.ndarray] = []
    target_rhos: list[np.ndarray] = []
    pred_z: list[np.ndarray] = []
    obs_z: list[np.ndarray] = []
    with torch.no_grad():
        for x, log_rho_true, zxy_obs in loader:
            x = x.to(device)
            log_rho_true = log_rho_true.to(device)
            zxy_obs = zxy_obs.to(device)
            log_rho_pred = model(x)
            rho_pred = torch.pow(10.0, log_rho_pred)
            zxy_pred = mt_forward_torch(rho_pred, thicknesses, frequencies)
            supervised_losses.append(torch.mean((log_rho_pred - log_rho_true) ** 2).item())
            physics_losses.append(physics_loss(zxy_pred, zxy_obs).item())
            pred_rhos.append(rho_pred.cpu().numpy())
            target_rhos.append(torch.pow(10.0, log_rho_true).cpu().numpy())
            pred_z.append(zxy_pred.cpu().numpy())
            obs_z.append(zxy_obs.cpu().numpy())

    pred_rho_np = np.concatenate(pred_rhos, axis=0)
    target_rho_np = np.concatenate(target_rhos, axis=0)
    pred_z_np = np.concatenate(pred_z, axis=0)
    obs_z_np = np.concatenate(obs_z, axis=0)
    return {
        "mse_model_rho": float(np.mean((pred_rho_np - target_rho_np) ** 2)),
        "mse_model_log10_rho": float(np.mean((np.log10(pred_rho_np) - np.log10(target_rho_np)) ** 2)),
        "mse_impedance_complex": mse_complex(pred_z_np, obs_z_np),
        "physics_loss": float(np.mean(physics_losses)),
        "supervised_loss": float(np.mean(supervised_losses)),
    }


def append_summary(output_dir: Path, row: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "attempts_summary.csv"
    exists = path.exists()
    fieldnames = [
        "attempt",
        "architecture",
        "regularizer",
        "reg_weight",
        "train_count",
        "val_count",
        "epochs",
        "best_epoch",
        "best_val_score",
        "mse_model_rho",
        "mse_model_log10_rho",
        "mse_impedance_complex",
        "physics_loss",
        "compliance_ok",
        "seconds",
    ]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def train_one(args: argparse.Namespace) -> dict[str, object]:
    attempt = attempt_by_name(args.attempt)
    set_reproducible_seed(args.seed)
    device = torch.device(args.device)

    data = load_database_subset(
        args.database_dir,
        train_count=args.train_count,
        val_count=args.val_count,
        seed=args.seed,
    )

    train_features = observation_features(data["train_zxy"], data["frequencies"])
    feature_mean = train_features.mean(axis=0).astype(np.float32)
    feature_std = train_features.std(axis=0).astype(np.float32)

    train_loader = make_loader(
        data["train_rho"],
        data["train_zxy"],
        data["frequencies"],
        feature_mean,
        feature_std,
        args.batch_size,
        shuffle=True,
    )
    val_loader = make_loader(
        data["val_rho"],
        data["val_zxy"],
        data["frequencies"],
        feature_mean,
        feature_std,
        args.batch_size,
        shuffle=False,
    )

    input_dim = train_features.shape[1]
    output_dim = data["train_rho"].shape[1]
    model = build_model(input_dim=input_dim, output_dim=output_dim, architecture=attempt.architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    thicknesses = torch.from_numpy(data["thicknesses"]).to(device)
    frequencies = torch.from_numpy(data["frequencies"]).to(device)

    run_dir = args.output_dir / attempt.name
    run_dir.mkdir(parents=True, exist_ok=True)

    compliance = compliance_report(
        attempt=attempt,
        model_layers=output_dim,
        thickness_layers=len(data["thicknesses"]),
    )
    save_json(run_dir / "compliance.json", compliance)

    best_score = float("inf")
    best_metrics: dict[str, float] = {}
    best_epoch = 0
    history: list[dict[str, float]] = []
    start = perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for x, log_rho_true, zxy_obs in train_loader:
            x = x.to(device)
            log_rho_true = log_rho_true.to(device)
            zxy_obs = zxy_obs.to(device)

            optimizer.zero_grad(set_to_none=True)
            log_rho_pred = model(x)
            rho_pred = torch.pow(10.0, log_rho_pred)
            zxy_pred = mt_forward_torch(rho_pred, thicknesses, frequencies)

            supervised = torch.mean((log_rho_pred - log_rho_true) ** 2)
            physical = physics_loss(zxy_pred, zxy_obs)
            reg = regularization_loss(log_rho_pred, attempt.regularizer)
            loss = (
                attempt.supervised_weight * supervised
                + attempt.physics_weight * physical
                + attempt.reg_weight * reg
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        metrics = evaluate(model, val_loader, thicknesses, frequencies, device)
        val_score = metrics["mse_model_log10_rho"] + attempt.physics_weight * metrics["physics_loss"]
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(epoch_losses)),
            "val_score": float(val_score),
            **metrics,
        }
        history.append(row)
        print(
            f"{attempt.name} epoch {epoch:03d} "
            f"train_loss={row['train_loss']:.6f} "
            f"val_logrho_mse={metrics['mse_model_log10_rho']:.6f} "
            f"val_z_mse={metrics['mse_impedance_complex']:.6e} "
            f"physics={metrics['physics_loss']:.6f}"
        )

        if val_score < best_score:
            best_score = val_score
            best_metrics = metrics
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "attempt": asdict(attempt),
                    "architecture": attempt.architecture,
                    "regularizer": attempt.regularizer,
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "frequencies": data["frequencies"],
                    "thicknesses": data["thicknesses"],
                    "best_epoch": best_epoch,
                    "best_metrics": best_metrics,
                    "compliance": compliance,
                },
                run_dir / "best_model.pt",
            )

    seconds = perf_counter() - start
    result = {
        "attempt": attempt.name,
        "architecture": attempt.architecture,
        "regularizer": attempt.regularizer,
        "reg_weight": attempt.reg_weight,
        "train_count": args.train_count,
        "val_count": args.val_count,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "seconds": seconds,
        "history": history,
        "best_metrics": best_metrics,
        "compliance": compliance,
        "compliance_ok": bool(
            compliance["uses_database_npz"]
            and compliance["forward_operator_modified"] is False
            and compliance["model_generator_modified"] is False
            and compliance["supervised_fidelity_term"]
            and compliance["physics_fidelity_term"]
            and compliance["regularization_term"]
        ),
    }
    save_json(run_dir / "metrics.json", result)
    append_summary(args.output_dir, {**result, **best_metrics})

    # A final numerical check that the local forward operator still reproduces
    # stored observations for validation targets.
    direct_check = mt_forward_numpy(data["val_rho"][:5], data["thicknesses"], data["frequencies"])
    result["forward_operator_max_abs_error_on_stored_models"] = float(
        np.max(np.abs(direct_check - data["val_zxy"][:5]))
    )
    save_json(run_dir / "metrics.json", result)
    return result


def main() -> None:
    args = parse_args()
    if args.list_attempts:
        print(available_attempts())
        return
    result = train_one(args)
    print(
        "BEST "
        f"{result['attempt']} epoch={result['best_epoch']} "
        f"logrho_mse={result['best_metrics']['mse_model_log10_rho']:.6f} "
        f"rho_mse={result['best_metrics']['mse_model_rho']:.6f} "
        f"z_mse={result['best_metrics']['mse_impedance_complex']:.6e} "
        f"compliance_ok={result['compliance_ok']}"
    )


if __name__ == "__main__":
    main()
