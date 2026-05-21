from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

try:
    from .mt_model import (
        build_model,
        mse_complex,
        mt_forward_numpy,
        observation_features,
        save_json,
        standardize,
    )
except ImportError:
    from mt_model import (
        build_model,
        mse_complex,
        mt_forward_numpy,
        observation_features,
        save_json,
        standardize,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained MT inversion network.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="A .npz file or a directory with .npz files.")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation_predictions.npz"))
    parser.add_argument("--metrics-json", type=Path, default=Path("outputs/evaluation_metrics.json"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_observations(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str]]:
    files = [path] if path.is_file() else sorted(path.glob("*.npz"))
    if not files:
        raise ValueError(f"No .npz files found in {path}")

    zxy: list[np.ndarray] = []
    rho: list[np.ndarray] = []
    names: list[str] = []
    frequencies: np.ndarray | None = None
    has_rho = True
    for file in files:
        data = np.load(file)
        if "zxy" not in data:
            raise ValueError(f"{file} does not contain required key 'zxy'.")
        if "frequencies" in data:
            freq = data["frequencies"].astype(np.float32)
            if frequencies is None:
                frequencies = freq
            elif not np.allclose(freq, frequencies):
                raise ValueError(f"Frequency grid differs in {file}.")
        zxy.append(data["zxy"].astype(np.complex64))
        if "resistivities" in data:
            rho.append(data["resistivities"].astype(np.float32))
        else:
            has_rho = False
        names.append(file.name)

    if frequencies is None:
        raise ValueError("Input observations must include a 'frequencies' array.")
    true_rho = np.stack(rho).astype(np.float32) if has_rho else None
    return np.stack(zxy).astype(np.complex64), frequencies, true_rho, names


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    model = build_model(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        architecture=str(checkpoint["architecture"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    zxy, frequencies, true_rho, names = load_observations(args.input)
    if not np.allclose(frequencies, checkpoint["frequencies"]):
        raise ValueError("Input frequencies differ from the frequency grid stored in the weights.")

    features = standardize(
        observation_features(zxy, frequencies),
        checkpoint["feature_mean"],
        checkpoint["feature_std"],
    )
    with torch.no_grad():
        log_rho_pred = model(torch.from_numpy(features.astype(np.float32)).to(device)).cpu().numpy()
    rho_pred = np.power(10.0, log_rho_pred).astype(np.float32)
    zxy_pred = mt_forward_numpy(rho_pred, checkpoint["thicknesses"], checkpoint["frequencies"]).astype(np.complex64)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        files=np.asarray(names),
        predicted_resistivities=rho_pred,
        predicted_zxy=zxy_pred,
        frequencies=checkpoint["frequencies"],
        thicknesses=checkpoint["thicknesses"],
    )

    metrics: dict[str, object] = {
        "weights": str(args.weights),
        "input": str(args.input),
        "count": int(len(names)),
        "architecture": checkpoint["architecture"],
        "regularizer": checkpoint["regularizer"],
    }
    if true_rho is not None:
        metrics["mse_model_rho"] = float(np.mean((rho_pred - true_rho) ** 2))
        metrics["mse_model_log10_rho"] = float(
            np.mean((np.log10(rho_pred) - np.log10(true_rho)) ** 2)
        )
    metrics["mse_impedance_complex"] = mse_complex(zxy_pred, zxy)
    save_json(args.metrics_json, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
