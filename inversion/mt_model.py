from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn


MU0 = 4.0 * math.pi * 1e-7
RHO_MIN = 1.0
RHO_MAX = 1000.0
LOG_RHO_MIN = math.log10(RHO_MIN)
LOG_RHO_MAX = math.log10(RHO_MAX)


@dataclass(frozen=True)
class AttemptConfig:
    name: str
    architecture: str
    regularizer: str
    reg_weight: float
    supervised_weight: float = 1.0
    physics_weight: float = 0.15


ATTEMPTS: tuple[AttemptConfig, ...] = (
    AttemptConfig(
        name="a1_mlp_smooth_l2",
        architecture="mlp_128x3",
        regularizer="smooth_l2",
        reg_weight=0.015,
    ),
    AttemptConfig(
        name="a2_mlp_dropout_smooth_tv",
        architecture="mlp_256x4_dropout",
        regularizer="smooth_tv",
        reg_weight=0.01,
    ),
    AttemptConfig(
        name="a3_residual_smooth_bounds",
        architecture="residual_192x4",
        regularizer="smooth_bounds",
        reg_weight=0.02,
    ),
)


def attempt_by_name(name: str) -> AttemptConfig:
    for attempt in ATTEMPTS:
        if attempt.name == name:
            return attempt
    names = ", ".join(a.name for a in ATTEMPTS)
    raise ValueError(f"Unknown attempt '{name}'. Available attempts: {names}")


def available_attempts() -> list[dict[str, object]]:
    return [asdict(a) for a in ATTEMPTS]


def set_reproducible_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def npz_path(database_dir: Path, index: int) -> Path:
    return database_dir / f"model_{index:07d}.npz"


def read_npz_sample(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    return (
        data["resistivities"].astype(np.float32),
        data["thicknesses"].astype(np.float32),
        data["frequencies"].astype(np.float32),
        data["zxy"].astype(np.complex64),
    )


def select_indices(total_files: int, count: int, seed: int) -> np.ndarray:
    if count > total_files:
        raise ValueError(f"Requested {count} samples, but only {total_files} files exist.")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total_files, size=count, replace=False))


def load_database_subset(
    database_dir: Path,
    train_count: int,
    val_count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    files = sorted(database_dir.glob("model_*.npz"))
    total = len(files)
    needed = train_count + val_count
    indices = select_indices(total, needed, seed)

    resistivities: list[np.ndarray] = []
    zxy: list[np.ndarray] = []
    frequencies: np.ndarray | None = None
    thicknesses: np.ndarray | None = None

    for index in indices:
        rho, th, freq, obs = read_npz_sample(npz_path(database_dir, int(index)))
        if frequencies is None:
            frequencies = freq
            thicknesses = th
        else:
            if not np.allclose(freq, frequencies):
                raise ValueError(f"Frequency grid differs in model_{index:07d}.npz")
            if not np.allclose(th, thicknesses):
                raise ValueError(f"Thickness vector differs in model_{index:07d}.npz")
        resistivities.append(rho)
        zxy.append(obs)

    if frequencies is None or thicknesses is None:
        raise ValueError("No samples were loaded.")

    rho_all = np.stack(resistivities).astype(np.float32)
    zxy_all = np.stack(zxy).astype(np.complex64)
    split = train_count
    return {
        "train_rho": rho_all[:split],
        "train_zxy": zxy_all[:split],
        "val_rho": rho_all[split:],
        "val_zxy": zxy_all[split:],
        "frequencies": frequencies.astype(np.float32),
        "thicknesses": thicknesses.astype(np.float32),
        "indices": indices.astype(np.int64),
    }


def observation_features(zxy: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    zxy = np.asarray(zxy)
    if zxy.ndim == 1:
        zxy = zxy[None, :]
    log_abs = np.log10(np.abs(zxy) + 1e-12).astype(np.float32)
    phase = np.angle(zxy).astype(np.float32)
    log_freq = np.log10(np.asarray(frequencies, dtype=np.float32) + 1e-30)
    log_freq = np.broadcast_to(log_freq[None, :], log_abs.shape).astype(np.float32)
    return np.concatenate([log_abs, phase, log_freq], axis=1)


def standardize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (features - mean) / np.maximum(std, 1e-6)


def target_log_resistivity(resistivities: np.ndarray) -> np.ndarray:
    return np.log10(np.asarray(resistivities, dtype=np.float32))


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.block(x))


class BoundedLogRhoNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, architecture: str) -> None:
        super().__init__()
        self.architecture = architecture
        if architecture == "mlp_128x3":
            layers: list[nn.Module] = []
            width = 128
            prev = input_dim
            for _ in range(3):
                layers.extend([nn.Linear(prev, width), nn.GELU(), nn.LayerNorm(width)])
                prev = width
            layers.append(nn.Linear(prev, output_dim))
            self.net = nn.Sequential(*layers)
        elif architecture == "mlp_256x4_dropout":
            layers = []
            width = 256
            prev = input_dim
            for _ in range(4):
                layers.extend(
                    [
                        nn.Linear(prev, width),
                        nn.GELU(),
                        nn.LayerNorm(width),
                        nn.Dropout(p=0.08),
                    ]
                )
                prev = width
            layers.append(nn.Linear(prev, output_dim))
            self.net = nn.Sequential(*layers)
        elif architecture == "residual_192x4":
            width = 192
            self.net = nn.Sequential(
                nn.Linear(input_dim, width),
                nn.GELU(),
                nn.LayerNorm(width),
                ResidualBlock(width),
                ResidualBlock(width),
                ResidualBlock(width),
                ResidualBlock(width),
                nn.Linear(width, output_dim),
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        return LOG_RHO_MIN + (LOG_RHO_MAX - LOG_RHO_MIN) * torch.sigmoid(raw)


def build_model(input_dim: int, output_dim: int, architecture: str) -> BoundedLogRhoNet:
    return BoundedLogRhoNet(input_dim=input_dim, output_dim=output_dim, architecture=architecture)


def mt_forward_torch(
    resistivities: torch.Tensor,
    thicknesses: torch.Tensor,
    frequencies: torch.Tensor,
) -> torch.Tensor:
    rho = resistivities.to(torch.float32)
    th = thicknesses.to(device=rho.device, dtype=torch.float32)
    freq = frequencies.to(device=rho.device, dtype=torch.float32)
    omega = 2.0 * math.pi * freq[None, :]
    one_i = torch.tensor(1j, device=rho.device, dtype=torch.complex64)

    z = torch.sqrt(one_i * omega.to(torch.complex64) * MU0 * rho[:, -1, None].to(torch.complex64))
    for layer in range(rho.shape[1] - 2, -1, -1):
        rho_j = rho[:, layer, None]
        dj = torch.sqrt(one_i * omega.to(torch.complex64) * MU0 / rho_j.to(torch.complex64))
        wj = dj * rho_j.to(torch.complex64)
        ej = torch.exp(-2.0 * dj * th[layer])
        rj = (wj - z) / (wj + z)
        z = wj * (1.0 - rj * ej) / (1.0 + rj * ej)
    return z


def mt_forward_numpy(
    resistivities: np.ndarray,
    thicknesses: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray:
    rho = np.asarray(resistivities, dtype=np.float64)
    if rho.ndim == 1:
        rho = rho[None, :]
    th = np.asarray(thicknesses, dtype=np.float64)
    freq = np.asarray(frequencies, dtype=np.float64)
    out = np.empty((rho.shape[0], freq.size), dtype=np.complex128)
    for sample in range(rho.shape[0]):
        for fi, f in enumerate(freq):
            omega = 2.0 * np.pi * f
            z = np.sqrt(1j * omega * MU0 * rho[sample, -1])
            for layer in range(rho.shape[1] - 2, -1, -1):
                dj = np.sqrt(1j * omega * MU0 / rho[sample, layer])
                wj = dj * rho[sample, layer]
                ej = np.exp(-2.0 * dj * th[layer])
                rj = (wj - z) / (wj + z)
                z = wj * (1.0 - rj * ej) / (1.0 + rj * ej)
            out[sample, fi] = z
    return out


def physics_loss(pred_zxy: torch.Tensor, obs_zxy: torch.Tensor) -> torch.Tensor:
    pred_log_abs = torch.log10(torch.abs(pred_zxy) + 1e-12)
    obs_log_abs = torch.log10(torch.abs(obs_zxy) + 1e-12)
    pred_phase = torch.angle(pred_zxy)
    obs_phase = torch.angle(obs_zxy)
    phase_delta = torch.atan2(torch.sin(pred_phase - obs_phase), torch.cos(pred_phase - obs_phase))
    return torch.mean((pred_log_abs - obs_log_abs) ** 2) + torch.mean(phase_delta**2)


def regularization_loss(log_rho: torch.Tensor, regularizer: str) -> torch.Tensor:
    diffs = log_rho[:, 1:] - log_rho[:, :-1]
    if regularizer == "smooth_l2":
        return torch.mean(diffs**2)
    if regularizer == "smooth_tv":
        return torch.mean(torch.sqrt(diffs**2 + 1e-6))
    if regularizer == "smooth_bounds":
        smooth = torch.mean(diffs**2)
        low = torch.relu(LOG_RHO_MIN - log_rho)
        high = torch.relu(log_rho - LOG_RHO_MAX)
        bounds = torch.mean(low**2 + high**2)
        return smooth + bounds
    raise ValueError(f"Unsupported regularizer: {regularizer}")


def mse_complex(pred: np.ndarray, obs: np.ndarray) -> float:
    pred = np.asarray(pred)
    obs = np.asarray(obs)
    return float(np.mean((pred.real - obs.real) ** 2 + (pred.imag - obs.imag) ** 2))


def compliance_report(attempt: AttemptConfig, model_layers: int, thickness_layers: int) -> dict[str, object]:
    return {
        "uses_database_npz": True,
        "model_layers_from_database": model_layers,
        "thickness_layers_from_database": thickness_layers,
        "supervised_fidelity_term": "MSE(log10 resistivity true, G(d))",
        "physics_fidelity_term": "MSE(log10 |d|, log10 |F(G(d))|) + phase MSE",
        "regularization_term": attempt.regularizer,
        "forward_operator_modified": False,
        "model_generator_modified": False,
        "only_changed_between_attempts": ["architecture", "regularizer", "reg_weight"],
        "architecture": attempt.architecture,
        "regularizer": attempt.regularizer,
        "reg_weight": attempt.reg_weight,
        "rho_output_bounds_ohm_m": [RHO_MIN, RHO_MAX],
    }


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def batch_iter(indices: Iterable[int], batch_size: int) -> Iterable[np.ndarray]:
    batch: list[int] = []
    for index in indices:
        batch.append(int(index))
        if len(batch) == batch_size:
            yield np.asarray(batch, dtype=np.int64)
            batch.clear()
    if batch:
        yield np.asarray(batch, dtype=np.int64)
