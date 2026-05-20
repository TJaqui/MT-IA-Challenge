import argparse
from pathlib import Path

import numpy as np


MU_0 = 4e-7 * np.pi
DEFAULT_FREQUENCIES = np.logspace(-3, 3, 31)


def mt_forward_model(resistivities, thicknesses, frequencies):
    n_layers = len(resistivities)
    zxy = []

    for frequency in frequencies:
        omega = 2 * np.pi * frequency
        impedances = [0j] * n_layers
        impedances[n_layers - 1] = np.sqrt(omega * MU_0 * resistivities[n_layers - 1] * 1j)

        for layer in range(n_layers - 2, -1, -1):
            resistivity = resistivities[layer]
            thickness = thicknesses[layer]

            dj = np.sqrt((omega * MU_0 * (1.0 / resistivity)) * 1j)
            wj = dj * resistivity
            ej = np.exp(-2 * thickness * dj)

            below_impedance = impedances[layer + 1]
            reflection = (wj - below_impedance) / (wj + below_impedance)
            reflected_exponential = reflection * ej
            impedances[layer] = wj * ((1 - reflected_exponential) / (1 + reflected_exponential))

        zxy.append(impedances[0])

    return np.array(zxy)


def random_model(rng, n_layers=10, depthmax=20000, log10_resistivity_range=(0.0, 3.0)):
    indexes = rng.choice(200, n_layers, replace=True)
    log10_resistivities = rng.uniform(
        log10_resistivity_range[0],
        log10_resistivity_range[1],
        200,
    )
    resistivities = 10 ** log10_resistivities[indexes]

    dz = depthmax // (n_layers - 1)
    depths = np.arange(n_layers) * dz
    thicknesses = np.diff(depths)
    return resistivities, thicknesses


def generate_database(
    output_dir,
    count=1_000_000,
    n_layers=10,
    depthmax=20000,
    min_log10_resistivity=0.0,
    max_log10_resistivity=3.0,
    seed=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    log10_range = (min_log10_resistivity, max_log10_resistivity)

    for index in range(count):
        resistivities, thicknesses = random_model(
            rng,
            n_layers=n_layers,
            depthmax=depthmax,
            log10_resistivity_range=log10_range,
        )
        zxy = mt_forward_model(resistivities, thicknesses, DEFAULT_FREQUENCIES)

        output_file = output_dir / f"model_{index:07d}.npz"
        np.savez_compressed(
            output_file,
            resistivities=resistivities,
            thicknesses=thicknesses,
            frequencies=DEFAULT_FREQUENCIES,
            zxy=zxy,
        )

        if (index + 1) % 1000 == 0:
            print(f"Generated {index + 1}/{count} files")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate random 1D MT model files using the notebook forward model."
    )
    parser.add_argument("--output-dir", default="database", type=Path)
    parser.add_argument("--count", default=1_000_000, type=int)
    parser.add_argument("--n-layers", default=10, type=int)
    parser.add_argument("--depthmax", default=20000, type=int)
    parser.add_argument("--min-log10-resistivity", default=0.0, type=float)
    parser.add_argument("--max-log10-resistivity", default=3.0, type=float)
    parser.add_argument("--seed", default=None, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    generate_database(
        output_dir=args.output_dir,
        count=args.count,
        n_layers=args.n_layers,
        depthmax=args.depthmax,
        min_log10_resistivity=args.min_log10_resistivity,
        max_log10_resistivity=args.max_log10_resistivity,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
