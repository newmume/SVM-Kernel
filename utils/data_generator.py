import numpy as np


def generate_ring_dataset(
    n_inner=35,
    n_outer=45,
    inner_radius_range=(0.0, 1.0),
    outer_radius_range=(1.6, 2.5),
    noise=0.08,
    random_seed=7,
):
    rng = np.random.default_rng(random_seed)

    inner_radii = rng.uniform(*inner_radius_range, n_inner)
    inner_angles = rng.uniform(0, 2 * np.pi, n_inner)
    X_inner = np.column_stack([
        inner_radii * np.cos(inner_angles),
        inner_radii * np.sin(inner_angles),
    ])

    outer_radii = rng.uniform(*outer_radius_range, n_outer)
    outer_angles = rng.uniform(0, 2 * np.pi, n_outer)
    X_outer = np.column_stack([
        outer_radii * np.cos(outer_angles),
        outer_radii * np.sin(outer_angles),
    ])

    X = np.vstack([X_inner, X_outer])
    y = np.hstack([np.zeros(n_inner), np.ones(n_outer)])

    if noise > 0:
        X += rng.normal(0, noise, X.shape)

    return X, y
