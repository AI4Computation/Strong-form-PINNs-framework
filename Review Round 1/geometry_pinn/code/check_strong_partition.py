"""CPU algebra checks only: no training, FEM reads, or first-round imports."""
from pathlib import Path
import hashlib
import json
from datetime import datetime
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "geometry_pinn/protocols/R2_P0_operator_check.json"
OUTPUT = ROOT / "geometry_pinn/results/R2_P0_operator_check.json"


def partition(x, centres, metrics):
    delta = x[:, None, :] - centres[None, :, :]
    score_gradient = -np.einsum("kij,nkj->nki", metrics, delta)
    scores = 0.5 * np.sum(delta * score_gradient, axis=-1)
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=1, keepdims=True)
    mean_gradient = np.sum(weights[..., None] * score_gradient, axis=1)
    gradients = weights[..., None] * (score_gradient - mean_gradient[:, None, :])
    return weights, gradients


def field(x, centres, metrics, offsets, slopes):
    weights, gradients = partition(x, centres, metrics)
    local = offsets[None, :, :] + np.einsum("nd,kod->nko", x, slopes)
    values = np.einsum("nk,nko->no", weights, local)
    local_derivative_only = np.einsum("nk,kod->nod", weights, slopes)
    jacobian = local_derivative_only + np.einsum("nkd,nko->nod", gradients, local)
    return values, jacobian, local_derivative_only


def mixed_residual(values, jacobian, young=1.333, poisson=0.26):
    """Plane-strain output order: ux, uy, sxx, syy, sxy. No body force."""
    mu = young / (2 * (1 + poisson))
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    strain_x = jacobian[:, 0, 0]
    strain_y = jacobian[:, 1, 1]
    shear_twice = jacobian[:, 0, 1] + jacobian[:, 1, 0]
    return np.column_stack([
        jacobian[:, 2, 0] + jacobian[:, 4, 1],
        jacobian[:, 4, 0] + jacobian[:, 3, 1],
        values[:, 2] - (lam + 2 * mu) * strain_x - lam * strain_y,
        values[:, 3] - lam * strain_x - (lam + 2 * mu) * strain_y,
        values[:, 4] - mu * shear_twice,
    ])


def maximum(x):
    return float(np.max(np.abs(x)))


def run():
    assert not OUTPUT.exists(), "Result already frozen; inspect before any replacement."
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    rng = np.random.default_rng(protocol["random_seed"])
    n, k = protocol["points"], protocol["local_fields"]
    x = rng.uniform(-1, 1, (n, 2))
    centres = rng.uniform(-0.7, 0.7, (k, 2))
    transforms = rng.normal(size=(k, 2, 2))
    metrics = np.einsum("kji,kjl->kil", transforms, transforms) + 0.5 * np.eye(2)
    a = rng.normal(size=(k, 5))
    b = rng.normal(size=(k, 5, 2))
    weights, dw = partition(x, centres, metrics)
    values, jacobian, incomplete = field(x, centres, metrics, a, b)
    fd = np.empty_like(jacobian)
    h = 1e-4
    for j in range(2):
        direction = np.zeros(2)
        direction[j] = h
        evaluate = lambda xx: field(xx, centres, metrics, a, b)[0]
        fd[:, :, j] = (-evaluate(x + 2 * direction) + 8 * evaluate(x + direction)
                       - 8 * evaluate(x - direction) + evaluate(x - 2 * direction)) / (12 * h)

    # Independent manufactured affine elastic patch, identical in every subspace.
    displacement_offset = np.array([0.11, -0.04])
    displacement_gradient = np.array([[0.12, -0.08], [0.03, -0.07]])
    young, poisson = 1.333, 0.26
    mu = young / (2 * (1 + poisson))
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    stress = np.array([
        (lam + 2 * mu) * 0.12 + lam * (-0.07),
        lam * 0.12 + (lam + 2 * mu) * (-0.07),
        mu * (-0.08 + 0.03),
    ])
    patch_a = np.tile(np.r_[displacement_offset, stress], (k, 1))
    patch_b = np.zeros((k, 5, 2))
    patch_b[:, :2] = displacement_gradient
    patch, patch_jac, _ = field(x, centres, metrics, patch_a, patch_b)
    exact = np.column_stack([x @ displacement_gradient.T + displacement_offset,
                             np.tile(stress, (n, 1))])
    zero, _, _ = field(x, centres, metrics, np.zeros_like(a), np.zeros_like(b))
    errors = {
        "partition_sum": maximum(weights.sum(axis=1) - 1),
        "partition_gradient_sum": maximum(dw.sum(axis=1)),
        "affine_reproduction": maximum(patch - exact),
        "elastic_patch_residual": maximum(mixed_residual(patch, patch_jac)),
        "finite_difference_gradient": maximum(jacobian - fd),
        "finite_difference_mixed_residual": maximum(mixed_residual(values, jacobian) - mixed_residual(values, fd)),
    }
    checks = {key: value <= protocol["tolerances"][key] for key, value in errors.items()}
    missing_window_error = maximum(mixed_residual(values, incomplete) - mixed_residual(values, fd))
    checks["negative_control_detected"] = missing_window_error > 1e-3
    checks["zero_correction_preserves_parent"] = bool(np.array_equal(values + zero, values))
    checks["all_windows_positive"] = bool(np.all(weights > 0))
    result = {
        "id": protocol["id"],
        "completed_local": datetime.now().astimezone().isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "errors": errors,
        "omitted_window_gradient_residual_discrepancy": missing_window_error,
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "training_runs": 0,
        "fem_access": False,
        "timing_valid_for_comparison": False,
        "scope": protocol["scope_limit"],
        "next": "Cavity geometry construction and neural implementation remain unverified.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
