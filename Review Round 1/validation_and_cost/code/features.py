"""Scalar features and exact first derivatives, shared by all five field outputs."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Feature:
    kind: str
    powers: tuple = (0, 0)
    center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    axes: tuple = (1., 1.)
    angle: float = 0.
    v: np.ndarray = field(default_factory=lambda: np.ones(2))
    bias: float = 0.

    @property
    def H(self):
        a = self.angle
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.diag(self.axes)

    def evaluate(self, xy):
        xy = np.asarray(xy)
        if self.kind == 'polynomial':
            x, y = (2 * xy).T
            p, q = self.powers
            val = x ** p * y ** q
            dx = 2 * p * x ** max(p - 1, 0) * y ** q
            dy = 2 * q * x ** p * y ** max(q - 1, 0)
            return val, dx, dy
        inverse = np.linalg.inv(self.H)
        xi = (xy - self.center) @ inverse.T
        gate = np.maximum(1 - np.sum(xi * xi, axis=1), 0)
        window = gate ** 3
        gw = -6 * gate[:, None] ** 2 * (xi @ inverse)
        tanh = np.tanh(xi @ self.v + self.bias)
        gt = (1 - tanh ** 2)[:, None] * (self.v @ inverse)
        gradient = gw * tanh[:, None] + window[:, None] * gt
        return window * tanh, gradient[:, 0], gradient[:, 1]

    def record(self):
        return dict(kind=self.kind, powers=list(self.powers), center=self.center.tolist(),
                   axes=list(self.axes), angle=float(self.angle), v=self.v.tolist(), bias=float(self.bias))

    def airy_stress(self, xy):
        """Three local Airy potentials: phi, (x-cx)phi, (y-cy)phi.

        Each column represents (Phi_yy, Phi_xx, -Phi_xy), hence is divergence
        free. The C2 compact potential gives continuous, zero-edge stresses.
        """
        xy = np.asarray(xy)
        inverse = np.linalg.inv(self.H)
        xi = (xy - self.center) @ inverse.T
        q = np.maximum(1 - np.sum(xi * xi, axis=1), 0)
        dq = -2 * (xi @ inverse)
        ddq = -2 * inverse.T @ inverse
        w = q ** 3
        dw = 3 * q[:, None] ** 2 * dq
        ddw = 6 * q[:, None, None] * np.einsum('ni,nj->nij', dq, dq) + 3 * q[:, None, None] ** 2 * ddq
        k = self.v @ inverse
        t = np.tanh(xi @ self.v + self.bias)
        dt = 1 - t ** 2
        grad = dw * t[:, None] + w[:, None] * dt[:, None] * k
        hessian = ddw * t[:, None, None]
        hessian += dt[:, None, None] * (dw[:, :, None] * k[None, None, :] + k[None, :, None] * dw[:, None, :])
        hessian -= (2 * w * t * dt)[:, None, None] * np.outer(k, k)
        potentials = [hessian]
        for axis in range(2):
            unit = np.eye(2)[axis]
            hh = (xy[:, axis] - self.center[axis])[:, None, None] * hessian
            hh += unit[None, :, None] * grad[:, None, :] + grad[:, :, None] * unit[None, None, :]
            potentials.append(hh)
        return np.stack([np.column_stack([h[:, 1, 1], h[:, 0, 0], -h[:, 0, 1]]) for h in potentials], axis=2)

    @classmethod
    def from_record(cls, r):
        return cls(r['kind'], tuple(r['powers']), np.asarray(r['center']), tuple(r['axes']),
                   r['angle'], np.asarray(r['v']), r['bias'])


def coarse_features(degree=3):
    return [Feature('polynomial', (p, total - p)) for total in range(degree + 1) for p in range(total + 1)]


def values(features, xy):
    data = [f.evaluate(xy) for f in features]
    return tuple(np.column_stack([a[k] for a in data]) for k in range(3))


def predict(features, coefficients, xy, batch=4096):
    coefficients = np.asarray(coefficients).reshape(len(features), 5, -1)
    out = []
    for start in range(0, len(xy), batch):
        points = xy[start:start + batch]
        phi = np.column_stack([f.bank.values_only(points)[:, f.index] if f.kind == 'frozen_hidden'
                               else f.evaluate(points)[0] for f in features])
        y = (phi @ coefficients.reshape(len(features), -1)).reshape(len(points), 5, -1)
        for j, f in enumerate(features):
            if f.kind == 'airy':
                y[:, 2:] -= phi[:, j, None, None] * coefficients[j, 2:][None]
                y[:, 2:] += np.einsum('nsc,cl->nsl', f.airy_stress(points), coefficients[j, 2:])
        out.append(y)
    return np.concatenate(out)


def candidate_pool(domain, check, residual, groups, rng, count=24, max_level=4):
    """All locations/scales follow input geometry or physical residuals, never FEM labels."""
    centres, normals, scores = [], [], []
    # Group slices contain already weighted, load-normalized residuals.
    for group in groups:
        a, b = group['slice']
        score = np.sum(residual[a:b] ** 2, axis=1)
        centres.append(group['xy'])
        normals.append(group['normal'])
        scores.append(score)
    centres, normals, scores = np.concatenate(centres), np.concatenate(normals), np.concatenate(scores)
    probabilities = (scores + 1e-15) / np.sum(scores + 1e-15)
    result, rejected = [], 0
    for attempt in range(20 * count):
        mode = attempt % 4
        if mode < 2:
            idx = rng.choice(len(centres), p=probabilities)
            centre, normal = centres[idx].copy(), normals[idx]
        elif mode == 2:
            curve = domain.curves[int(rng.integers(len(domain.curves)))]
            xy, normal, _, _ = curve.evaluate(rng.random(1))
            centre, normal = xy[0], normal[0]
        else:
            centre, normal = domain.random_interior(1, rng)[0], np.zeros(2)
        level = int(rng.integers(0, max_level + 1))
        scale = float(np.max(domain.hi - domain.lo) / 2 ** level)
        anisotropy = int(rng.integers(0, 2))
        axes = (scale, scale / 2 ** anisotropy)
        angle = np.arctan2(normal[1], normal[0]) + np.pi / 2 if np.linalg.norm(normal) else rng.uniform(0, np.pi)
        v = rng.normal(size=2)
        v = 2 * v / np.linalg.norm(v)
        bias = rng.uniform(-1.5, 1.5)
        if attempt % 5 == 0:
            v, bias = np.zeros(2), 1.
        feature = Feature('local', center=centre, axes=axes, angle=angle, v=v, bias=bias)
        if not domain.support_connected(feature):
            rejected += 1
            continue
        result.append(feature)
        if len(result) == count:
            return result, rejected
    return result, rejected
