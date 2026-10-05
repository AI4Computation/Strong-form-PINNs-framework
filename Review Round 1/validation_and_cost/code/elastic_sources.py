"""Classical plane-strain Kelvin features, affine completion and boundary rows.

This is an implementation of established fundamental-solution mechanics, not a
claim of a new kernel or a new MFS. All singular sources must lie outside rock.
"""
import numpy as np
from mechanics import directions


def fields(xy, sources, E, nu, affine=True):
    xy, sources = np.asarray(xy), np.asarray(sources).reshape(-1, 2)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    blocks = []
    if affine:
        a = np.zeros((len(xy), 5, 6))
        a[:, 0, 0] = 1.; a[:, 1, 1] = 1.
        a[:, 0, 2:4] = xy; a[:, 1, 4:6] = xy
        a[:, 2, 2] = lam + 2 * mu; a[:, 2, 5] = lam
        a[:, 3, 2] = lam; a[:, 3, 5] = lam + 2 * mu
        a[:, 4, 3] = mu; a[:, 4, 4] = mu
        blocks.append(a)
    if len(sources):
        r = xy[:, None, :] - sources[None, :, :]
        r2 = np.sum(r * r, axis=2)
        if np.any(r2 < 1e-24):
            raise ValueError('A source coincides with an evaluation point.')
        inverse = 1 / r2
        factor = 1 / (8 * np.pi * mu * (1 - nu))
        kappa = 3 - 4 * nu
        eye = np.eye(2)
        U = factor * (-.5 * kappa * np.log(r2)[:, :, None, None] * eye +
                       r[:, :, :, None] * r[:, :, None, :] * inverse[:, :, None, None])
        gradient = np.empty((len(xy), len(sources), 2, 2, 2))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    gradient[:, :, i, j, k] = factor * (
                        -kappa * eye[i, j] * r[:, :, k] * inverse +
                        (eye[i, k] * r[:, :, j] + r[:, :, i] * eye[j, k]) * inverse -
                        2 * r[:, :, i] * r[:, :, j] * r[:, :, k] * inverse ** 2)
        trace = gradient[:, :, 0, :, 0] + gradient[:, :, 1, :, 1]
        S = np.stack([lam * trace + 2 * mu * gradient[:, :, 0, :, 0],
                      lam * trace + 2 * mu * gradient[:, :, 1, :, 1],
                      mu * (gradient[:, :, 0, :, 1] + gradient[:, :, 1, :, 0])], axis=2)
        combined = np.concatenate([U, S], axis=2)
        blocks.append(combined.transpose(0, 2, 1, 3).reshape(len(xy), 5, -1))
    return np.concatenate(blocks, axis=2) if blocks else np.empty((len(xy), 5, 0))


def boundary_rows(problem, spacing, order=5, rng=None):
    boundary = problem.domain.boundary(spacing, order=order, rng=rng)
    totals = {}
    for b in boundary:
        totals[b['tag']] = totals.get(b['tag'], 0.) + b['w'].sum()
    groups = []
    for curve_index, b in enumerate(boundary):
        for c in problem.conditions[b['tag']]:
            groups.append(dict(xy=b['xy'], normal=b['normal'], kind=c.field,
                               direction=directions(c, b), weight=(100. if c.field == 'displacement' else 10.),
                               w=b['w'] / totals[b['tag']], target=c.target(b['xy'], b['normal']),
                               name=b['tag'] + '_' + c.field, curve_index=curve_index))
    for g in problem.gauges:
        xy = np.atleast_2d(g['xy']); n = np.zeros_like(xy)
        groups.append(dict(xy=xy, normal=n, kind='displacement', direction=np.broadcast_to(g['direction'], xy.shape),
                           weight=100., w=np.ones(len(xy)), target=g['target'](xy, n), name='gauge', curve_index=-1))
    return groups


def assemble(problem, sources, groups, affine=True, field_evaluator=None):
    matrices, targets, cache = [], [], {}
    field_evaluator = fields if field_evaluator is None else field_evaluator
    for g in groups:
        key = id(g['xy'])
        if key not in cache:
            cache[key] = field_evaluator(g['xy'], sources, problem.E, problem.nu, affine)
        f = cache[key]
        qx, qy = g['direction'].T
        if g['kind'] == 'displacement':
            block = qx[:, None] * f[:, 0] + qy[:, None] * f[:, 1]
        else:
            nx, ny = g['normal'].T
            block = (nx * qx)[:, None] * f[:, 2] + (ny * qy)[:, None] * f[:, 3]
            block += (ny * qx + nx * qy)[:, None] * f[:, 4]
        weight = np.sqrt(g['weight'] * g['w'])[:, None]
        matrices.append(block * weight); targets.append(g['target'] * weight)
    return np.concatenate(matrices), np.concatenate(targets)


def source_from_boundary(domain, xy, normal, level):
    """Find a safe exterior offset from the first rock re-entry along the normal.

    The same ray test works for outer boundaries, holes, narrow voids and facets.
    No radius, ellipse axis, tunnel label or FEM value enters the rule.
    """
    length = float(np.max(domain.hi - domain.lo))
    distances = np.geomspace(length * 1e-7, length, 80)
    rock = domain.contains(xy + distances[:, None] * normal)
    if rock[0]:
        return None
    hit = np.flatnonzero(rock)
    if len(hit):
        i = int(hit[0]); lo, hi = distances[i - 1], distances[i]
        for _ in range(30):
            mid = (lo + hi) / 2
            if domain.contains((xy + mid * normal)[None])[0]:
                hi = mid
            else:
                lo = mid
        clearance = lo
    else:
        clearance = .5 * length
    offset = clearance / 2 ** level
    source = xy + offset * normal
    if offset < 1e-6 * length or domain.contains(source[None])[0]:
        return None
    return dict(xy=source.tolist(), boundary_xy=xy.tolist(), normal=normal.tolist(),
                offset=float(offset), first_reentry_clearance=float(clearance), level=int(level))


def candidates(domain, groups, residual, rng, count=24):
    points, normals, scores, weights = [], [], [], []
    start = 0
    for g in groups:
        stop = start + len(g['xy'])
        if g['curve_index'] >= 0:
            points.append(g['xy']); normals.append(g['normal'])
            scores.append(np.sum(residual[start:stop] ** 2, axis=1))
            weights.append(g['w'])
        start = stop
    xy, normal, score, weight = map(np.concatenate, [points, normals, scores, weights])
    probability = .75 * (score + 1e-20) / (score.sum() + len(score) * 1e-20)
    probability += .25 * weight / weight.sum()
    result = []
    for _ in range(10 * count):
        idx = rng.choice(len(xy), p=probability)
        candidate = source_from_boundary(domain, xy[idx], normal[idx], int(rng.integers(2, 7)))
        if candidate is not None:
            result.append(candidate)
        if len(result) == count:
            break
    return result


def predict(problem, sources, coefficients, xy, batch=2048):
    return np.concatenate([fields(xy[i:i + batch], sources, problem.E, problem.nu) @ coefficients
                           for i in range(0, len(xy), batch)])
