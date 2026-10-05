"""Linear isotropic plane-strain mixed operator, with generic boundary projections."""
from dataclasses import dataclass
import numpy as np
from features import values


@dataclass
class Condition:
    field: str  # displacement or traction
    direction: object  # constant vector, normal, or tangent
    target: object  # callable (xy, normal) -> [n, load_modes]


@dataclass
class Problem:
    domain: object
    E: float
    nu: float
    conditions: dict
    gauges: list
    n_loads: int
    body: object = None
    objective: str = 'integral'

    @property
    def lame(self):
        return (self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu)), self.E / (2 * (1 + self.nu)))


def constant(value):
    value = np.atleast_1d(np.asarray(value, float))
    return lambda xy, normal: np.broadcast_to(value, (len(xy), len(value)))


def directions(condition, boundary):
    if isinstance(condition.direction, str):
        return boundary[condition.direction]
    return np.broadcast_to(np.asarray(condition.direction), boundary['xy'].shape)


def rows(problem, sample):
    """Compact row descriptors permit both one-block and full-matrix assembly."""
    result = []
    domain = sample['domain']
    normalizers = {}
    for boundary in sample['boundary']:
        normalizers[boundary['tag']] = normalizers.get(boundary['tag'], 0.) + boundary['w'].sum()
    n = len(domain['xy'])
    zero = np.zeros((n, problem.n_loads))
    body = problem.body(domain['xy']) if problem.body else np.zeros((n, 2, problem.n_loads))
    for component, name in enumerate(['equilibrium_x', 'equilibrium_y', 'constitutive_xx', 'constitutive_yy', 'constitutive_xy']):
        rhs = -body[:, component] if component < 2 else zero
        domain_w = domain['w'] / problem.domain.area if problem.objective == 'group_mean' else domain['w']
        result.append(dict(kind=component, xy=domain['xy'], w=domain_w, weight=1., rhs=rhs,
                           name=name, normal=np.zeros((n, 2))))
    for boundary in sample['boundary']:
        for cond in problem.conditions[boundary['tag']]:
            boundary_w = boundary['w'] / normalizers[boundary['tag']] if problem.objective == 'group_mean' else boundary['w']
            result.append(dict(kind=cond.field, xy=boundary['xy'], w=boundary_w,
                               weight=100. if cond.field == 'displacement' else 10.,
                               normal=boundary['normal'], direction=directions(cond, boundary),
                               rhs=cond.target(boundary['xy'], boundary['normal']),
                               name=boundary['tag'] + '_' + cond.field))
    for gauge in problem.gauges:
        xy = np.atleast_2d(gauge['xy'])
        normal = np.zeros_like(xy)
        result.append(dict(kind='displacement', xy=xy, w=np.ones(len(xy)), weight=100., normal=normal,
                           direction=np.broadcast_to(gauge['direction'], xy.shape),
                           rhs=gauge['target'](xy, normal), name='gauge'))
    offset = 0
    for row in result:
        row['slice'] = (offset, offset + len(row['xy']))
        offset += len(row['xy'])
    return result


def assemble(problem, features, groups, include_rhs=True):
    lam, mu = problem.lame
    nrows = groups[-1]['slice'][1]
    matrix = np.zeros((nrows, len(features), 5))
    rhs = np.empty((nrows, problem.n_loads)) if include_rhs else None
    cache, airy_cache = {}, {}
    for row in groups:
        key = id(row['xy'])
        if key not in cache:
            cache[key] = values(features, row['xy'])
        phi, dx, dy = cache[key]
        a, b = row['slice']
        block = matrix[a:b]
        kind = row['kind']
        if kind == 0:
            block[:, :, 2], block[:, :, 4] = dx, dy
        elif kind == 1:
            block[:, :, 4], block[:, :, 3] = dx, dy
        elif kind == 2:
            block[:, :, 2] = phi
            block[:, :, 0], block[:, :, 1] = -(lam + 2 * mu) * dx, -lam * dy
        elif kind == 3:
            block[:, :, 3] = phi
            block[:, :, 0], block[:, :, 1] = -lam * dx, -(lam + 2 * mu) * dy
        elif kind == 4:
            block[:, :, 4] = phi
            block[:, :, 0], block[:, :, 1] = -mu * dy, -mu * dx
        elif kind == 'displacement':
            block[:, :, 0] = row['direction'][:, 0, None] * phi
            block[:, :, 1] = row['direction'][:, 1, None] * phi
        elif kind == 'traction':
            nx, ny = row['normal'].T
            qx, qy = row['direction'].T
            block[:, :, 2] = (nx * qx)[:, None] * phi
            block[:, :, 3] = (ny * qy)[:, None] * phi
            block[:, :, 4] = (ny * qx + nx * qy)[:, None] * phi
        else:
            raise ValueError(kind)
        for j, feature in enumerate(features):
            if feature.kind != 'airy':
                continue
            block[:, j, 2:] = 0.
            if kind in (0, 1, 'displacement'):
                continue
            akey = (j, key)
            if akey not in airy_cache:
                airy_cache[akey] = feature.airy_stress(row['xy'])
            stress = airy_cache[akey]
            if kind in (2, 3, 4):
                block[:, j, 2:] = stress[:, kind - 2, :]
            elif kind == 'traction':
                nx, ny = row['normal'].T
                qx, qy = row['direction'].T
                block[:, j, 2:] = ((nx * qx)[:, None] * stress[:, 0] + (ny * qy)[:, None] * stress[:, 1]
                                    + (ny * qx + nx * qy)[:, None] * stress[:, 2])
        factor = np.sqrt(row['weight'] * row['w'])
        block *= factor[:, None, None]
        if include_rhs:
            rhs[a:b] = row['rhs'] * factor[:, None]
    return matrix.reshape(nrows, -1), rhs


def residual_summary(residual, groups):
    result = {'total': float(np.sum(residual ** 2))}
    for row in groups:
        a, b = row['slice']
        result[row['name']] = result.get(row['name'], 0.) + float(np.sum(residual[a:b] ** 2))
    return result


def evaluate_residual(problem, features, coefficients, groups):
    """Independent field-level residual path; avoids a full check-set design matrix."""
    coeff = np.asarray(coefficients).reshape(len(features), 5, -1)
    lam, mu = problem.lame
    cache, output = {}, []
    for row in groups:
        key = id(row['xy'])
        if key not in cache:
            phi, dx, dy = values(features, row['xy'])
            conventional = coeff.copy()
            for j, feature in enumerate(features):
                if feature.kind == 'airy':
                    conventional[j, 2:] = 0.
            field, gx, gy = tuple(np.einsum('nf,fkl->nkl', v, conventional) for v in [phi, dx, dy])
            for j, feature in enumerate(features):
                if feature.kind == 'airy':
                    field[:, 2:] += np.einsum('nsc,cl->nsl', feature.airy_stress(row['xy']), coeff[j, 2:])
            # gx/gy contain conventional stress derivatives only. Their divergence
            # is the full divergence because every added Airy stress has div=0.
            cache[key] = field, gx, gy
        y, dx, dy = cache[key]
        kind = row['kind']
        if kind == 0:
            lhs = dx[:, 2] + dy[:, 4]
        elif kind == 1:
            lhs = dx[:, 4] + dy[:, 3]
        elif kind == 2:
            lhs = y[:, 2] - (lam + 2 * mu) * dx[:, 0] - lam * dy[:, 1]
        elif kind == 3:
            lhs = y[:, 3] - lam * dx[:, 0] - (lam + 2 * mu) * dy[:, 1]
        elif kind == 4:
            lhs = y[:, 4] - mu * (dy[:, 0] + dx[:, 1])
        elif kind == 'displacement':
            qx, qy = row['direction'].T
            lhs = qx[:, None] * y[:, 0] + qy[:, None] * y[:, 1]
        else:
            nx, ny = row['normal'].T
            qx, qy = row['direction'].T
            lhs = (nx * qx)[:, None] * y[:, 2] + (ny * qy)[:, None] * y[:, 3]
            lhs += (ny * qx + nx * qy)[:, None] * y[:, 4]
        output.append((lhs - row['rhs']) * np.sqrt(row['weight'] * row['w'])[:, None])
    return np.concatenate(output)


def check_sample(domain, seed, n=8192, spacing=.025):
    rng = np.random.default_rng(seed)
    return dict(domain={'xy': domain.random_interior(n, rng), 'w': np.full(n, domain.area / n)},
                boundary=domain.boundary(spacing, rng=rng))
