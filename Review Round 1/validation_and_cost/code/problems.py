"""Problem data adapters only; the adaptive algorithm never branches on these names."""
import json
from pathlib import Path
import numpy as np
from geometry import Domain
from mechanics import Problem, Condition, constant

ROOT = Path(__file__).resolve().parents[1]
REVISION = ROOT.parent
OUTER = [[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]
TAGS = ['bottom', 'right', 'top', 'left']


def geometry(name):
    if name == 'circle':
        holes = [dict(kind='ellipse', tag='cavity', center=[0., 0.], axes=[.1, .1])]
    elif name == 'square':
        holes = [dict(kind='polygon', tag='cavity', vertices=[[-.1, -.1], [.1, -.1], [.1, .1], [-.1, .1]])]
    elif name == 'slender':
        holes = [dict(kind='polygon', tag='cavity', vertices=[[-.2, -.015], [.2, -.015], [.2, .015], [-.2, .015]])]
    elif name == 'submitted_tunnel':
        data = json.loads((REVISION / 'controlled_pinn/config/geometry.json').read_text(encoding='utf-8'))
        holes = [dict(kind='polygon', tag='cavity', vertices=data['tunnel_vertices_normalized']),
                 dict(kind='ellipse', tag='water', center=[.2, .2], axes=[.1, .05], angle=-np.pi / 4)]
    else:
        raise ValueError(name)
    return Domain(OUTER, holes, TAGS)


def pressure_modes(domain, E=1.333, nu=.3333):
    """Horizontal and vertical unit compression, with original bottom support."""
    conditions = {}
    for tag in TAGS:
        if tag == 'bottom':
            conditions[tag] = [Condition('displacement', [0, 1], constant([0, 0])),
                               Condition('traction', [1, 0], constant([0, 0]))]
        else:
            conditions[tag] = [
                Condition('traction', [1, 0], lambda xy, n: np.column_stack([-n[:, 0], np.zeros(len(xy))])),
                Condition('traction', [0, 1], lambda xy, n: np.column_stack([np.zeros(len(xy)), -n[:, 1]]))]
    for hole in domain.holes:
        conditions[hole['tag']] = [Condition('traction', [1, 0], constant([0, 0])),
                                   Condition('traction', [0, 1], constant([0, 0]))]
    return Problem(domain, E, nu, conditions,
                   [dict(xy=[0., -.5], direction=[1, 0], target=constant([0, 0]))], 2)


def affine_problem(domain, E=1.333, nu=.3333):
    gradient = np.array([[.13, -.07], [.11, -.19]])
    shift = np.array([.23, -.31])
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    stress_tensor = lam * np.trace(gradient) * np.eye(2) + mu * (gradient + gradient.T)
    conditions = {}
    for curve in domain.curves:
        if curve.hole:
            conditions[curve.tag] = [Condition('traction', [1, 0], lambda xy, n: (n @ stress_tensor.T)[:, :1]),
                                     Condition('traction', [0, 1], lambda xy, n: (n @ stress_tensor.T)[:, 1:2])]
        else:
            conditions[curve.tag] = [Condition('displacement', [1, 0], lambda xy, n: (xy @ gradient.T + shift)[:, :1]),
                                     Condition('displacement', [0, 1], lambda xy, n: (xy @ gradient.T + shift)[:, 1:2])]
    problem = Problem(domain, E, nu, conditions, [], 1)
    # coarse_features(1): constant, 2y, 2x; feature-major, then field component.
    coefficients = np.zeros((3, 5, 1))
    coefficients[0, :2, 0] = shift
    coefficients[0, 2:, 0] = stress_tensor[0, 0], stress_tensor[1, 1], stress_tensor[0, 1]
    coefficients[1, :2, 0] = gradient[:, 1] / 2
    coefficients[2, :2, 0] = gradient[:, 0] / 2
    return problem, coefficients.reshape(-1, 1)


def combine_loads(problem, alpha):
    """Specify one physical combination before adaptation; no reference solution."""
    import copy
    problem = copy.deepcopy(problem)
    alpha = np.asarray(alpha).reshape(-1, 1)
    for conditions in problem.conditions.values():
        for cond in conditions:
            fn = cond.target
            cond.target = lambda xy, n, fn=fn: fn(xy, n) @ alpha
    for gauge in problem.gauges:
        fn = gauge['target']
        gauge['target'] = lambda xy, n, fn=fn: fn(xy, n) @ alpha
    problem.n_loads = 1
    return problem
