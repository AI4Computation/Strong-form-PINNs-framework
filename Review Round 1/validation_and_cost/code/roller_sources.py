"""Elastic image-source features for a straight frictionless roller boundary.

This is a classical reflection construction, not a new fundamental solution.
Geometry and boundary conditions determine the plane and gauge; no hole-specific
formula or reference field is used. Both real and image poles must be outside rock.
"""
import numpy as np
from elastic_sources import fields
from mechanics import directions


def roller_plane(problem):
    if len(problem.gauges) != 1:
        raise ValueError('This prototype requires one tangential point gauge.')
    gauge = problem.gauges[0]
    for curve in problem.domain.curves:
        if curve.hole or curve.kind != 'line':
            continue
        q = curve.quadrature(7, 3)
        normal, tangent = q['normal'][0], q['tangent'][0]
        conditions = problem.conditions[curve.tag]
        correct = set()
        for condition in conditions:
            direction = directions(condition, q)
            alignment = np.abs(direction @ (normal if condition.field == 'displacement' else tangent))
            if np.max(np.abs(alignment - 1)) < 1e-12 and np.max(np.abs(condition.target(q['xy'], q['normal']))) < 1e-14:
                correct.add(condition.field)
        origin = q['xy'][0]
        location = np.asarray(gauge['xy'])
        if correct != {'displacement', 'traction'}:
            continue
        if abs((location - origin) @ normal) > 1e-12 or abs(abs(np.asarray(gauge['direction']) @ tangent) - 1) > 1e-12:
            continue
        if np.max(np.abs(gauge['target'](location[None], normal[None]))) > 1e-14:
            continue
        return dict(origin=origin.tolist(), normal=normal.tolist(), tangent=tangent.tolist(),
                    gauge=location.tolist(), tag=curve.tag)
    raise ValueError('No compatible zero-normal-displacement/zero-shear straight support was found.')


def images(sources, plane):
    sources = np.asarray(sources).reshape(-1, 2)
    n, origin = np.asarray(plane['normal']), np.asarray(plane['origin'])
    return sources - 2 * ((sources - origin) @ n)[:, None] * n


def evaluator(problem, plane=None):
    plane = roller_plane(problem) if plane is None else plane
    n, t = np.asarray(plane['normal']), np.asarray(plane['tangent'])
    origin, gauge = np.asarray(plane['origin']), np.asarray(plane['gauge'])
    reflection = np.eye(2) - 2 * np.outer(n, n)
    def evaluate(xy, sources, E, nu, affine=True):
        xy, sources = np.asarray(xy), np.asarray(sources).reshape(-1, 2)
        blocks = []
        if affine:
            mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
            block = np.zeros((len(xy), 5, 2))
            block[:, :2, 0] = ((xy - gauge) @ t)[:, None] * t
            block[:, :2, 1] = ((xy - origin) @ n)[:, None] * n
            for j, direction in enumerate([t, n]):
                stress = lam * np.eye(2) + 2 * mu * np.outer(direction, direction)
                block[:, 2:, j] = [stress[0, 0], stress[1, 1], stress[0, 1]]
            blocks.append(block)
        if len(sources):
            mirror = images(sources, plane)
            base = fields(xy, sources, E, nu, affine=False).reshape(len(xy), 5, -1, 2)
            reflected = fields(xy, mirror, E, nu, affine=False).reshape(len(xy), 5, -1, 2) @ reflection
            combined = (base + reflected).reshape(len(xy), 5, -1)
            at_gauge = fields(gauge[None], sources, E, nu, affine=False).reshape(1, 5, -1, 2)
            at_gauge += fields(gauge[None], mirror, E, nu, affine=False).reshape(1, 5, -1, 2) @ reflection
            shift = np.einsum('i,imj->mj', t, at_gauge[0, :2]).reshape(-1)
            combined[:, :2] -= t[None, :, None] * shift[None, None, :]
            blocks.append(combined)
        return np.concatenate(blocks, axis=2) if blocks else np.empty((len(xy), 5, 0))
    return evaluate


def verify():
    from problems import geometry, pressure_modes
    output = {}
    for name in ['circle', 'square', 'slender', 'submitted_tunnel']:
        p = pressure_modes(geometry(name))
        plane = roller_plane(p)
        evaluate = evaluator(p, plane)
        s = np.array([[0., 0.], [.7, .2], [-.8, .1]])
        assert not p.domain.contains(s).any() and not p.domain.contains(images(s, plane)).any()
        support = p.domain.curves[0].quadrature(41, 5)
        f = evaluate(support['xy'], s, p.E, p.nu)
        n, t = np.asarray(plane['normal']), np.asarray(plane['tangent'])
        displacement = np.einsum('nib,i->nb', f[:, :2], n)
        traction = np.stack([f[:, 2] * n[0] + f[:, 4] * n[1], f[:, 4] * n[0] + f[:, 3] * n[1]], 1)
        shear = np.einsum('nib,i->nb', traction, t)
        gauge = evaluate(np.asarray(plane['gauge'])[None], s, p.E, p.nu)
        error = max(float(abs(displacement).max()), float(abs(shear).max()), float(abs(gauge[:, :2]).max()))
        assert error < 1e-12
        # Finite-difference stresses from displacement include reflection and shift.
        xy = p.domain.random_interior(41, np.random.default_rng(20260927))
        epsilon = 1e-6
        derivatives = [(evaluate(xy + np.eye(2)[i] * epsilon, s, p.E, p.nu)[:, :2]
                        - evaluate(xy - np.eye(2)[i] * epsilon, s, p.E, p.nu)[:, :2]) / (2 * epsilon) for i in range(2)]
        dx, dy = derivatives
        lam, mu = p.lame
        stress = np.stack([(lam + 2 * mu) * dx[:, 0] + lam * dy[:, 1],
                           lam * dx[:, 0] + (lam + 2 * mu) * dy[:, 1],
                           mu * (dy[:, 0] + dx[:, 1])], 1)
        reference = evaluate(xy, s, p.E, p.nu)[:, 2:]
        constitutive = float(np.max(np.abs(stress - reference)))
        assert constitutive < 1e-7
        output[name] = dict(support_and_gauge_max_error=error, constitutive_finite_difference_max_error=constitutive,
                            plane=plane, all_real_and_image_poles_outside=True)
    return output
