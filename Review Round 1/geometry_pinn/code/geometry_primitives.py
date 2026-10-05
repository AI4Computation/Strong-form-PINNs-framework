"""Geometry and quadrature in nondimensional coordinates; no solver case branches."""
from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.ndimage import label


def polygon_area(vertices):
    v = np.asarray(vertices, float)
    return .5 * np.sum(v[:, 0] * np.roll(v[:, 1], -1) - v[:, 1] * np.roll(v[:, 0], -1))


def in_polygon(xy, vertices):
    x, y = np.asarray(xy).T
    result = np.zeros(len(x), bool)
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        if a[1] == b[1]:
            continue
        hit = ((a[1] > y) != (b[1] > y))
        xcross = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
        result ^= hit & (x < xcross)
    return result


@dataclass
class Curve:
    tag: str
    kind: str
    data: dict
    hole: bool = False

    def evaluate(self, t):
        t = np.asarray(t)
        if self.kind == 'line':
            a, b = np.asarray(self.data['a']), np.asarray(self.data['b'])
            tangent = np.broadcast_to(b - a, (len(t), 2))
            xy = a + t[:, None] * (b - a)
        elif self.kind == 'ellipse':
            angle = self.data.get('angle', 0.)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            axes = np.asarray(self.data['axes'])
            theta = 2 * np.pi * t
            xy = (np.column_stack([np.cos(theta), np.sin(theta)]) * axes) @ rotation.T
            xy += np.asarray(self.data['center'])
            tangent = 2 * np.pi * (np.column_stack([-np.sin(theta), np.cos(theta)]) * axes) @ rotation.T
        else:
            raise ValueError(self.kind)
        jac = np.linalg.norm(tangent, axis=1)
        tangent = tangent / jac[:, None]
        normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])
        if self.hole:
            normal = -normal
        return xy, normal, tangent, jac

    def quadrature(self, panels, order=3):
        nodes, weights = leggauss(order)
        t = ((np.arange(panels)[:, None] + (nodes + 1) / 2) / panels).ravel()
        xy, normal, tangent, jac = self.evaluate(t)
        w = np.tile(weights / (2 * panels), panels) * jac
        return dict(xy=xy, normal=normal, tangent=tangent, w=w, tag=self.tag)

    def cuts_box(self, lo, hi):
        if self.kind == 'line':
            a, b = np.asarray(self.data['a']), np.asarray(self.data['b'])
            d = b - a
            t0, t1 = 0., 1.
            for k in range(2):
                if abs(d[k]) < 1e-16:
                    # A straight boundary coinciding with a box face does not cut
                    # its interior and needs no cut-cell quadrature refinement.
                    if a[k] <= lo[k] + 1e-14 or a[k] >= hi[k] - 1e-14:
                        return False
                else:
                    s0, s1 = sorted(((lo[k] - a[k]) / d[k], (hi[k] - a[k]) / d[k]))
                    t0, t1 = max(t0, s0), min(t1, s1)
            return t1 > t0 + 1e-14
        c = np.asarray(self.data['center'])
        angle = self.data.get('angle', 0.)
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        metric = rot @ np.diag(1 / np.asarray(self.data['axes']) ** 2) @ rot.T
        corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]]) - c
        values = np.einsum('ni,ij,nj->n', corners, metric, corners).tolist()
        minimum = 0. if np.all(c >= lo) and np.all(c <= hi) else min(values)
        for fixed in range(2):
            free = 1 - fixed
            for edge in [lo[fixed], hi[fixed]]:
                z = np.zeros(2)
                z[fixed] = edge - c[fixed]
                z[free] = np.clip(-metric[free, fixed] * z[fixed] / metric[free, free], lo[free] - c[free], hi[free] - c[free])
                minimum = min(minimum, z @ metric @ z)
        return minimum <= 1 <= max(values)


class Domain:
    def __init__(self, outer, holes, outer_tags=None):
        self.outer = np.asarray(outer, float)
        if polygon_area(self.outer) <= 0:
            raise ValueError('Outer polygon must be counterclockwise.')
        self.lo, self.hi = self.outer.min(0), self.outer.max(0)
        self.holes = holes
        self.curves = []
        self.area = polygon_area(self.outer)
        tags = outer_tags or [f'outer_{i}' for i in range(len(outer))]
        for i, (a, b) in enumerate(zip(self.outer, np.roll(self.outer, -1, axis=0))):
            self.curves.append(Curve(tags[i], 'line', {'a': a.tolist(), 'b': b.tolist()}))
        self.corners = self.outer.tolist()
        for hole in holes:
            if hole['kind'] == 'ellipse':
                self.area -= np.pi * np.prod(hole['axes'])
                self.curves.append(Curve(hole['tag'], 'ellipse', hole, True))
            elif hole['kind'] == 'polygon':
                v = np.asarray(hole['vertices'], float)
                if np.allclose(v[0], v[-1]):
                    v = v[:-1]
                if polygon_area(v) < 0:
                    v = v[::-1]
                hole['vertices'] = v.tolist()
                self.area -= polygon_area(v)
                self.corners.extend(v.tolist())
                for a, b in zip(v, np.roll(v, -1, axis=0)):
                    self.curves.append(Curve(hole['tag'], 'line', {'a': a.tolist(), 'b': b.tolist()}, True))
            else:
                raise ValueError(hole['kind'])
        if self.area <= 0:
            raise ValueError('Nonpositive solid area.')

    def contains(self, xy):
        xy = np.asarray(xy)
        inside = in_polygon(xy, self.outer)
        for hole in self.holes:
            if hole['kind'] == 'polygon':
                inside &= ~in_polygon(xy, np.asarray(hole['vertices']))
            else:
                a = hole.get('angle', 0.)
                rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                local = (xy - hole['center']) @ rot
                inside &= np.sum((local / hole['axes']) ** 2, axis=1) > 1
        return inside

    def random_interior(self, n, rng):
        out, remaining = [], n
        for _ in range(10000):
            xy = rng.uniform(self.lo, self.hi, (max(2 * remaining, 32), 2))
            xy = xy[self.contains(xy)][:remaining]
            out.append(xy)
            remaining -= len(xy)
            if not remaining:
                return np.concatenate(out)
        raise RuntimeError('Could not sample solid domain.')

    def boundary(self, spacing=.025, order=3, rng=None):
        out = []
        for curve in self.curves:
            rough_length = float(curve.quadrature(16)['w'].sum())
            panels = max(1, int(np.ceil(rough_length / spacing)))
            if rng is None:
                out.append(curve.quadrature(panels, order))
            else:
                n = panels * order
                xy, normal, tangent, jac = curve.evaluate((np.arange(n) + rng.random(n)) / n)
                out.append(dict(xy=xy, normal=normal, tangent=tangent, w=jac / n, tag=curve.tag))
        return out

    def support_connected(self, feature):
        # Two-grid screen, deliberately not advertised as a topological proof.
        for size in [25, 41]:
            grid = np.linspace(-.98, .98, size)
            xi = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
            xy = xi @ feature.H.T + feature.center
            solid = (np.sum(xi ** 2, axis=1) < .96) & self.contains(xy)
            _, components = label(solid.reshape(size, size))
            if components != 1:
                return False
        return True


class Quadrature:
    def __init__(self, domain, base_depth=3, cut_depth=7, order=3, max_depth=9):
        self.domain, self.base_depth, self.cut_depth = domain, base_depth, cut_depth
        self.order, self.max_depth = order, max_depth
        self.cells = [(domain.lo.copy(), domain.hi.copy(), 0)]
        self.refine([])

    def refine(self, features):
        pending, leaves = self.cells, []
        while pending:
            lo, hi, depth = pending.pop()
            centre = (lo + hi) / 2
            cut = any(curve.cuts_box(lo, hi) for curve in self.domain.curves)
            if not cut and not self.domain.contains(centre[None])[0]:
                continue
            split = depth < self.base_depth or (cut and depth < self.cut_depth)
            for f in features:
                if f.kind not in ('local', 'airy'):
                    continue
                # Conservative AABB intersection; sufficient resolution is checked independently.
                radius = np.linalg.norm(f.H, axis=1)
                overlaps = np.all(hi >= f.center - radius) and np.all(lo <= f.center + radius)
                if overlaps and np.max(hi - lo) > min(f.axes) / 2:
                    split = True
                    break
            if split and depth < self.max_depth:
                mid = (lo + hi) / 2
                for ix in range(2):
                    for iy in range(2):
                        a = np.array([lo[0] if ix == 0 else mid[0], lo[1] if iy == 0 else mid[1]])
                        b = np.array([mid[0] if ix == 0 else hi[0], mid[1] if iy == 0 else hi[1]])
                        pending.append((a, b, depth + 1))
            else:
                leaves.append((lo, hi, depth))
        self.cells = sorted(leaves, key=lambda c: (c[0][0], c[0][1], c[2]))
        return self.points()

    def points(self, order=None):
        nodes, w1 = leggauss(order or self.order)
        xi = np.stack(np.meshgrid(nodes, nodes), axis=-1).reshape(-1, 2)
        wi = np.outer(w1, w1).ravel()
        xy, weights = [], []
        for lo, hi, _ in self.cells:
            p = (lo + hi) / 2 + xi * (hi - lo) / 2
            inside = self.domain.contains(p)
            xy.append(p[inside])
            weights.append(wi[inside] * np.prod(hi - lo) / 4)
        xy, weights = np.concatenate(xy), np.concatenate(weights)
        return {'xy': xy, 'w': weights, 'area': float(weights.sum()), 'cells': len(self.cells)}


def nondimensional_geometry(outer, holes):
    """Common length and origin derived from outer geometry, independent of hole type."""
    import copy
    outer = np.asarray(outer, float)
    origin = (outer.min(0) + outer.max(0)) / 2
    length = float(np.ptp(outer, axis=0).max())
    result = copy.deepcopy(holes)
    for h in result:
        if h['kind'] == 'ellipse':
            h['center'] = ((np.asarray(h['center']) - origin) / length).tolist()
            h['axes'] = (np.asarray(h['axes']) / length).tolist()
        else:
            h['vertices'] = ((np.asarray(h['vertices']) - origin) / length).tolist()
    return (outer - origin) / length, result, origin, length
