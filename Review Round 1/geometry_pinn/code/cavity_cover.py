"""Geometry-only compact cover; no case-name switches, FEM, or file writes."""
import copy
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import label
from geometry_primitives import Domain, polygon_area


def normalized_domain(data):
    outer = np.asarray(data['outer'], float)
    origin = (outer.min(0) + outer.max(0)) / 2
    length = np.ptp(outer, axis=0).max()
    holes = copy.deepcopy(data['holes'])
    for h in holes:
        if h['kind'] == 'polygon':
            h['vertices'] = ((np.asarray(h['vertices']) - origin) / length).tolist()
        else:
            h['center'] = ((np.asarray(h['center']) - origin) / length).tolist()
            h['axes'] = (np.asarray(h['axes']) / length).tolist()
    return Domain((outer - origin) / length, holes, data.get('outer_tags')), origin, length


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def ray_hits(domain, x, direction):
    """Exact intersections with polygon edges / analytic ellipses, skipping t=0."""
    distance = np.full(len(x), np.inf)
    owner = np.full(len(x), -1, dtype=int)
    for ci, curve in enumerate(domain.curves):
        if curve.kind == 'line':
            a, b = np.asarray(curve.data['a']), np.asarray(curve.data['b'])
            edge = b - a
            denominator = cross(direction, edge)
            ok = np.abs(denominator) > 1e-14
            t = np.full(len(x), np.inf)
            v = np.full(len(x), np.inf)
            t[ok] = cross(a - x[ok], edge) / denominator[ok]
            v[ok] = cross(a - x[ok], direction[ok]) / denominator[ok]
            t[(v < -1e-10) | (v > 1 + 1e-10) | (t < 1e-9)] = np.inf
        else:
            angle = curve.data.get('angle', 0.)
            rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            axes = np.asarray(curve.data['axes'])
            q = (x - curve.data['center']) @ rot / axes
            d = direction @ rot / axes
            aa = np.sum(d*d, axis=1)
            bb = 2*np.sum(q*d, axis=1)
            cc = np.sum(q*q, axis=1) - 1
            discriminant = bb*bb - 4*aa*cc
            root = np.sqrt(np.maximum(discriminant, 0))
            roots = np.column_stack([(-bb-root)/(2*aa), (-bb+root)/(2*aa)])
            roots[(roots < 1e-9) | (discriminant[:, None] < -1e-10)] = np.inf
            t = roots.min(axis=1)
        better = t < distance
        distance[better], owner[better] = t[better], ci
    return distance, owner


def boundary_samples(domain, spacing, holes_only=True):
    records = []
    for ci, curve in enumerate(domain.curves):
        if holes_only and not curve.hole:
            continue
        length = float(curve.quadrature(32, 5)['w'].sum())
        count = max(4 if curve.kind == 'ellipse' else 1, int(np.ceil(length / spacing)))
        t = (np.arange(count) + .5) / count
        x, normal, tangent, jac = curve.evaluate(t)
        if curve.kind == 'ellipse':
            ax, ay = curve.data['axes']
            curvature_radius = ((ax*np.sin(2*np.pi*t))**2 + (ay*np.cos(2*np.pi*t))**2)**1.5/(ax*ay)
        else:
            curvature_radius = np.full(count, np.inf)
        records.append(dict(x=x, normal=normal, tangent=tangent,
                            radius=curvature_radius, curve=np.full(count, ci)))
    return {key: np.concatenate([r[key] for r in records]) for key in records[0]}


def sharp_corners(domain, degrees):
    out = []
    for h in domain.holes:
        if h['kind'] != 'polygon':
            continue
        v = np.asarray(h['vertices'])
        prev = v - np.roll(v, 1, axis=0)
        nxt = np.roll(v, -1, axis=0) - v
        cosine = np.sum(prev*nxt, axis=1) / (np.linalg.norm(prev, axis=1)*np.linalg.norm(nxt, axis=1))
        turn = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        out.extend(v[turn >= degrees].tolist())
    return np.asarray(out).reshape(-1, 2)


def construct(domain, protocol):
    samples = boundary_samples(domain, protocol['boundary_spacing_over_outer_length'])
    x, normal = samples['x'], samples['normal']
    void_chord, void_hit = ray_hits(domain, x, normal)
    solid_gap, solid_hit = ray_hits(domain, x, -normal)
    minimum = 2.**(-protocol['maximum_depth'])
    target = np.clip(np.minimum.reduce([.5*void_chord, .5*solid_gap,
                                       .5*samples['radius'], np.full(len(x), .125)]), minimum, .125)
    corners = sharp_corners(domain, protocol['corner_turn_degrees'])
    feature_x = np.vstack([x, corners])
    feature_h = np.r_[target, np.full(len(corners), minimum)]
    tree = cKDTree(feature_x)
    pending = [(domain.lo.copy(), domain.hi.copy(), 0)]
    leaves = []
    while pending:
        lo, hi, depth = pending.pop()
        centre = (lo+hi)/2
        cuts = any(c.cuts_box(lo, hi) for c in domain.curves)
        if not cuts and not domain.contains(centre[None])[0]:
            continue
        width = float(np.max(hi-lo))
        neighbours = tree.query_ball_point(centre, 2*width)
        refine = depth < protocol['base_depth'] or (neighbours and np.min(feature_h[neighbours]) < width-1e-12)
        if refine and depth < protocol['maximum_depth']:
            mid = (lo+hi)/2
            for i in range(2):
                for j in range(2):
                    lower = np.array([lo[0] if i == 0 else mid[0], lo[1] if j == 0 else mid[1]])
                    upper = np.array([mid[0] if i == 0 else hi[0], mid[1] if j == 0 else hi[1]])
                    pending.append((lower, upper, depth+1))
        else:
            leaves.append((lo, hi, depth))
    leaves.sort(key=lambda c: tuple(c[0]))
    centres = np.array([(lo+hi)/2 for lo,hi,_ in leaves])
    halfwidths = np.array([(hi-lo)*protocol['overlap_ratio']/2 for lo,hi,_ in leaves])
    return dict(centres=centres, halfwidths=halfwidths, depths=np.array([d for _,_,d in leaves]),
                boundary=x, normal=normal, tangent=samples['tangent'], target=target,
                void_chord=void_chord, solid_gap=solid_gap, void_hit=void_hit,
                solid_hit=solid_hit, corners=corners)


def windows(x, centres, halfwidths, derivatives=False):
    z = (x[:, None, :] - centres[None, :, :]) / halfwidths[None, :, :]
    one = np.maximum(1-z*z, 0)
    base = one**3
    raw = np.prod(base, axis=2)
    total = raw.sum(axis=1)
    if np.any(total <= 0):
        raise ValueError('Uncovered location; no global floor is allowed to mask the gap.')
    weight = raw / total[:, None]
    if not derivatives:
        return weight, total, np.count_nonzero(raw, axis=1)
    db = -6*z*one**2/halfwidths[None, :, :]
    draw = np.stack([db[:, :, 0]*base[:, :, 1], base[:, :, 0]*db[:, :, 1]], axis=2)
    grad = (draw - weight[:, :, None]*draw.sum(axis=1)[:, None, :])/total[:, None, None]
    return weight, grad


def support_components(domain, centres, halfwidths, sizes=(65,97)):
    counts = []
    for size in sizes:
        v = np.linspace(-.999, .999, size)
        grid = np.stack(np.meshgrid(v,v), axis=-1).reshape(-1,2)
        counts.append([int(label(domain.contains(grid*h+c).reshape(size,size))[1])
                       for c,h in zip(centres,halfwidths)])
    return np.array(counts)
