"""Bounded, truth-blind adaptive classical MFS pilot; accuracy only.

Kelvin kernels and greedy source selection are established techniques. This
prototype tests suitability as a baseline/complement; it does not assert novelty.
"""
import os
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from scipy.linalg import svdvals
from elastic_sources import fields, assemble as assemble_fields, boundary_rows, candidates
from linear import LinearSpace
from problems import ROOT, geometry, pressure_modes, combine_loads
from mechanics import Condition


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_problem(settings):
    E, nu = settings.get('E', 1.333), settings.get('nu', .3333)
    problem = combine_loads(pressure_modes(geometry(settings['geometry']), E, nu), settings['loads'])
    for tag, pressure in settings.get('cavity_pressures', {}).items():
        problem.conditions[tag] = [Condition('traction', [1, 0], lambda xy, n, p=pressure: p * n[:, :1]),
                                   Condition('traction', [0, 1], lambda xy, n, p=pressure: p * n[:, 1:2])]
    if settings.get('support') == 'roller':
        from roller_sources import roller_plane, evaluator
        problem.roller_plane = roller_plane(problem)
        problem.elastic_field_evaluator = evaluator(problem, problem.roller_plane)
    return problem


def assemble(problem, sources, groups, affine=True):
    return assemble_fields(problem, sources, groups, affine,
                           field_evaluator=getattr(problem, 'elastic_field_evaluator', None))


def prospective_condition(space, proposal):
    n, k = len(space.R), len(proposal['R'])
    trial = np.zeros((n + k, n + k))
    trial[:n, :n] = space.R
    trial[:n, n:] = proposal['cross']
    trial[n:, n:] = proposal['R']
    s = svdvals(trial, check_finite=False)
    return float(s[0] / s[-1])


def residual_report(problem, sources, coeff, groups):
    A, b = assemble(problem, sources, groups)
    residual = b - A @ coeff
    total, start, components = float(np.sum(residual ** 2)), 0, {}
    for g in groups:
        stop = start + len(g['xy'])
        components[g['name']] = components.get(g['name'], 0.) + float(np.sum(residual[start:stop] ** 2))
        start = stop
    return dict(total=total, relative_rms=float(np.sqrt(total / np.sum(b ** 2))), components=components), residual


def solve(folder, settings):
    folder.mkdir(parents=True, exist_ok=False)
    verification = ROOT / 'checks/elastic_sources_verification.json'
    check = json.loads(verification.read_text(encoding='utf-8'))
    assert check['passed']
    for name, digest in check['source_sha256'].items():
        assert sha(Path(__file__).with_name(name)) == digest
    codefiles = ['run_elastic_sources.py', 'elastic_sources.py', 'linear.py', 'geometry.py', 'problems.py', 'mechanics.py', 'roller_sources.py', 'evaluate_elastic_gpu.py']
    if settings.get('support') == 'roller':
        roller_check = ROOT / 'checks/roller_sources_verification.json'
        checked = json.loads(roller_check.read_text(encoding='utf-8'))
        assert checked['passed'] and checked['source_sha256'] == sha(Path(__file__).with_name('roller_sources.py'))
    manifest = dict(status='running', source_sha256={n:sha(Path(__file__).with_name(n)) for n in codefiles},
                    verification_sha256=sha(verification), timing_valid_for_comparison=False)
    with zipfile.ZipFile(folder / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for n in codefiles:
            z.write(Path(__file__).with_name(n), n)
    manifest['source_archive_sha256'] = sha(folder / 'source.zip')
    (folder / 'protocol.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    problem = make_problem(settings)
    assert problem.body is None
    rng = np.random.default_rng(settings['seed'])
    check_groups = boundary_rows(problem, .003, order=3, rng=np.random.default_rng(settings['seed'] + 100000))
    spacing, order, sources, source_records = .01, 5, [], []
    fit = boundary_rows(problem, spacing, order)
    A, b = assemble(problem, sources, fit)
    space = LinearSpace(A, b, settings['rank_tolerance'])
    history, pools, stop_reason = [], [], 'source_budget'
    sampling_verified = False
    while True:
        # Audit integration before checking termination or choosing another pool.
        for _ in range(8):
            coeff = space.coefficients()
            fine = boundary_rows(problem, spacing / 2, order=7)
            probe, _ = residual_report(problem, sources, coeff, fine)
            coarse_total = float(np.sum(space.residual ** 2))
            discrepancy = abs(coarse_total - probe['total']) / max(probe['total'], 1e-24)
            if discrepancy <= settings['quadrature_tolerance']:
                sampling_verified = True
                break
            sampling_verified = False
            if sum(len(g['xy']) for g in fine) > settings['max_boundary_rows']:
                stop_reason = 'quadrature_budget'
                break
            fit, spacing, order = fine, spacing / 2, 7
            A, b = assemble(problem, sources, fit)
            space = LinearSpace(A, b, settings['rank_tolerance'])
        report, check_residual = residual_report(problem, sources, space.coefficients(), check_groups)
        record = dict(sources=len(sources), coefficients=space.R.shape[0], fit_objective=float(np.sum(space.residual ** 2)),
                      check=report, finer_quadrature=probe, quadrature_relative_discrepancy=float(discrepancy),
                      fit_rows=len(space.rhs), spacing=spacing, order=order, sampling_verified=sampling_verified,
                      qr_orthogonality_error=space.orthogonality_error())
        history.append(record)
        (folder / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(json.dumps(dict(run=folder.name, sources=len(sources), boundary_relative_rms=report['relative_rms'],
                              quadrature_difference=discrepancy)), flush=True)
        if not sampling_verified:
            break
        if report['relative_rms'] <= settings['relative_rms_tolerance']:
            stop_reason = 'independent_boundary_tolerance'
            break
        if len(sources) >= settings['source_budget']:
            break
        pool = candidates(problem.domain, check_groups, check_residual, rng, settings['pool'])
        proposals, pool_log = [], []
        for candidate in pool:
            point = candidate['xy']
            image_inside = False
            if hasattr(problem, 'roller_plane'):
                from roller_sources import images
                image_inside = bool(problem.domain.contains(images([point], problem.roller_plane)).any())
            if image_inside or (sources and np.min(np.linalg.norm(np.asarray(sources) - point, axis=1)) < 1e-6):
                proposals.append(None)
            else:
                block, _ = assemble(problem, [point], fit, affine=False)
                proposals.append(space.proposal(block))
            pool_log.append(dict(**candidate, initial_gain=None if proposals[-1] is None else proposals[-1]['gain']))
        accepted = []
        batch_size = min(settings['batch'], settings['source_budget'] - len(sources))
        while len(accepted) < batch_size:
            eligible = [i for i, p in enumerate(proposals) if p is not None]
            if not eligible:
                break
            best = max(eligible, key=lambda i: proposals[i]['gain'])
            p = proposals[best]
            if p['gain'] <= 1e-20:
                break
            if settings.get('global_rank_guard', False):
                condition = prospective_condition(space, p)
                if condition >= 1 / settings['rank_tolerance']:
                    pool_log[best]['global_rank_rejected_condition'] = condition
                    proposals[best] = None
                    continue
            space.append(p)
            sources.append(pool[best]['xy']); source_records.append(pool[best])
            accepted.append(dict(index=best, gain=p['gain']))
            proposals[best] = None
            for i in eligible:
                if proposals[i] is not None:
                    proposals[i] = space.refresh(proposals[i], p['Q'])
        pools.append(dict(pool=pool_log, accepted=accepted))
        if not accepted:
            stop_reason = 'no_independent_candidate'
            break
    coeff = space.coefficients()
    singular = svdvals(space.R, check_finite=False)
    final_condition = float(singular[0] / singular[-1])
    if settings.get('global_rank_guard', False):
        assert final_condition < 1 / settings['rank_tolerance']
    audit_groups = boundary_rows(problem, .0015, order=5, rng=np.random.default_rng(settings['seed'] + 200000))
    audit, _ = residual_report(problem, sources, coeff, audit_groups)
    assert not problem.domain.contains(np.asarray(sources).reshape(-1, 2)).any()
    assert np.isfinite(coeff).all()
    output = dict(settings=settings, sources=source_records, history=history, final_audit=audit,
                  coefficients=len(coeff), sampling_verified=sampling_verified, stop_reason=stop_reason,
                  column_normalized_condition=final_condition, global_rank_passed=final_condition < 1 / settings['rank_tolerance'],
                  exact_roller_plane=getattr(problem, 'roller_plane', None),
                  source_selection_uses_fem=False, timing_valid_for_comparison=False,
                  interpretation='Established Kelvin/MFS/greedy-source prototype; not a novelty claim.')
    np.savez_compressed(folder / 'coefficients.npz', coefficients=coeff, sources=np.asarray(sources).reshape(-1, 2))
    (folder / 'candidate_pools.json').write_text(json.dumps(pools, indent=2), encoding='utf-8')
    (folder / 'result.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    manifest.update(status='complete', result_sha256=sha(folder / 'result.json'),
                    coefficients_sha256=sha(folder / 'coefficients.npz'))
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return output


def full_evaluation(folder):
    result_path = folder / 'result.json'
    result = json.loads(result_path.read_text(encoding='utf-8'))
    settings = result['settings']
    case = settings.get('case', 'C1')
    reference_path = ROOT.parent / 'controlled_pinn/fem/references' / (case + '.npz')
    weight_file = 'tr3_circle_sq0p0025_UnitL.npz' if case == 'C1' else 'tr3_tunnel_tq0p125_Load.npz'
    weights_path = ROOT.parent / 'controlled_pinn/fem' / weight_file
    data = np.load(folder / 'coefficients.npz')
    reference, weights = np.load(reference_path), np.load(weights_path)
    assert np.array_equal(weights['xy_s'], reference['xy_s'])
    problem = make_problem(settings)
    length_scale = 1. if case == 'C1' else 75.
    field_evaluator = getattr(problem, 'elastic_field_evaluator', fields)
    accelerator_check = None
    accelerator = None
    if settings.get('evaluation_device') == 'cuda':
        from evaluate_elastic_gpu import GPUFields
        accelerator = GPUFields(problem, data['sources'], data['coefficients'])
        points = problem.domain.random_interior(113, np.random.default_rng(20260927))
        expected = (field_evaluator(points, data['sources'], problem.E, problem.nu) @ data['coefficients'])[:, :, 0]
        difference = float(np.max(np.abs(accelerator(points) - expected)))
        relative = difference / max(float(np.max(np.abs(expected))), 1.)
        assert relative < 1e-6, relative
        accelerator_check = dict(cpu64_gpu64_max_abs_difference=difference, scaled_max_difference=relative)
    metrics = {}
    for field, components in [('u', [0, 1]), ('s', [2, 3, 4])]:
        xy, truth = reference['xy_' + field], reference[field]
        num = den = area_num = area_den = near_num = near_den = 0.
        for start in range(0, len(xy), 2048):
            stop = start + 2048
            normalized = xy[start:stop] / length_scale
            value = ((field_evaluator(normalized, data['sources'], problem.E, problem.nu) @ data['coefficients'])[:, components, 0]
                     if accelerator is None else accelerator(normalized)[:, components])
            value *= (1. if case == 'C1' else (.01875 if field == 'u' else 2.5))
            difference = np.sum((value - truth[start:stop]) ** 2, axis=1)
            norm = np.sum(truth[start:stop] ** 2, axis=1)
            num += difference.sum(); den += norm.sum()
            if field == 's':
                w = weights['volume'][start:stop]
                area_num += np.sum(w * difference); area_den += np.sum(w * norm)
                if case == 'C1':
                    near = np.linalg.norm(xy[start:stop], axis=1) <= .2
                    near_num += difference[near].sum(); near_den += norm[near].sum()
        metrics[field + '_full_pct'] = float(100 * np.sqrt(num / den))
        if field == 's':
            metrics['s_area_weighted_full_pct'] = float(100 * np.sqrt(area_num / area_den))
            if case == 'C1':
                metrics['s_near_full_pct'] = float(100 * np.sqrt(near_num / near_den))
        print(json.dumps(dict(run=folder.name, evaluated=field, metrics=metrics)), flush=True)
    output = dict(metrics=metrics, sources=len(data['sources']), points=dict(u=len(reference['u']), s=len(reference['s'])),
                  selection_uses_fem=False, scope='Final endpoint, full existing authoritative ' + case + ' FEM grid.',
                  timing_valid_for_comparison=False,
                  accelerator_verification=accelerator_check,
                  source_sha256=dict(result=sha(result_path), coefficients=sha(folder / 'coefficients.npz'),
                                     reference=sha(reference_path), weights=sha(weights_path), evaluator=sha(__file__),
                                     gpu_evaluator=sha(Path(__file__).with_name('evaluate_elastic_gpu.py')) if accelerator else None))
    (folder / 'full_evaluation.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    return output


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20260925)
    parser.add_argument('--budget', type=int, default=96)
    parser.add_argument('--case', choices=['C1', 'T1'], default='C1')
    parser.add_argument('--global-rank-guard', action='store_true')
    parser.add_argument('--support', choices=['soft', 'roller'], default='soft')
    parser.add_argument('--evaluation-device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    settings = dict(geometry='circle', loads=[1., 5.], seed=args.seed, source_budget=args.budget,
                    pool=24, batch=4, rank_tolerance=1e-10, quadrature_tolerance=.05,
                    max_boundary_rows=100000, relative_rms_tolerance=1e-4,
                    permitted_use='Accuracy and suitability screening; no method-timing or novelty claim.')
    settings.update(global_rank_guard=args.global_rank_guard, support=args.support, evaluation_device=args.evaluation_device)
    if args.case == 'T1':
        settings.update(case='T1', geometry='submitted_tunnel', E=1., nu=.26, loads=[4., 4.],
                        cavity_pressures={'water': -1.}, reference_length_scale=75.,
                        reference_u_scale=.01875, reference_stress_scale=2.5,
                        geometry_sha256=sha(ROOT.parent / 'controlled_pinn/config/geometry.json'),
                        geometry_note='Exact submitted polygon facets and rotated ellipse; no smoothing or hand placed poles.')
    suffix = ('_roller' if args.support == 'roller' else '') + ('_globalrank' if args.global_rank_guard else '')
    folder = ROOT / 'elastic_source_runs' / f'{args.case}_adaptive_kelvin{suffix}_n{args.budget}_seed{args.seed}'
    with threadpool_limits(limits=2):
        if not (folder / 'result.json').exists():
            solve(folder, settings)
        if not (folder / 'full_evaluation.json').exists():
            full_evaluation(folder)
