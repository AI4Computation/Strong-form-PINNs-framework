"""Human-readable complete development outcome, without changing frozen evidence."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'results/R2_P2A'
def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    a=read(B/'evaluation/analysis.json');p=read(ROOT/'protocols/R2_P2A_development.json')
    rows=['# R2 P2A: controlled strong-form development screen','',
          'This is a single-seed, fixed-budget development result. All 15 terminal states were frozen before FEM fields were opened. It does not establish confirmed superiority, novelty, or fair time-to-accuracy. No first-round results or submission files were changed.','',
          '## Design and primary outcomes','',
          'Five representations used the same 6,000 interior points, 500 points per outer edge, and 1,000 cavity-wall points. The interior objective combines 90% uniform-point mean and 10% common support-probe mean. The latter is an auxiliary constraint, not unbiased area quadrature. All hidden layers remain trainable. Each run reached 400 accepted L-BFGS steps; actual closure counts are reported below. Parameters range from 110,685 to 110,705.','',
          '| Case | Representation | U, area L2 (%) | S, area L2 (%) | Wall U L2 (%) | Closures |',
          '|---|---|---:|---:|---:|---:|']
    for case in p['cases']:
        for method in p['methods']:
            m=a['cases'][case][method]['metrics'];r=read(B/(case+'_'+method)/'result.json')
            rows.append(f"| {case} | {method} | {m['u_area']['relative_l2_percent']:.6f} | {m['s_area']['relative_l2_percent']:.6f} | {m['wall_u']['relative_l2_percent']:.6f} | {r['closure_evaluations']} |")
    rows+=['','## Registered decision','',
           f"Independent-gain gate: **{a['screening']['independent_gain_gate']}**. Half-bandwidth Fourier gate: **{a['screening']['half_fourier_gate']}**. Advance to confirmation/expanded geometries under this protocol: **{a['screening']['advance']}**.",
           f"The geometric mean of geometry-local / uniform-local U and S error ratios is {a['screening']['geometric_mean_uniform_ratio']:.6f}. The complete individual ratios and wall guard are in `evaluation/analysis.json`. A failed screen stops expansion of this exact candidate; it does not disprove local PINNs as a family or settle performance at longer budgets.",
           '', '## Spatial and engineering evidence','',
           '| Case | Representation | Near-wall S (%) | Far-wall S (%) | Corner S (%) | S vector error p99 | Area with lower S error than half Fourier |',
           '|---|---|---:|---:|---:|---:|---:|']
    for case in p['cases']:
        for method in p['methods']:
            r=a['cases'][case][method];m=r['metrics'];corner=m.get('s_corner',{}).get('relative_l2_percent')
            frac=r['area_fraction_lower_vector_error'].get('fourier_half',{}).get('s')
            cv='—' if corner is None else f'{corner:.6f}';fv='—' if frac is None else f'{100*frac:.3f}%'
            rows.append(f"| {case} | {method} | {m['s_near_wall']['relative_l2_percent']:.6f} | {m['s_far_wall']['relative_l2_percent']:.6f} | {cv} | {m['s_area']['vector_absolute_error_quantiles'][-1]:.6f} | {fv} |")
    rows+=['','The near-wall region is within 0.05 normalized length; polygon corner neighborhoods have radius 0.02. All-domain errors include corners. Error quantiles are weighted vector absolute errors, not percentages or pointwise relative errors. Every pairwise comparison, U distribution and weighted MAE is retained in the JSON.',
           '', '| Case | Representation | Vertical convergence absolute error | Horizontal convergence absolute error |',
           '|---|---|---:|---:|']
    for case in p['cases']:
        for method in p['methods']:
            e=a['cases'][case][method]['metrics']['engineering']['convergence_absolute_error']
            rows.append(f'| {case} | {method} | {e[0]:.8f} | {e[1]:.8f} |')
    rows+=['','Convergence is crown-minus-invert vertical displacement and right-minus-left horizontal displacement. Signed reference/predicted values and all four extremum displacement vectors are included in each case JSON. Absolute errors avoid dividing by a near-zero individual engineering quantity.',
           '', '## Reference and computational scope','',
           '| Case | Common integration points | Stress reconstruction discrepancy (%) | Reference wall-offset sensitivity |',
           '|---|---:|---:|---:|']
    for case,r in a['references'].items():
        rows.append(f"| {case} | {r['integration_points']} | {r['stress_reconstruction_percent']:.8f} | {r['wall_offset_sensitivity_max']:.3e} |")
    rows+=['','Displacements at stress integration points were reconstructed with the finite-element shape functions; exported integration-point stresses remain the stress reference. The original nodal displacement and unweighted integration-point metrics are retained separately. Exact input coordinates are used for the slender mesh. Wall reference queries use an offset of 1e-8 into solid; doubling that offset is checked. These checks establish interpolation consistency, not rigorous true-solution error bounds.',
           '', 'All per-run `result.json` files contain actual parameter counts, closure counts, peak allocated/reserved GPU bytes, preparation time and optimization time including checkpoint I/O. These are development telemetry. Formal timing and fair resource-to-accuracy claims require the separately confirmed idle-window protocol; no such comparison was run here.',
           '', '## Source-data map','',
           f'- [Frozen protocol]({(ROOT/"protocols/R2_P2A_development.json").as_posix()})',
           f'- [Complete results and decision]({(B/"evaluation/analysis.json").as_posix()})',
           f'- [Batch manifest and terminal hashes]({(B/"manifest.json").as_posix()})',
           f'- [Numerical preflight]({(B/"preflight.json").as_posix()})',
           '- `evaluation/{case}_reference.npz`: all common coordinates, area weights, U/S reference fields, wall/extremum reference displacements and spatial masks.',
           '- `evaluation/{case}_{method}_predictions.npz`: every predicted integration-point field, original nodal displacement, wall displacement and extremum displacement.',
           '- `evaluation/{case}_metrics.json`: point/area metrics, distributions, pairwise improved-area fractions and engineering quantities.',
           '- `{case}_{method}/terminal.pt`, `step_*.pt`, `trace.json`, `result.json`: frozen weights, reached checkpoints, accepted-step history and development resource telemetry.',
           '- `{geometry}_points.npz`, `{geometry}_covers.npz`: actual shared training points/weights and both local decompositions.',
           '', 'Original anchoring is only represented by its specified baseline here. Neither a new local-method benefit nor an independent-marginal result is evidence of an independent anchoring benefit. This new protocol must not be numerically pooled with the original C1/original-bandwidth Fourier study.']
    rows+=['','## Resource telemetry','',
           'Peak memory below is PyTorch allocation for this development implementation, including prepared feature/window caches and optimizer states. Reserved memory is allocator reservation, not live tensors. These figures are not a complete resource-to-accuracy comparison; CPU geometry preparation and inference are additional costs.',
           '', '| Case | Representation | Trainable parameters | Peak allocated MiB | Peak reserved MiB |',
           '|---|---|---:|---:|---:|']
    for case in p['cases']:
        for method in p['methods']:
            r=read(B/(case+'_'+method)/'result.json')
            rows.append(f"| {case} | {method} | {r['parameters']} | {r['peak_allocated_bytes']/2**20:.2f} | {r['peak_reserved_bytes']/2**20:.2f} |")
    diagnostic=B/'diagnostic/analysis.json'
    if diagnostic.exists():
        d=read(diagnostic)
        rows+=['','## Post-fit physical diagnostic','',
               'After the C1 accuracy discrepancy was observed, a separate diagnostic protocol was registered. It evaluates all 15 unchanged terminals on 24,000 independent uniform-solid points per geometry and on the two training strata. It performs no optimization and reads no FEM fields. This is post-fit diagnosis, not a pre-registered primary outcome.',
               '', '| Case | Representation | Training-uniform residual MSE sum | Independent residual MSE sum | Independent/training ratio |',
               '|---|---|---:|---:|---:|']
        for case in p['cases']:
            for method in p['methods']:
                r=d['results'][case+'_'+method]
                rows.append(f"| {case} | {method} | {r['training_uniform']['total_mean']:.6f} | {r['independent']['total_mean']:.6f} | {r['independent_to_training_uniform_total_ratio']:.3f} |")
        rows += ['', 'The residual is the sum of squared two equilibrium and three constitutive components, excluding boundary penalties. A large ratio identifies poor residual generalization of a fitted field; it does not by itself identify a unique cause or validate a repair. Pointwise diagnostic residuals and actual points are saved under `results/R2_P2A/diagnostic/`.']
    dest=ROOT/'P2A_Development_Report.md';assert not dest.exists()
    dest.write_text('\n'.join(rows)+'\n',encoding='utf-8')
    print(dest)


if __name__=='__main__':main()
