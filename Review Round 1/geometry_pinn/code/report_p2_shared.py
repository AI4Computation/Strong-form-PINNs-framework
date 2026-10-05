"""Generate the evidence table and source map from frozen P2D outputs."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results/R2_P2D_shared_features'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def link(label,p):return f'[{label}](<{Path(p).resolve().as_posix()}>)'

def main():
    target=ROOT/'P2D_Shared_Feature_Report.md';assert not (B/'completion.json').exists()
    p=read(ROOT/'protocols/R2_P2D_shared_features.json');a=read(B/'physical_validation/analysis.json');pre=read(B/'preflight.json')
    lines=['# P2D: shared global PINN with cavity-scale input features', '',
      'This is a single-seed, finite-budget development study. It evaluates the common cavity-derived centre/scale rule inside one fully trainable global mixed strong-form PINN. It does not establish novelty, displacement/stress accuracy, a separate anchoring advantage, or fair computational efficiency.', '',
      f"Registered physical-screen decision: **{'PASS' if a['gates']['advance_to_FEM'] else 'FAIL'}**. All 21 terminal fields were frozen before independent validation. FEM fields were not read for this batch.", '',
      '## Construction and controls','',
      'Every representation has 1,000 fixed input features followed by the same trainable 100–100–5 network: 110,705 trainable parameters. K unnormalized Gaussian inputs replace K original inputs; Gaussian widths and centres come from the frozen geometry cover. No per-patch trainable networks or normalized output windows are used. All hidden layers remain trainable. The uniform-scale control has exactly K solid-domain farthest-grid centres and one common width 0.75/sqrt(K). This tests the combined position/scale rule; it does not isolate position from scale.', '',
      'Before training, a uniform centre in the T1 cavity caused a weak-response feature. The control was corrected to reject void centres, and the complete preflight was rerun. This change and its timing are recorded in the protocol. The control is consequently domain-aware but does not adapt to local wall scales.', '',
      '| Geometry | Gaussian features | Remaining global features | Lowest maximum Gaussian response, geometry / uniform |',
      '|---|---:|---:|---:|']
    for g,v in pre['geometry'].items():
        lines.append(f"| {g} | {v['gaussians']} | {v['global_features']} | {v['responses']['geometry']['min_max_response']:.4g} / {v['responses']['uniform']['min_max_response']:.4g} |")
    lines += ['', 'The 28 geometry/representation combinations passed field, spatial-Jacobian, independent objective/parameter-gradient and full-batch finite-GPU-gradient checks. Translation and isotropic scaling are handled by the common normalization. T1 and S1 were audited, but were not trained: this is not cross-shape performance confirmation.', '',
      'Training used the exact P2A points and loss weights: 6,000 interior points with a 90% uniform-point and 10% support-probe objective, 500 points per outer edge and 1,000 hole-boundary points. Seed 260926, 400 accepted L-BFGS steps maximum, 600 closure budget, history 50, float32, TF32 disabled. The original three plain models reproduced their P2A terminal tensors exactly in all three cases.', '',
      '## Independent physical validation','',
      'Two new uniform-solid sets contain 24,000 points each (seeds 29260001 and 29260002). R is the mean sum of squared equilibrium and constitutive residuals. These are dimensionless physical residual scores under the shared formulation, not relative FEM errors. Lower is better; ratios should not be read as solution-error ratios.', '',
      '| Case | Representation | R set 1 | R set 2 | Independent / training-uniform range |',
      '|---|---|---:|---:|---:|']
    for case in p['cases']:
        for method in p['methods']:
            v=a['results'][case+'_'+method];r=[s['total_mean'] for s in v['validation']];ratio=v['independent_training_ratios']
            lines.append(f'| {case} | {method} | {r[0]:.6g} | {r[1]:.6g} | {min(ratio):.3g}–{max(ratio):.3g} |')
    lines += ['', '## Registered primary gates','',
      'Geometry-RBF-Fourier must have R ≤ 0.9 × uniform-RBF-Fourier and R ≤ 1.1 × plain half-Fourier on every case and both sets. Its independent/training-uniform ratio must not exceed 10. Passing these necessary screens would only justify a subsequent frozen-terminal FEM and engineering evaluation.', '',
      '| Case | Geometry / uniform, both sets | Geometry / half-Fourier, both sets | Gain / competitiveness / reliability |',
      '|---|---:|---:|---|']
    for case,v in a['gates']['cases'].items():
        pair=lambda key:' / '.join(f'{x:.4g}' for x in v[key])
        lines.append(f"| {case} | {pair('geometry_over_uniform')} | {pair('geometry_over_fourier')} | {v['independent_gain']} / {v['competitive']} / {v['reliability']} |")
    lines += ['', '## Spatial concentration, without FEM','',
      'The following table uses the first independent set for a compact display. Complete quantiles and five residual components for both sets are saved. The percentage of points improved compares the residual norm at identical coordinates, not the displacement/stress error. It helps distinguish widespread improvement from a small set of extreme residuals.', '',
      '| Case | Representation | Residual norm median | 99th percentile | Top 1% residual-square mass | Points better than plain half-Fourier |',
      '|---|---|---:|---:|---:|---:|']
    for case in p['cases']:
        with np.load(B/f'physical_validation/{case}_fourier_half_residuals.npz') as z:base=(z['validation0']**2).sum(1)
        for method in p['methods']:
            v=a['results'][case+'_'+method]['validation'][0]
            with np.load(B/f'physical_validation/{case}_{method}_residuals.npz') as z:e=(z['validation0']**2).sum(1)
            win='—' if method=='fourier_half' else f'{100*np.mean(e<base):.2f}%'
            lines.append(f"| {case} | {method} | {v['norm_quantiles'][0]:.5g} | {v['norm_quantiles'][3]:.5g} | {100*v['top_one_percent_mass']:.2f}% | {win} |")
    lines += ['', '## Budget and resource record','',
      'All fits ran sequentially on one GPU. The table reports allocated GPU memory and actual closure counts. Time-to-accuracy and speedup are not reported: no formal idle-machine timing window was registered. Preparation, optimization and checkpoint-I/O elapsed times remain development telemetry in each result.json. Peak allocation covers the runner/preparation/optimizer sequence; it is not total system memory or a portable resource guarantee.', '',
      '| Representation | Accepted steps range | Closure evaluations range | Peak allocated MiB range |',
      '|---|---:|---:|---:|']
    for method in p['methods']:
        rows=[read(B/(case+'_'+method)/'result.json') for case in p['cases']]
        span=lambda key:f'{min(v[key] for v in rows)}–{max(v[key] for v in rows)}'
        mem=[v['peak_allocated_bytes']/2**20 for v in rows]
        lines.append(f"| {method} | {span('accepted_steps')} | {span('closure_evaluations')} | {min(mem):.2f}–{max(mem):.2f} |")
    lines += ['', '## Interpretation and limits','',
      'The common geometry rule reduced the held-out physical score relative to the uniform-scale hybrid by 16.3–24.0% on C1 and 45.4–46.8% on L1. On L1 it also reduced that score by 33.0–38.9% relative to plain half-Fourier; 79.60% of points in the first independent set had a lower residual norm. Thus the L1 signal is spatially broad, not only a lower value at a few extreme points. C1 was comparable with plain half-Fourier, with opposite small changes on the two independent sets.', '',
      'C8 failed the independent-gain condition: geometry/uniform was 1.031–1.043, although geometry/plain-Fourier was within the registered 10% competitiveness margin. All three cases passed the residual generalization condition (geometry-RBF-Fourier independent/training ratios 1.42–3.49). The full advance gate nevertheless failed. These independent point sets do not replace independent training seeds.', '',
      'The geometry-plus-independent-bias hybrid had lower physical residuals than the geometry-plus-anchored hybrid on every case and both independent sets. The present results therefore provide no rescue of an independent anchoring-accuracy claim. A promising geometry-rule signal must be separated from original anchoring attribution.', '',
      'Next hypothesis, not a result: replacing 244–460 of the global features imposes a fixed geometry-dependent trade-off regardless of load. A separate candidate could preserve the complete global Fourier representation at initialization and admit geometry features through trainable amplitudes driven only by the common physical objective. It would require uniform-scale and redundant-global amplitude controls to distinguish geometry from added parameters and reparameterization. P2D results and gates remain unchanged; no such follow-up has been trained or validated here.', '',
      'The primary gate is fixed before training. An exploratory anchored or independent-bias hybrid cannot replace the primary candidate after seeing the results. Those pairs are reported in full; neither a single seed nor an independent-marginal comparison proves a positive anchoring effect. Exact bias permutation and repetitions would be required for that claim.', '',
      'A failed residual screen is not proof that all displacement or stress errors are larger. No FEM error maps, wall displacement or engineering convergence values were computed here. Such quantities must not be inferred from this report or imported from a different candidate. Nor does the present finite-budget test reject all Gaussian embeddings or all shared PINNs.', '',
      'Generic Gaussian/RBF input maps and RBF–Fourier combinations have prior literature. The relevant proposed value is the automatic cavity-scale rule and its independent engineering benefit; neither is established merely by implementing a hybrid. See the primary-source boundary note linked below.', '',
      '## Exact source-data map','',
      '- '+link('Frozen protocol',ROOT/'protocols/R2_P2D_shared_features.json'),
      '- '+link('Preflight, source/input hashes and numerical errors',B/'preflight.json'),
      '- '+link('Training manifest and terminal hashes',B/'manifest.json'),
      '- '+link('All physical scores, gates and exact plain-model reproduction',B/'physical_validation/analysis.json'),
      '- '+link('Batch directory',B)+': `<case>_<method>/terminal.pt`, `step_0000/0100/0200/0400.pt`, `trace.json`, `result.json`; `<geometry>_points.npz` and `<geometry>_covers.npz` provide actual training locations, weights and encoder geometry.',
      '- '+link('Pointwise residual directory',B/'physical_validation')+': `<geometry>_points.npz` contains both independent coordinate arrays. `<case>_<method>_residuals.npz` contains `training`, `validation0`, `validation1`, each with columns [equilibrium_x, equilibrium_y, constitutive_xx, constitutive_yy, constitutive_xy]. Training coordinates are in the batch geometry points file. No FEM data are present.',
      '- '+link('Shared feature implementation',ROOT/'code/shared_geometry_features.py')+', '+link('validation code',ROOT/'code/validate_p2_shared.py')+', '+link('report generator',Path(__file__)),
      '- '+link('Primary-source overlap note',ROOT/'literature/候选方向与已有方法边界.md'),'',
      'The first-round manuscripts, research files and results remain frozen. No second-round Word manuscript or response letter was edited in this batch.']
    target.write_text('\n'.join(lines)+'\n',encoding='utf-8');print(target)

if __name__=='__main__':main()
