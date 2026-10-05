"""Evidence-grounded English report for the separately authorized FEM evaluation."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'results/R2_P2D_engineering_exploratory'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def link(label,p):return f'[{label}](<{Path(p).resolve().as_posix()}>)'
def fmt(x):return f'{x:.5g}'

def main():
    assert not (B/'completion.json').exists()
    a=read(B/'analysis.json');p=read(ROOT/'protocols/R2_P2D_engineering_exploratory.json')
    lines=['# P2D: separate post-hoc engineering evaluation','',
      'All seven representations and all three cases have now been evaluated against the existing fine FEM references. These are the 21 already frozen P2D terminal fields, with no new optimization, checkpoint selection, geometry tuning or FEM-based sampling. This author-authorized study is exploratory: P2D originally failed its physical-residual gate, and that record remains unchanged. This evaluation is stored in a separate directory.', '',
      '## Main finding','',
      'The geometry-derived Gaussian–Fourier input has a useful single-seed signal on the slender cavity L1. Relative to complete half-bandwidth Fourier, area-weighted displacement error falls from 19.245% to 10.796% and wall displacement error from 25.253% to 13.588%. Displacement improves over 96.44% of the solid area. Stress improves less, from 20.035% to 19.178%. The independent geometry-rule contrast is also favorable: the uniform-scale Gaussian–Fourier control has displacement/stress/wall errors of 26.367/21.037/35.110%.', '',
      'C1 is a trade-off: geometry-derived inputs improve global and wall displacement slightly relative to complete half-Fourier, but global stress and both prescribed convergence errors worsen. C8 does not show overall superiority; its wall displacement error increases from 0.39445% to 0.45634%. Thus neither a universal engineering advantage nor restoration of the original anchoring attribution is supported.', '',
      'The L1 result is worth a separately registered reproducibility and budget study, not immediate manuscript promotion. Its remaining 10.8% displacement and 19.2% stress errors at this short development budget are substantial, and near-wall stress error remains above 50%. Strong comparative gains do not establish adequate engineering accuracy.', '',
      '## Evidence and reference checks','',
      'All terminal, encoder, protocol and previous result hashes were checked before evaluation. The nine plain-model terminals equal their P2A states tensor by tensor; their existing frozen prediction files were reused without copying, and recomputed global/wall metrics exactly matched the prior evaluation. Twelve hybrid fields were evaluated afresh at the same locations. All 21 have separate pointwise error-norm files in this study.', '',
      'References and exact interpolation arrays are reused by path and hash from P2A. Node and integration-point metrics are kept separate. The stress vector uses [sxx, syy, sxy] with the same Euclidean convention as previous evaluation; it is not a von Mises error. Displacement values and convergence are in the normalized study units, not converted to physical millimetres.', '',
      '| Case | Integration points | FEM stress reconstruction error, % | Wall offset sensitivity, max absolute | Solid area |',
      '|---|---:|---:|---:|---:|']
    for case,v in a['references'].items():
        q=v['provenance'];lines.append(f"| {case} | {q['integration_points']} | {q['stress_reconstruction_percent']:.6g} | {q['wall_offset_sensitivity_max']:.6g} | {q['area']:.9g} |")
    lines += ['', '## All global and wall results','',
      'All entries are relative L2 percentages. Area U and S use the same FEM integration-point locations and quadrature weights. Point U/S use unweighted means at those locations. Node U uses the original displacement nodes. Wall U is arc-length weighted.', '',
      '| Case | Representation | Area U | Area S | Wall U | Point U | Point S | Node U |',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    for case,rows in a['cases'].items():
        for method in p['methods']:
            v=rows[method]['metrics'];vals=[v[k]['relative_l2_percent'] for k in ['u_area','s_area','wall_u','u_point','s_point','u_original_nodes']]
            lines.append('| '+case+' | '+method+' | '+' | '.join(fmt(x) for x in vals)+' |')
    lines += ['', '## Geometry-rule and strong-baseline contrasts','',
      'Ratios below one favor geometry_rbf_fourier. These are descriptive comparisons, not a new pass/fail gate. The uniform-scale hybrid has the same number of Gaussian/global inputs and trainable parameters, isolating the combined centre/scale rule. Complete half-Fourier is the unchanged stronger benchmark. Gaussian centres and widths are not separately isolated.', '',
      '| Case | Comparator | Area U ratio | Area S ratio | Wall U ratio | U area improved | S area improved | Wall length improved |',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    for case,controls in a['contrasts'].items():
        row=a['cases'][case]['geometry_rbf_fourier']
        for control,ratios in controls.items():
            fractions=row['area_fraction_lower_vector_error'][control]
            lines.append(f"| {case} | {control} | {ratios['u_area']:.5g} | {ratios['s_area']:.5g} | {ratios['wall_u']:.5g} | {100*fractions['u']:.2f}% | {100*fractions['s']:.2f}% | {100*row['wall_fraction_lower_displacement_error'][control]:.2f}% |")
    lines += ['', '## Spatial regions for every representation','',
      'The near-wall band is distance ≤0.05 in normalized coordinates, with the complement as the far field. L1 corner regions are within 0.02 of a polygon vertex. These masks were fixed from geometry before this evaluation. Region-relative L2 percentages have their own FEM norm denominator; they cannot be added to obtain the global percentage.', '',
      '| Case | Representation | Near-wall U | Far-field U | Near-wall S | Far-field S | Corner U | Corner S |',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    for case,rows in a['cases'].items():
        for method in p['methods']:
            v=rows[method]['metrics']
            values=[fmt(v[k]['relative_l2_percent']) if k in v else '—' for k in ['u_near_wall','u_far_wall','s_near_wall','s_far_wall','u_corner','s_corner']]
            lines.append('| '+case+' | '+method+' | '+' | '.join(values)+' |')
    lines += ['', '## Absolute-error distribution and concentration','',
      'Each quantity is the Euclidean error norm of U or S. MAE and percentiles use quadrature area weights. The worst 1% area share is the fraction of total squared error contained in the highest-error 1% of solid area, with a partial quadrature-bin contribution at the cutoff. This avoids confusing dense corner sampling with a large affected area. Complete 90% and 95% quantiles and maxima are additionally in each metrics JSON.', '',
      '| Case | Representation | Field | Weighted MAE | Median | 99th percentile | Worst 1% area share of squared error |',
      '|---|---|---|---:|---:|---:|---:|']
    for case,rows in a['cases'].items():
        for method in p['methods']:
            for field in ['u','s']:
                v=rows[method]['metrics'][field+'_area'];q=v['vector_absolute_error_quantiles']
                lines.append(f"| {case} | {method} | {field.upper()} | {v['vector_mae']:.6g} | {q[0]:.6g} | {q[3]:.6g} | {100*v['worst_one_percent_area_squared_error_share']:.2f}% |")
    lines += ['', '## Prescribed engineering quantities','',
      'Vertical convergence is uy(crown) − uy(invert); horizontal convergence is ux(right) − ux(left). Signs are retained. Absolute errors are shown without dividing by possibly small components. Full signed displacement vectors, extrema-vector errors and componentwise absolute errors are retained in JSON and raw prediction arrays.', '',
      '| Case | Representation | Vertical prediction | Vertical reference | Vertical absolute error | Horizontal prediction | Horizontal reference | Horizontal absolute error |',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    for case,rows in a['cases'].items():
        for method in p['methods']:
            e=rows[method]['metrics']['engineering'];cp=e['convergence_prediction'];cr=e['convergence_reference'];ce=e['convergence_absolute_error']
            lines.append('| '+case+' | '+method+' | '+' | '.join(fmt(x) for x in [cp[0],cr[0],ce[0],cp[1],cr[1],ce[1]])+' |')
    lines += ['', 'For L1, geometry_rbf_fourier reduces vertical convergence absolute error from 0.98984 to 0.44293 and horizontal error from 0.24932 to 0.012149 relative to half-Fourier. This is a concrete engineering signal alongside the broad displacement improvement. Some individual components still regress: invert uy absolute error rises from 0.02439 to 0.03982, for example.', '',
      'On C1, the corresponding convergence errors increase from 0.01557/0.01038 to 0.02567/0.02243 even though wall-displacement L2 improves. On C8, they increase from 0.000494/0.005322 to 0.000925/0.010448. A lower aggregate wall error therefore cannot stand in for all prescribed engineering quantities.', '',
      '## Resources and attribution','',
      'All P2D representations have 110705 trainable parameters. K fixed Gaussians replace K global features rather than adding a new local network. The original development run records below include preparation/optimization/checkpoint telemetry, but no idle-machine benchmark was registered. No fair speedup or time-to-accuracy inference is licensed.', '',
      '| Case | Representation | Closure evaluations | Peak allocated GPU MiB |',
      '|---|---|---:|---:|']
    for case,rows in a['cases'].items():
        for method in p['methods']:
            v=read(rows[method]['training_record_path'])
            lines.append(f"| {case} | {method} | {v['closure_evaluations']} | {v['peak_allocated_bytes']/2**20:.2f} |")
    lines += ['', 'The geometry enhancement improves the tanh anchored and independent-bias representations particularly on L1, but the independent-bias hybrid remains competitive or better on many metrics. The experiment supports investigating the automatic geometry rule; it does not restore an independent accuracy claim for the original anchoring relation. C1 against original-bandwidth Fourier is a separate historical result and is not redefined by this half-bandwidth comparison.', '',
      '## Research decision','',
      'Retain the P2D shared geometry-input candidate for a bounded reproducibility and budget study, with complete half-Fourier and uniform-scale hybrids kept as controls on all three cases. Prespecify new seeds and all budget endpoints, report every endpoint rather than selecting the FEM-best one, and count construction/preparation, training and memory. Preserve the C8 and C1 engineering regressions as explicit constraints. A formal time-to-accuracy experiment additionally requires a newly confirmed idle-machine window.', '',
      'This decision is based on an exploratory single-seed engineering signal. It is not evidence of cross-shape validation on T1/S1, final engineering adequacy, a successful revision, or acceptance by reviewers. The current finding favors a limited follow-up over another unconstrained architecture search, while leaving the alternative optimization/conditioning route available.', '',
      '## Source-data map','',
      '- '+link('Evaluation protocol',ROOT/'protocols/R2_P2D_engineering_exploratory.json')+'.',
      '- '+link('Complete analysis and all reference/prediction paths and hashes',B/'analysis.json')+'.',
      '- '+link('Evaluation directory',B)+': `<case>_metrics.json` contains every method, region, quantile and prescribed engineering quantity. `<case>_<method>_error_norms.npz` contains pointwise `u`, `s` and `wall_u` error norms in exact reference-array order.',
      '- Twelve new hybrid `*_predictions.npz` files contain `q_ip=[ux,uy,sxx,syy,sxy]`, `u_node`, `u_wall` and `u_extrema`. The nine unchanged plain-model files are referenced in place from '+link('frozen P2A evaluation',ROOT/'results/R2_P2A/evaluation')+'; they were not modified or duplicated.',
      '- The three `*_reference.npz` files in that P2A directory contain `xy_ip`, `area_weight`, `u_ip`, `s_ip`, original nodal coordinates/displacements, full wall coordinates/weights/displacements, extrema and all spatial masks. The present analysis records source FEM provenance and hashes for each.',
      '- '+link('Closed P2D training batch',ROOT/'results/R2_P2D_shared_features')+' supplies terminal weights, input encoders, training points, complete accepted-step traces, original residual evaluation and the unchanged failed gate. Its original report records the state at closure; this separate report records the later author-authorized evaluation.',
      '- '+link('Evaluator',ROOT/'code/evaluate_p2d_engineering.py')+' and '+link('report generator',Path(__file__))+'.', '',
      'No manuscript, response letter, first-round file or previously closed batch was changed. Scientific training count remains 66; this study adds evaluations only.']
    out=ROOT/'P2D_Engineering_Exploration_Report.md';out.write_text('\n'.join(lines)+'\n',encoding='utf-8');print(out)

if __name__=='__main__':main()
