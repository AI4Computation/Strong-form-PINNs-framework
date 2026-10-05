"""Pure-statistics postprocessing; safe to test while the GPU cohort runs."""
import math
import statistics
import random

def summary(values):
    values = list(values)
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                median=statistics.median(values) if values else None,
                sd=statistics.stdev(values) if len(values)>1 else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None)

def crossing(trace, keys, target):
    assert trace and all(a['accepted_step'] < b['accepted_step'] for a,b in zip(trace,trace[1:]))
    ok = [all(row[k] is not None and math.isfinite(row[k]) and row[k] <= target for k in keys)
          for row in trace]
    first = next((i for i,v in enumerate(ok) if v), None)
    stable = next((i for i in range(len(ok)-2) if all(ok[i:i+3])), None)
    result = dict(reached=first is not None, first=None, consecutive_three=None,
                  terminal_reached=ok[-1], pass_to_fail_transitions=sum(a and not b for a,b in zip(ok,ok[1:])),
                  terminal_step=trace[-1]['accepted_step'])
    if first is not None:
        row = trace[first]
        previous = trace[first-1] if first else None
        result['first'] = dict(accepted_step=row['accepted_step'],
                              previous_observed_step=previous['accepted_step'] if previous else None)
        for clock in ['optimization_s', 'state_elapsed_s', 'available_elapsed_s', 'cold_available_s']:
            result['first'][clock+'_lower'] = previous[clock] if previous else 0.
            result['first'][clock+'_upper'] = row[clock]
    if stable is not None:
        a,b = trace[stable],trace[stable+2]
        result['consecutive_three'] = dict(start_step=a['accepted_step'], confirmation_step=b['accepted_step'])
        for clock in ['optimization_s', 'available_elapsed_s', 'cold_available_s']:
            result['consecutive_three']['start_'+clock] = a[clock]
            result['consecutive_three']['confirmation_'+clock] = b[clock]
    return result

def summarize_crossings(rows):
    return dict(n=len(rows), reached=sum(r['reached'] for r in rows),
                not_reached=sum(not r['reached'] for r in rows),
                terminal_reached=sum(r['terminal_reached'] for r in rows),
                consecutive_three_count=sum(r['consecutive_three'] is not None for r in rows),
                first_success_only={clock:summary(r['first'][clock+'_upper'] for r in rows if r['reached'])
                    for clock in ['optimization_s','available_elapsed_s','cold_available_s']},
                confirmed_success_only={clock:summary(r['consecutive_three']['confirmation_'+clock]
                    for r in rows if r['consecutive_three'] is not None)
                    for clock in ['optimization_s','available_elapsed_s','cold_available_s']})

def paired_ratio(numerator, denominator, seed=81234):
    assert len(numerator) == len(denominator) and len(numerator)>0
    assert all(x>0 for x in numerator+denominator)
    log = [math.log(a/b) for a,b in zip(numerator,denominator)]
    rng = random.Random(seed)
    bootstrap = sorted(math.exp(statistics.mean(rng.choices(log,k=len(log)))) for _ in range(10000))
    return dict(pairs=len(log), geometric_mean_ratio=math.exp(statistics.mean(log)),
                bootstrap_percentile_95=[bootstrap[249],bootstrap[9749]],
                numerator_smaller_pairs=sum(a<b for a,b in zip(numerator,denominator)))

def selfcheck():
    trace = [dict(accepted_step=50*i, a=a, b=b, optimization_s=float(i)*3,
                  state_elapsed_s=float(i)*5+1, available_elapsed_s=float(i)*5+2,
                  cold_available_s=float(i)*5+7)
             for i,(a,b) in enumerate([(6,4),(4,4),(3,6),(4,4),(3,3),(2,2)])]
    r = crossing(trace,['a','b'],5)
    assert r['first']['accepted_step']==50 and r['first']['optimization_s_lower']==0
    assert r['first']['optimization_s_upper']==3
    assert r['consecutive_three']['start_step']==150 and r['consecutive_three']['confirmation_step']==250
    assert r['pass_to_fail_transitions']==1 and r['terminal_reached']
    assert not crossing(trace,['a','b'],1)['reached']
    assert crossing(trace[-2:],['a','b'],5)['consecutive_three'] is None
    assert crossing(trace,['a','b'],6)['first']['accepted_step']==0
    assert summarize_crossings([r,crossing(trace,['a','b'],1)])['not_reached']==1
    bad=[dict(trace[0],a=float('nan'))]
    assert not crossing(bad,['a','b'],6)['reached']
    ratio=paired_ratio([1.]*8,[2.]*8)
    assert ratio['geometric_mean_ratio']==.5 and ratio['numerator_smaller_pairs']==8
    return dict(passed=True, assertions=10,
                cases=['nonmonotone_joint_crossing','actual_observation_bracket','three_consecutive',
                       'never_reached','insufficient_followup','initial_success','failure_count',
                       'nonfinite_metric','paired_ratio'])

if __name__ == '__main__':
    from common import HERE,write,sha,utc
    write(HERE/'post_metrics_checks.json',dict(**selfcheck(),checked_utc=utc(),source_sha256=sha(__file__),
                                             fem_evaluations=0,formal_timing=False))
    print('Postprocessing logic checks passed; no model or FEM data loaded.')
