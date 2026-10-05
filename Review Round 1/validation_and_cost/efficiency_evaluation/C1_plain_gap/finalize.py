"""Audit and report the complete measured cohort without selecting favourable runs."""
import sys
from pathlib import Path
from common import HERE,ROOT,OLD,METHODS,read,write,sha,utc,verify_hashes
from post_metrics import summary,summarize_crossings,paired_ratio,selfcheck

LABELS={'vanilla_matched':'容量配平普通PINN','anchored':'原锚定PINN','fourier_half':'半带宽Fourier'}

def main():
    assert not (HERE/'delivery_audit.json').exists(), 'A frozen delivery cannot be overwritten.'
    selfcheck()
    p,m=read(HERE/'protocol.json'),read(HERE/'manifest.json')
    assert m['status']=='post_evaluation_complete' and len(m['completed'])==24 and not m['flagged']
    c,e=read(HERE/'cohort_freeze.json'),read(HERE/'evaluation_freeze.json')
    post=read(HERE/'post_protocol.json')
    for mapping in [p['source_sha256'],p['protected_inputs_sha256'],c['files_sha256'],
                    e['files_sha256'],post['source_sha256'],post['reference_sha256']]:
        verify_hashes(mapping)
    raw={}; evaluated={}; monitors={}; checks=[]
    def check(name,passed):
        checks.append(dict(name=name,passed=bool(passed)))
    for rec in p['run_order']:
        identity=rec['identity']; r=read(HERE/'runs'/identity/'result.json')
        v=read(HERE/'evaluation'/f'{identity}.json')
        mon=read(HERE/'telemetry'/f'{identity}_run.json')
        raw[identity],evaluated[identity],monitors[identity]=r,v,mon
        check(identity+'_frozen_timing',r['formal_timing'] and r['protocol_sha256']==m['protocol_sha256'])
        check(identity+'_no_interference',not mon['flagged'] and mon['exit_code']==0)
        check(identity+'_capacity',r['trainable_parameters']==p['parameter_counts'][rec['method']])
        env=r['environment']
        check(identity+'_runtime',env['gpu_count']==1 and env['torch_threads']==4 and env['dtype']=='float32'
              and not env['tf32'] and env['deterministic'] and env['start_power']['ac_line_status']==1)
        check(identity+'_closures',r['closure_evaluations']==len(r['closure_losses'])
              and r['accepted_steps']<=2000 and r['closure_evaluations']>=r['accepted_steps'])
        check(identity+'_step_schedule',[o['accepted_step'] for o in r['observations']]
              ==sorted(set([0]+list(range(50,r['accepted_steps']+1,50))+[r['accepted_steps']])))
        check(identity+'_time_order',all(0<=o['optimization_s']<=o['state_elapsed_s']<=o['available_elapsed_s']
              <=r['observed_solution_s'] for o in r['observations']))
        check(identity+'_cold_clock',abs(r['script_entry_to_solution_s']-r['before_solver_process_elapsed_s']
              -r['observed_solution_s'])<1e-6)
        check(identity+'_monotone_observation_times',all(a['optimization_s']<b['optimization_s']
              and a['available_elapsed_s']<b['available_elapsed_s'] for a,b in zip(r['observations'],r['observations'][1:])))
        check(identity+'_evaluation_matches_clock',all(a['checkpoint_sha256']==b['checkpoint_sha256']
              and a['optimization_s']==b['optimization_s'] for a,b in zip(r['observations'],v['trace'])))
        check(identity+'_query_budget',r['query']['points']==4096 and len(r['query']['device_resident_batch_s'])==50
              and len(r['query']['host_to_device_to_host_batch_s'])==50)
    check('unique_query_coordinates',len({r['query']['point_sha256'] for r in raw.values()})==1)
    check('all_seed_pairs_present',all(f'C1_{method}_s{seed}' in raw for method in METHODS for seed in range(41,49)))
    write(HERE/'numerical_delivery_checks.json',dict(passed=all(x['passed'] for x in checks),checks=checks,utc=utc()))
    assert all(x['passed'] for x in checks), 'Delivery audit failed.'
    groups={}
    for method in METHODS:
        ids=[f'C1_{method}_s{seed}' for seed in range(41,49)]
        rr=[raw[i] for i in ids]; vv=[evaluated[i] for i in ids]
        group=dict(parameters=p['parameter_counts'][method],targets={},cost={},resources={},endpoint={})
        for mode in ['point','area','point_u_area_s']:
            group['targets'][mode]={str(t):summarize_crossings([v['targets'][mode][str(t)] for v in vv])
                                    for t in p['accuracy_targets_pct']}
        for key in ['optimization_s','observed_solution_s','script_entry_to_solution_s','import_s',
                    'uniform_cuda_warmup_s','observation_and_logging_s','final_solution_save_s',
                    'accepted_steps','closure_evaluations']:
            group['cost'][key]=summary(r[key] for r in rr)
        group['cost']['process_launch_to_exit_s']=summary(monitors[i]['process_launch_to_exit_s'] for i in ids)
        for key in ['model_ready_s','samples_ready_s','optimizer_ready_s','setup_audit_s']:
            group['cost'][key]=summary(r['construction'][key] for r in rr)
        for key in ['device_resident_batch_s','host_to_device_to_host_batch_s']:
            # Treat seeds as replicates; fifty repeated queries within a seed are not 400 independent samples.
            group['cost'][key]=summary(summary(r['query'][key])['median'] for r in rr)
        for key in ['allocated_MiB','reserved_MiB']:
            group['resources'][key]=summary(r['gpu_solver_allocator_peaks'][key] for r in rr)
        for key in ['peak_rss_lifetime_MiB','peak_private_commit_lifetime_MiB']:
            group['resources'][key]=summary(r['process_memory_at_solution'][key] for r in rr)
        group['resources']['portable_model_bytes']=summary((HERE/'runs'/i/'model.pt').stat().st_size for i in ids)
        for key in ['point_stress_pct','area_stress_pct','u_point_pct','u_area_pct',
                    'mean_absolute_error_pct','near_cavity_area_stress_pct']:
            group['endpoint'][key]=summary(v['endpoint'][key] for v in vv)
        for key in ['p50','p90','p95','p99','p99.9']:
            group['endpoint'][key]=summary(v['endpoint']['quantiles_pct'][key] for v in vv)
        for key in ['horizontal','vertical']:
            group['endpoint'][key+'_wall_error_pct']=summary(v['endpoint']['wall'][key]['relative_error_pct'] for v in vv)
        group['endpoint']['wall_vector_u_pct']=summary(v['endpoint']['wall']['wall_vector_u_pct'] for v in vv)
        groups[method]=group
    comparisons={}
    for other in ['vanilla_matched','fourier_half']:
        pair={}
        for key in ['optimization_s','observed_solution_s','script_entry_to_solution_s']:
            pair[key]=paired_ratio([raw[f'C1_anchored_s{s}'][key] for s in range(41,49)],
                                   [raw[f'C1_{other}_s{s}'][key] for s in range(41,49)])
        for mode in ['point','area']:
            for target in ['5','2','1']:
                aa=[evaluated[f'C1_anchored_s{s}']['targets'][mode][target] for s in range(41,49)]
                bb=[evaluated[f'C1_{other}_s{s}']['targets'][mode][target] for s in range(41,49)]
                matched=[s for s,(a,b) in enumerate(zip(aa,bb)) if a['reached'] and b['reached']]
                key=f'{mode}_{target}_first_time_ratio'
                pair[key]=dict(all_seeds=8,simultaneously_successful_pairs=len(matched),
                    scope='Ratio only within pairs where both methods were observed to reach target; not an all-seed speedup.')
                if matched:
                    pair[key]['successful_pairs']=paired_ratio([aa[s]['first']['optimization_s_upper'] for s in matched],
                                                               [bb[s]['first']['optimization_s_upper'] for s in matched])
        comparisons['anchored_vs_'+other]=pair
    spatial=read(HERE/'evaluation/spatial_pairs.json')['pairs']
    spatial_summary={key:dict(better_area=summary(r['anchored_better_area_pct'] for r in rows),
                             majority_area_wins=sum(r['anchored_better_area_pct']>50 for r in rows))
                     for key,rows in spatial.items()}
    result=dict(completed_utc=utc(),runs=24,groups=groups,paired= comparisons,spatial=spatial_summary,
        accepted_steps=sum(r['accepted_steps'] for r in raw.values()),
        closures=sum(r['closure_evaluations'] for r in raw.values()),
        checkpoints=e['checkpoint_evaluations'],checks_passed=len(checks),
        scope='Current controlled C1 cost cohort only. No mixing with older timings; no independent new-method or cross-shape validation.')
    write(HERE/'analysis.json',result)
    report=make_report(result)
    path=ROOT/'C1普通PINN共同精度成本补测.md'
    path.write_text(report,encoding='utf-8')
    sealed=[HERE/n for n in ['analysis.json','protocol.json','post_protocol.json','cohort_freeze.json',
        'evaluation_freeze.json','idle_confirmation.json','numerical_delivery_checks.json','preparation_audit.json']]
    sealed += [Path(x) for x in c['files_sha256']] + [Path(x) for x in e['files_sha256']] + [path]
    write(HERE/'delivery_audit.json',dict(passed=True,completed_utc=utc(),runs=24,
        files_sha256={str(file):sha(file) for file in sealed},source_sha256={**p['source_sha256'],**post['source_sha256']},
        protected_inputs_sha256=p['protected_inputs_sha256'],reference_sha256=post['reference_sha256']))
    m.update(status='delivered',active=None,updated_utc=utc())
    write(HERE/'manifest.json',m)
    print('DELIVERED',path,flush=True)

def make_report(a):
    g=a['groups']
    def f(value,digits=2):
        return 'NR' if value is None else f'{value:.{digits}f}'
    def ms(value):
        return f(value*1000,3)
    primary=['| 方法 | 5%首次达标 | 5%连续三次确认 | 首次达标净优化/秒 | 保存可用/秒 | 冷启动至保存可用/秒 |',
             '|---|---:|---:|---:|---:|---:|']
    for method in METHODS:
        r=g[method]['targets']['point']['5']; costs=r['first_success_only']
        primary.append(f"| {LABELS[method]} | {r['reached']}/8 | {r['consecutive_three_count']}/8 | {f(costs['optimization_s']['median'])} | {f(costs['available_elapsed_s']['median'])} | {f(costs['cold_available_s']['median'])} |")
    bands=['| 口径及目标 | 普通PINN 首次/确认/终点 | 锚定 首次/确认/终点 | 半Fourier 首次/确认/终点 |',
           '|---|---:|---:|---:|']
    for mode,label in [('point','原点权重'),('area','面积权重')]:
        for target in ['5','2','1']:
            cells=[]
            for method in METHODS:
                r=g[method]['targets'][mode][target]
                cells.append(f"{r['reached']}/{r['consecutive_three_count']}/{r['terminal_reached']}")
            bands.append(f"| {label} {target}% | "+' | '.join(cells)+' |')
    costs=['| 方法 | 全程净优化/秒 | 构建至解保存/秒 | 脚本启动至解保存/秒 | 分配/保留显存 MiB | 进程寿命峰值RSS MiB |',
           '|---|---:|---:|---:|---:|---:|']
    endpoints=['| 方法 | 点权重U/S % | 面积权重U/S % | 平均绝对应力误差 % | 洞壁U % | 水平/竖向相对位移误差 % |',
               '|---|---:|---:|---:|---:|---:|']
    queries=['| 方法 | 4096点驻GPU查询/ms | 4096点含传入传出/ms |', '|---|---:|---:|']
    for method in METHODS:
        c,r,e=g[method]['cost'],g[method]['resources'],g[method]['endpoint']
        costs.append(f"| {LABELS[method]} | {f(c['optimization_s']['median'])} | {f(c['observed_solution_s']['median'])} | {f(c['script_entry_to_solution_s']['median'])} | {f(r['allocated_MiB']['median'])}/{f(r['reserved_MiB']['median'])} | {f(r['peak_rss_lifetime_MiB']['median'])} |")
        endpoints.append(f"| {LABELS[method]} | {f(e['u_point_pct']['mean'],3)}/{f(e['point_stress_pct']['mean'],3)} | {f(e['u_area_pct']['mean'],3)}/{f(e['area_stress_pct']['mean'],3)} | {f(e['mean_absolute_error_pct']['mean'],3)} | {f(e['wall_vector_u_pct']['mean'],3)} | {f(e['horizontal_wall_error_pct']['mean'],3)}/{f(e['vertical_wall_error_pct']['mean'],3)} |")
        queries.append(f"| {LABELS[method]} | {ms(c['device_resident_batch_s']['median'])} | {ms(c['host_to_device_to_host_batch_s']['median'])} |")
    spatial='；'.join(f"相对{LABELS[other]}，锚定的较准岩体面积平均为{a['spatial']['anchored_vs_'+other]['better_area']['mean']:.2f}%，在{a['spatial']['anchored_vs_'+other]['majority_area_wins']}/8个种子中超过一半"
                     for other in ['vanilla_matched','fourier_half'])
    return f'''# C1普通PINN：共同精度、完整成本与空间误差补测

2026-09-22。三种原mixed PINN在作者明确确认持续空闲后，于同一连续窗口完成24次串行新进程测试，均未触发预定干扰规则。使用原C1、原种子41—48、原配点、损失与L-BFGS；三种模型可训练参数为110913、110705、110705。没有新增DEM实验，没有采用新方法。

实际共{a['accepted_steps']}个接受步、{a['closures']}次closure，全部{a['checkpoints']}个观测状态先冻结后作FEM评价。{a['checks_passed']}项交付检查通过。这里不与旧32次成本样本合并，也不覆盖82005参数旧普通基线。

## 共同精度的时间

主口径要求位移和三分量应力的点权重相对L2同时不超过目标；下面的时间为八种子中成功者的中位数，NR表示预算内未观测到达标。失败次数必须与时间并读，不对未达标方法计算有限的达标加速倍数。

{chr(10).join(primary)}

完整分档如下。每个数字均以8次为分母，依次是“首次观测达标／连续三次观测确认／终点达标”。面积口径在同一FEM积分点上对位移和应力均采用面积权重；位移由原单元形函数插值。与旧报告衔接的“点权重位移＋面积权重应力”另存于分析JSON，不混称为两者均面积加权。

{chr(10).join(bands)}

每50个接受步保存权重，另含初始与终止状态。各次首次观测区间及连续确认时间保留在逐运行JSON中；不按步数比例推算秒数。前后观测之间可能存在未观测到的跨越，三次确认也不证明后续永久达标。全部训练只按固定优化器停止条件终止，未用FEM选择训练步。

## 完整成本与资源

下表时间为全部8次的中位数。净优化包括所有试探closure和线搜索，扣除实际记录/检查点开销；“构建至解保存”包括模型与配点构建、初值核验、优化、全部观测保存及最终解写出。“脚本启动至解保存”进一步包括导入与统一CUDA预热。完整新进程启动至退出还含查询与事后记录，原始数值单列在JSON中。

{chr(10).join(costs)}

统一使用一块RTX 5080 Laptop GPU、float32、关闭TF32、确定性算法及4个Torch CPU线程。显存为求解窗口内分配器峰值；RSS是进程寿命峰值，不把它写成训练瞬时峰值。GPU驻留与主机传入/传出查询都使用同一4096点、五个原生输出；每次预热10批后各测50批，先在种子内取中位数，再汇总八个种子。

{chr(10).join(queries)}

## 空间误差与工程量

以下为八种子终点均值。平均绝对应力误差用同一FEM全域面积RMS应力归一化，避免用局部接近零的应力作分母。全分位数、近洞区域、较准面积和洞壁量均保留。

{chr(10).join(endpoints)}

{spatial}。结合全域误差与工程量判断，不用单个极值或单一L2遮盖主体分布。

## 对本轮返修的作用

本批直接补充R2.5关于普通PINN共同精度下真实时间的证据，并支持R1.4/R1.7中明确成本边界。锚定相对普通PINN的表示提升不等于锚定绑定的独立收益；严格偏置对照和半带宽Fourier强对照继续保留。原C1相对原带宽Fourier的限定优势仍引用原配对证据，本批没有重测原带宽Fourier。跨洞型共同自动规则与最终方法贡献仍需单独收敛。

可追溯数据位于 `efficiency_evaluation/C1_plain_gap/`：`protocol.json`、作者空闲确认、24次负载监测、`cohort_freeze.json`、逐检查点评价、`analysis.json`和`delivery_audit.json`。冻结科学代码和旧结果未改变，未安装软件或修改conda环境。
'''

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8'); main()
