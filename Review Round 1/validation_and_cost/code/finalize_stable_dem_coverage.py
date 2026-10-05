"""Complete baseline coverage audit and per-case, paired descriptive statistics."""
from run_stable_dem_coverage import *
import numpy as np
import torch


def stats(values):
    x=np.asarray(values,dtype=float)
    return dict(n=len(x),mean=float(x.mean()),sample_sd=float(x.std(ddof=1)),minimum=float(x.min()),maximum=float(x.max()))


def main():
    m=initialize();p=read(BATCH/'protocol.json');assert m['status']=='complete' and len(m['completed'])==46
    summary=read(ROOT/'summary/stable_dem_coverage.json');assert sha(ROOT/'summary/stable_dem_coverage.json')==m['summary_sha256']
    original=read(ROOT.parent/'controlled_pinn/config/run_manifest.json');source_inputs={}
    rows=['| 工况 | 种子数 | 稳定DEM U均值±SD/% | 稳定DEM S均值±SD/% | 原锚定S/% | 原Fourier S/% | 半带宽Fourier S/% | 数值通过 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    cases={};paired={};run_audits=[];rng=np.random.default_rng(20261108)
    for config in p['jobs']:
        name=config['id'];folder=BATCH/name;e=summary[name];r=read(folder/'result.json');history=read(folder/'history.json')
        for filename,h in m['endpoint_sha256'][name].items():assert sha(folder/filename)==h
        assert sha(folder/'full_evaluation.json')==m['evaluation_sha256'][name]
        assert r['attempted_accepted_steps']==sum(x['block_steps'] for x in history)
        assert r['closure_evaluations']==sum(x['closure_evaluations'] for x in history)
        assert r['retained_accepted_steps']==sum(x['block_steps'] for x in history if x['audit']['passed'])
        initial_folder=PILOT/name if name in m['reused'] else folder
        first=torch.load(initial_folder/'initial.pt',map_location='cpu',weights_only=True)['state_dict']
        original_checkpoint=ROOT.parent/'controlled_pinn/runs'/name/'step_00000.pt'
        initial_reference='same_run_checkpoint'
        if not original_checkpoint.exists():
            original_checkpoint=ROOT.parent/'controlled_pinn/runs'/f"C1_dem_s{config['seed']}"/'step_00000.pt'
            initial_reference='same_seed_C1_checkpoint_load_independent_initialization'
        old=torch.load(original_checkpoint,map_location='cpu',weights_only=True)['state_dict']
        assert all(torch.equal(v.float(),old[k]) for k,v in first.items())
        source_inputs[str(original_checkpoint)]=sha(original_checkpoint)
        if name not in m['reused']:assert e['evaluated_utc']>m['all_new_endpoints_frozen_utc']
        reliable=bool(r['sampling_verified'] and r['stop_reason'] not in ['numerical_quadrature_budget','rollback_state_unresolved'])
        run_audits.append(dict(run=name,case=config['case'],seed=config['seed'],reused_from_pilot=name in m['reused'],
            numerical_reliability_passed=reliable,attempted_steps=r['attempted_accepted_steps'],retained_steps=r['retained_accepted_steps'],
            closure_evaluations=r['closure_evaluations'],stop_reason=r['stop_reason'],final_base_depth=r['final_base_depth'],
            promotions=r['promotions'],final_audit=r['final_audit'],original_initialization_preserved=True,
            initial_reference=initial_reference,initial_reference_path=str(original_checkpoint)))
    for case in sorted({c['case'] for c in p['jobs']}):
        group=sorted([v for v in summary.values() if v['configuration']['case']==case],key=lambda e:e['configuration']['seed'])
        seeds=[v['configuration']['seed'] for v in group];case_stats={key:stats([v['metrics_percent'][key] for v in group]) for key in ['u','s','s_area','s_near']}
        methods={};paired[case]={}
        for method in ['anchored','fourier','fourier_half','vanilla','vanilla_matched','xpinn']:
            configs=[next((c for c in original if c['case']==case and c['seed']==seed and c['method']==method),None) for seed in seeds]
            if any(c is None for c in configs):continue
            baseline=[]
            for c in configs:
                path=ROOT.parent/'controlled_pinn/runs'/c['id']/'result.json';old=read(path)
                source_inputs[str(path)]=sha(path);baseline.append(old)
            methods[method]={key:stats([e['metrics'][legacy_key] for e in baseline]) for key,legacy_key in [('u','u_vector_pct'),('s','s_vector_pct'),('s_near','s_near_pct')]}
            paired[case][method]={}
            for key,legacy_key in [('u','u_vector_pct'),('s','s_vector_pct'),('s_near','s_near_pct')]:
                delta=np.array([e['metrics_percent'][key]-old['metrics'][legacy_key] for e,old in zip(group,baseline)])
                boot=delta[rng.integers(0,len(delta),(20000,len(delta)))].mean(axis=1)
                paired[case][method][key]=dict(difference='stable_DEM_minus_existing_method_percentage_points',seeds=seeds,
                    raw_differences=delta.tolist(),mean=float(delta.mean()),ci95_percentile_bootstrap=np.quantile(boot,[.025,.975]).tolist(),
                    stable_dem_lower_count=int((delta<0).sum()),n=len(delta),multiplicity_adjusted=False)
        passed=sum(x['numerical_reliability_passed'] for x in run_audits if x['case']==case)
        cases[case]=dict(seeds=seeds,stable_dem=case_stats,existing_methods=methods,numerically_reliable=passed)
        fmt=lambda x:f"{x['mean']:.6f}±{x['sample_sd']:.6f}"
        half=fmt(methods['fourier_half']['s']) if 'fourier_half' in methods else '未在本工况运行'
        rows.append(f"| {case} | {len(seeds)} | {fmt(case_stats['u'])} | {fmt(case_stats['s'])} | {fmt(methods['anchored']['s'])} | {fmt(methods['fourier']['s'])} | {half} | {passed}/{len(seeds)} |")
    passed_count=sum(x['numerical_reliability_passed'] for x in run_audits)
    counts={key:sum(x[key] for x in run_audits) for key in ['attempted_steps','retained_steps','closure_evaluations']}
    new_counts={key:sum(x[key] for x in run_audits if not x['reused_from_pilot']) for key in counts}
    analysis=dict(cases=cases,paired_differences=paired,run_audits=run_audits,numerically_reliable_count=passed_count,
        total_runs=46,new_runs=41,reused_pilot_runs=5,total_work=counts,new_work=new_counts,
        all_initializations_preserved=True,all41_new_endpoints_frozen_before_new_fem=True,
        independent_confirmation=False,paired_bootstrap_replicates=20000,paired_bootstrap_seed=20261108,
        selection_uses_fem=False,timing_valid_for_comparison=False,source_inputs=source_inputs)
    write(ROOT/'summary/stable_dem_coverage_analysis.json',analysis)
    status=('46组均通过预设数值可靠性核验，原DEM的统一稳定化覆盖完成。' if passed_count==46 else f'{passed_count}/46组通过预设数值可靠性核验，其余停止及原因必须单列；统一可靠性尚未全部建立。')
    report=f'''# 原46组DEM的统一稳定化核验

{status} 5组开发结果原样复用，本批新增41组；全部使用同一冻结算法、容差和1000个接受更新预算。没有按旧误差筛选重训对象，也没有用FEM最低误差挑终点。

{chr(10).join(rows)}

表中均值与样本标准差按每个工况独立计算。U/S沿用原全参考点误差口径，近洞与面积加权应力另见分析JSON。原锚定、Fourier、普通网络与XPINN结果原样保留；新旧统计不混成同一种DEM配置。各方法原训练预算和架构不同，这张表是各自声明预算下的终点比较，不是等时间或等成本比较。原C1锚定相对原Fourier的优势与半带宽Fourier更强的结果均未改变。

原网络、位移边界构造和46份新初始参数均经核对：有原初始检查点的16组直接比较；其余30组按原代码中与荷载无关的种子初始化规则，对照同种子的C1初始权重。稳定化包括float64、贴合几何的积分、独立内能/外力功及参数梯度检查、块间L-BFGS历史重置。出现检查失败便回退、加密；失败块仍计入训练预算。全46组共{counts['attempted_steps']}个接受更新、{counts['retained_steps']}个保留更新、{counts['closure_evaluations']}次closure；其中本批新增部分分别为{new_counts['attempted_steps']}、{new_counts['retained_steps']}、{new_counts['closure_evaluations']}。积分点数、验证工作和停止原因逐次保留，不能仅用参数量或步数推断效率。

开发阶段的5组固定检查点诊断另列，剩余41组先整体冻结再作新FEM评价。数值检查通过表明在本批审计标准下没有未解决的积分失真，不等于有限参考下误差会逐步单调下降，也不等于已得到DEM最优精度。不同荷载、先前开发种子及不同方法的配对差仅为描述性比较；JSON中的bootstrap区间未作多重比较校正。

该批补足的是原圆孔DEM基线的可信度。它不建立新结合框架的跨洞型精度或锚点绑定的独立优势。当前结合配置按既定停止条件不再扩展，后续应将可信基线、原C1限定优势、强Fourier和严格消融一并纳入返修取舍，不能继续以旧DEM灾难性终点夸大本文优势。

正式效率比较尚未进行；需要单独通知作者并协调电脑空闲窗口。正文标红修改、图表定稿与逐条回复仍未完成，本报告不是提交文件。

[冻结协议](<{(BATCH/'protocol.json').as_posix()}>)；[46个终点](<{(ROOT/'summary/stable_dem_coverage.json').as_posix()}>)；[全部指标、配对差与工作量](<{(ROOT/'summary/stable_dem_coverage_analysis.json').as_posix()}>)；[5组开发与轨迹诊断](<{(ROOT/'DEM稳定化开发验证.md').as_posix()}>)。
'''
    report_path=ROOT/'DEM全部工况稳定化核验.md';report_path.write_text(report,encoding='utf-8')
    legacy_record=read(ROOT.parent/'controlled_pinn/checks/stage3_completion.json')
    for group,directory in [('source_code_sha256','code'),('summary_sha256','summary'),('run_result_sha256','runs')]:
        for name,h in legacy_record[group].items():
            target=ROOT.parent/'controlled_pinn'/directory/name
            if directory=='runs':target=target/'result.json'
            assert sha(target)==h,target
    for name,h in legacy_record['submitted_originals_unchanged'].items():assert sha(ROOT.parents[1]/'再次提交TUST'/name)==h
    record_path=ROOT/'研究记录.md';text=record_path.read_text('utf-8');start=text.index('## 最新DEM稳定化')
    older=text[start:].replace('## 最新DEM稳定化','## 前序5组DEM开发',1)
    older=older.replace('原46组统计尚未全部由稳定化配置覆盖，未进行正式计时。','该段为开发阶段结论；46组完整核验见顶部，仍未进行正式计时。')
    older=older.replace('DEM已完成5组稳定化开发验证，仍需按同一规则覆盖原全部46组；','DEM已完成全部46组统一稳定化覆盖，数值可靠性结论见顶部；')
    text='# 自适应扩展研究记录\n\n'+status+'\n\n## 最新46组DEM稳定化覆盖\n\n'+status+' 5组复用、41组新增，完整统计及剩余工作见[报告](<'+report_path.as_posix()+'>)。未做正式计时，正文与回复尚未定稿。\n\n'+older
    record_path.write_text(text,encoding='utf-8')
    state=read(ROOT/'进度.json');state.update(updated_utc=utc(),current_jobs=[],milestone='stable_dem_coverage_complete',
        stable_dem_46_coverage_complete=True,stable_dem_coverage_completed=46,stable_dem_numerically_reliable_count=passed_count,
        stable_dem_coverage_new_runs=41,stable_dem_coverage_reused_runs=5,full_fem_evaluation_complete=True,
        this_round_new_training_runs=41,this_round_new_field_endpoints=41,this_round_full_evaluations=41,
        ready_to_adopt_final_method=False,new_timing_comparison_performed=False,
        report_sha256=sha(record_path),detail_report_sha256=sha(report_path))
    for name in ['stable_dem_coverage.json','stable_dem_coverage_analysis.json']:state['summary_sha256'][name]=sha(ROOT/'summary'/name)
    write(ROOT/'进度.json',state)
    complete=dict(passed=True,completed_utc=utc(),numerically_reliable_count=passed_count,total_runs=46,new_runs=41,reused_runs=5,
        legacy350_preserved=True,submitted_files_preserved=True,source_archive_sha256=m['source_archive_sha256'],
        report_sha256=sha(report_path),record_sha256=sha(record_path),progress_sha256=sha(ROOT/'进度.json'),timing_valid_for_comparison=False)
    write(BATCH/'completion.json',complete);write(ROOT/'checks/stable_dem_coverage_completion.json',complete)
    print(json.dumps(complete,ensure_ascii=False),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=2):main()
