"""Audit pilot reliability and fixed-checkpoint observations before wider coverage."""
from run_stable_dem import *


def main():
    m=initialize();p=read(BATCH/'protocol.json');assert m['status']=='complete'
    summary=read(ROOT/'summary/stable_dem_pilot.json');assert sha(ROOT/'summary/stable_dem_pilot.json')==m['summary_sha256']
    curves=read(ROOT/'summary/stable_dem_checkpoint_diagnostics.json')
    rows=['| 工况与种子 | 稳定位移U/% | 稳定应力S/% | 原DEM终点S/% | 总接受更新/保留更新 | closure | 加密次数 | 终点S/最低已存检查点S |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    records=[];all_steps=all_retained=all_closures=0
    for config in p['jobs']:
        name=config['id'];folder=BATCH/name;e=summary[name];r=read(folder/'result.json');history=read(folder/'history.json')
        assert sha(folder/'full_evaluation.json')==m['evaluation_sha256'][name]
        for file,h in m['endpoint_sha256'][name].items():assert sha(folder/file)==h
        assert e['evaluated_utc']>m['all_endpoints_frozen_utc'] and curves[name]['evaluated_utc']>m['all_endpoints_frozen_utc']
        assert e['endpoint_sha256']==m['endpoint_sha256'][name] and curves[name]['endpoint_sha256']==m['endpoint_sha256'][name]
        for path,h in curves[name]['checkpoint_sha256'].items():assert sha(path)==h
        assert r['attempted_accepted_steps']==sum(x['block_steps'] for x in history)
        assert r['closure_evaluations']==sum(x['closure_evaluations'] for x in history)
        assert r['retained_accepted_steps']==sum(x['block_steps'] for x in history if x['audit']['passed'])
        assert not r['selection_uses_fem'] and not r['timing_valid_for_comparison']
        ratio=curves[name]['descriptive_only']['s']['endpoint_to_lowest_ratio'];metrics=e['metrics_percent']
        passing=bool(r['sampling_verified'] and r['stop_reason'] not in ['numerical_quadrature_budget','rollback_state_unresolved'])
        records.append(dict(run=name,pilot_numerically_reliable=passing,metrics_percent=metrics,
            attempted_steps=r['attempted_accepted_steps'],retained_steps=r['retained_accepted_steps'],closures=r['closure_evaluations'],
            promotions=r['promotions'],stop_reason=r['stop_reason'],endpoint_to_lowest_observed_stress_ratio=ratio,
            numerical_audits=[dict(attempted_steps=x['attempted_steps'],base_depth=x['base_depth'],passed=x['audit']['passed'],
                component_difference=x['audit']['component_difference'],component_tolerance=x['audit']['component_tolerance'],
                gradient_relative_difference=x['audit']['gradient_relative_difference'],potential=x['audit']['final']['potential_energy']) for x in history]))
        all_steps+=r['attempted_accepted_steps'];all_retained+=r['retained_accepted_steps'];all_closures+=r['closure_evaluations']
        rows.append(f"| {name} | {metrics['u']:.6f} | {metrics['s']:.6f} | {e['original_terminal_metrics']['s_vector_pct']:.6f} | {r['attempted_accepted_steps']}/{r['retained_accepted_steps']} | {r['closure_evaluations']} | {len(r['promotions'])} | {ratio:.6f} |")
    reliable=all(x['pilot_numerically_reliable'] for x in records)
    checkpoint_count=sum(len(v['records']) for v in curves.values())
    endpoint_stress_lowest_count=sum(v['descriptive_only']['s']['endpoint_to_lowest_ratio']<=1.+1e-12 for v in curves.values())
    largest_u_ratio=max(v['descriptive_only']['u']['endpoint_to_lowest_ratio'] for v in curves.values())
    decision=('按同一冻结算法覆盖原46组DEM工况/种子；本批5组原样复用，新增其余41组，不仅补救旧失效样本。'
              if reliable else '本批尚未通过预设数值可靠性门槛，先定位未解决的问题，不启动46组覆盖。')
    analysis=dict(records=records,pilot_numerically_reliable=reliable,total_attempted_accepted_steps=all_steps,
        total_retained_accepted_steps=all_retained,total_closure_evaluations=all_closures,
        decision=decision,independent_confirmation=False,timing_valid_for_comparison=False,
        original_architecture_parameters=81402,all_endpoints_frozen_before_fem=True,selection_uses_fem=False,
        source_archive_sha256=m['source_archive_sha256'],protocol_sha256=m['protocol_sha256'],
        checkpoint_evaluations=checkpoint_count,endpoint_stress_lowest_observed_count=endpoint_stress_lowest_count,
        largest_endpoint_to_lowest_observed_u_ratio=largest_u_ratio)
    write(ROOT/'summary/stable_dem_pilot_analysis.json',analysis)
    report=f'''# DEM稳定化开发验证

5组预设开发运行已全部完成训练、独立数值审计、全参考点终点评价及固定检查点诊断。预设数值可靠性门槛：{'通过' if reliable else '未通过'}。下一步：{decision}

{chr(10).join(rows)}

本批沿用原DEM的2–200–200–200–2网络、81402个参数、位移边界构造及原CPU float32初值。计算改为float64，采用贴合圆孔的复合Gauss积分、独立内能/外力功和参数梯度核验，每50步重置L-BFGS历史。它是一组数值稳定化改动，不能把误差变化单独归因于某一项，也不能称为原DEM可能达到的最优性能。

每次训练最多1000个优化器接受更新，失败块的更新同样计入预算。失败时回退至此前通过数值核验的场，加密积分后继续；FEM不参与选步、积分加密或停止。全批共{all_steps}个接受更新、{all_retained}个保留更新、{all_closures}次closure。积分及独立验证开销另有完整记录；本轮未做正式计时，不能按步数宣称等成本。

终点误差采用原全部参考点的U/S定义；近洞应力、面积加权应力与完整数据见JSON。旧DEM灾难性终点仅用于追溯被修复的数值问题，不能拿旧大误差代表可信DEM的一般性能。本批各荷载只有1–2个种子，不据此替换原全部46次统计。

共评价{checkpoint_count}份已存初始、每百步附近接受块与失败块权重，均在全部终点冻结之后完成。{endpoint_stress_lowest_count}/5组的终点应力误差达到各自已存接受检查点中的最低值；位移却仍有波动，最大的终点/最低已存误差比为{largest_u_ratio:.6f}（约高{100*(largest_u_ratio-1):.2f}%）。这区分了灾难性积分失真与个别误差指标不单调两种现象。

表中最低已存检查点仅为事后诊断，绝不用于正式终点选择。它与连续每步的真实最小值不同；位移和应力也可能在不同步达到最小。数值积分通过而误差略有波动，不能一概称为DEM失效。

本批只验证原圆孔DEM基线。它不证明方洞、狭长洞的新结合框架已经自动获得足够精度，也不补足锚点绑定的独立贡献。原C1与强Fourier对照继续保留各自协议和结论。

[冻结协议](<{(BATCH/'protocol.json').as_posix()}>)；[完整终点](<{(ROOT/'summary/stable_dem_pilot.json').as_posix()}>)；[固定检查点](<{(ROOT/'summary/stable_dem_checkpoint_diagnostics.json').as_posix()}>)；[数值审计及取舍](<{(ROOT/'summary/stable_dem_pilot_analysis.json').as_posix()}>)。
'''
    path=ROOT/'DEM稳定化开发验证.md';path.write_text(report,encoding='utf-8')
    legacy_record=read(ROOT.parent/'controlled_pinn/checks/stage3_completion.json')
    for group,directory in [('source_code_sha256','code'),('summary_sha256','summary'),('run_result_sha256','runs')]:
        for name,h in legacy_record[group].items():
            target=ROOT.parent/'controlled_pinn'/directory/name
            if directory=='runs':target=target/'result.json'
            assert sha(target)==h,target
    for name,h in legacy_record['submitted_originals_unchanged'].items():assert sha(ROOT.parents[1]/'再次提交TUST'/name)==h
    record=ROOT/'研究记录.md';text=record.read_text('utf-8')
    old='本轮已完成已学表示转入完整空间的12个线性重解及对应对照。'
    new=f'当前已完成5组DEM稳定化开发验证；数值可靠性门槛：'+('通过' if reliable else '未通过')+'。'+decision
    text=text.replace(old,new,1)
    text=text.replace('## 最新已学表示重解','## 最新DEM稳定化\n\n'+new+'\n\n[完整验证](<'+path.as_posix()+'>)。原46组统计尚未全部由稳定化配置覆盖，未进行正式计时。\n\n## 前序已学表示重解',1)
    text=text.replace('DEM晚期退化仍需独立稳定化验证，受检的混合能量训练不能代替原DEM修复。','DEM已完成5组稳定化开发验证，仍需按同一规则覆盖原全部46组；受检的混合能量训练不能代替原DEM修复。')
    record.write_text(text,encoding='utf-8')
    progress_file=ROOT/'进度.json';current=read(progress_file)
    current.update(updated_utc=utc(),current_jobs=[],milestone='stable_dem_pilot_complete',stable_dem_pilot_runs=5,
        stable_dem_pilot_numerically_reliable=reliable,stable_dem_46_coverage_complete=False,
        full_fem_evaluation_complete=True,ready_to_adopt_final_method=False,new_timing_comparison_performed=False,
        this_round_new_training_runs=5,this_round_new_field_endpoints=5,this_round_full_evaluations=5,
        detail_report_sha256=sha(path),report_sha256=sha(record))
    for name in ['stable_dem_pilot.json','stable_dem_pilot_analysis.json','stable_dem_checkpoint_diagnostics.json']:
        current['summary_sha256'][name]=sha(ROOT/'summary'/name)
    write(progress_file,current)
    complete=dict(passed=True,completed_utc=utc(),pilot_numerically_reliable=reliable,
        source_archive_sha256=m['source_archive_sha256'],report_sha256=sha(path),record_sha256=sha(record),progress_sha256=sha(progress_file),
        legacy350_preserved=True,submitted_files_preserved=True,final_method_adopted=False,timing_valid_for_comparison=False)
    write(BATCH/'completion.json',complete);write(ROOT/'checks/stable_dem_pilot_completion.json',complete)
    print(json.dumps(complete,ensure_ascii=False),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
