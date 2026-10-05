"""Full versus reduced integration and same-point shape-reference convergence."""
from check_shape_references import *

FULL=BATCH/'full_integration'


def main():
    records=read(FULL/'mesh_manifest.json');prior=read(ROOT/'checks/shape_reference_verification.json')
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
        reference_preparation_complete=True,new_method_cross_shape_accuracy_verified=False,
        timing_valid_for_comparison=False,strict_true_error_bound=False,cases={},sources={})
    table=['| 洞型 | 比较（同一最细CPE8积分点） | U差/% | 面积加权S差/% | 距角点≥0.02的S差/% |',
           '|---|---|---:|---:|---:|']
    for geometry in ['square','slender']:
        rr=[r for r in records if r['geometry']==geometry]
        finest=rr[-1];fine=dict(np.load(FULL/(finest['job']+'_Load.npz')));mesh=dict(np.load(finest['mesh_file']))
        xy,w=fine['xy_s'],fine['volume'];fu,fs=evaluate(mesh,fine,xy)
        a,b=(.1,.1) if geometry=='square' else (.2,.015)
        corners=np.array([[-a,-b],[a,-b],[a,b],[-a,b]])
        away=np.min(np.linalg.norm(xy[:,None,:]-corners[None,:,:],axis=2),axis=1)>=.02
        checks=[];comparisons={}
        for rec in rr:
            job=rec['job'];path=FULL/(job+'_Load.npz');inp=Path(rec['input_file']);work=inp.parent
            assert sha(inp)==rec['input_sha256']
            reduced_inp=work.parent/'tust_shape_reference_20260921'/(job.replace('tsf_','tsr_')+'.inp')
            assert inp.read_text('ascii')==reduced_inp.read_text('ascii').replace('CPE8R','CPE8')
            assert 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in (work/(job+'.sta')).read_text()
            msg=(work/(job+'.msg')).read_text();dat=(work/(job+'.dat')).read_text()
            for phrase in ['WARNING MESSAGES DURING ANALYSIS','ERROR MESSAGES']:
                found=re.findall(r'(\d+)\s+'+phrase,msg);assert found and all(int(v)==0 for v in found)
            input_warnings=int(re.findall(r'(\d+)\s+WARNING MESSAGES DURING USER INPUT PROCESSING',msg)[-1])
            aspects=re.findall(r'\*\*\*WARNING: The aspect ratio for (\d+) elements exceeds 100 to 1',dat)
            assert input_warnings==len(aspects) and dat.count('***WARNING:')==len(aspects)
            data=dict(np.load(path));mm=dict(np.load(rec['mesh_file']));meta=read(FULL/(job+'.json'))
            assert np.array_equal(data['connectivity'],mm['connectivity']-1) and set(data['element_types'])=={'CPE8'}
            assert len(data['s'])==9*rec['elements'] and abs(data['volume'].sum()-(1-4*a*b))<1e-6
            assert (data['volume']>0).all() and np.linalg.norm(data['rf'].sum(0)-[0,5])<1e-6
            bottom=mm['xy'][:,1]==-.5;pin=bottom&(mm['xy'][:,0]==0)
            assert np.max(np.abs(data['u'][bottom,1]))<1e-12 and abs(data['u'][pin,0][0])<1e-12
            _,st=evaluate(mm,data,data['xy_s']);defect=relative(st,data['s'],data['volume']);assert defect<.02
            energies=meta['steps']['Load']['energies'];assert abs(energies['ALLSE']-energies['ALLWK'])<1e-6*energies['ALLSE']
            u,s=evaluate(mm,data,xy)
            comparison=dict(u_pct=relative(u,fu,w),s_pct=relative(s,fs,w),away_s_pct=relative(s[away],fs[away],w[away]))
            comparisons[f'full_g{rec["level"]}_vs_full_g3']=comparison
            checks.append(dict(job=job,level=rec['level'],nodes=rec['nodes'],elements=rec['elements'],stress_reconstruction_pct=defect,
                strain_energy=energies['ALLSE'],area=float(data['volume'].sum()),reaction=data['rf'].sum(0).tolist(),
                aspect_warning_elements=sum(map(int,aspects)),analysis_warnings=0))
            for file in [inp,work/(job+'.odb'),work/(job+'.sta'),work/(job+'.msg'),work/(job+'.dat'),path,FULL/(job+'.json')]:result['sources'][str(file.resolve())]=sha(file)
        for level in [2,3]:
            reduced=dict(np.load(BATCH/f'tsr_{geometry}_g{level}_Load.npz'))
            mr=dict(np.load(BATCH/f'tsr_{geometry}_g{level}_mesh.npz'))
            u,s=evaluate(mr,reduced,xy)
            comparisons[f'reduced_g{level}_vs_full_g3']=dict(u_pct=relative(u,fu,w),s_pct=relative(s,fs,w),away_s_pct=relative(s[away],fs[away],w[away]))
        for name,row in comparisons.items():
            if name=='full_g3_vs_full_g3':continue
            table.append(f"| {geometry} | {name} | {row['u_pct']:.6f} | {row['s_pct']:.6f} | {row['away_s_pct']:.6f} |")
        result['cases'][geometry]=dict(checks=checks,comparisons=comparisons,comparison_points=len(xy),
            energy_change_pct=100*abs(checks[-1]['strain_energy']-checks[0]['strain_energy'])/checks[-1]['strain_energy'],
            reference_npz=str((FULL/(finest['job']+'_Load.npz')).resolve()),reference_sha256=sha(FULL/(finest['job']+'_Load.npz')))
        print(json.dumps(dict(geometry=geometry,comparisons=comparisons),ensure_ascii=False),flush=True)
    for file in [Path(__file__),Path(__file__).with_name('export_full_shape_references.py'),Path(__file__).with_name('check_shape_references.py'),
        BATCH/'protocol.json',BATCH/'mesh_manifest.json',FULL/'protocol.json',FULL/'mesh_manifest.json',ROOT/'checks/shape_reference_verification.json']:
        result['sources'][str(file.resolve())]=sha(file)
    (ROOT/'checks/shape_full_integration_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    detail=f'''## 完全积分单元的独立复核

在约0.5%的应力加密差及长宽比警告出现后，对两种洞型的第2、3级网格各增加一次CPE8完全积分求解，共4项。输入文件逐字符核验，仅单元类型由CPE8R改为CPE8；实际几何、材料、载荷、支承和节点保持不变。候选神经/物理方法尚未在这些形状上评价。

{chr(10).join(table)}

full为CPE8完全积分，reduced为CPE8R。此表统一使用最细CPE8的9点高斯积分位置及面积权重，因此不能把前表在CPE8R积分点上的差值直接当作同一测量。每一级位移通过该级形函数插值，应力通过位移梯度重构；另行核查与其自身ODB应力一致性。两种单元的差异及网格差异都完整保留，不能只保留数值更小的那一组。

完全积分参考的最后一级U/S加密差：方形洞{result['cases']['square']['comparisons']['full_g2_vs_full_g3']['u_pct']:.6f}/{result['cases']['square']['comparisons']['full_g2_vs_full_g3']['s_pct']:.6f}%，狭长洞{result['cases']['slender']['comparisons']['full_g2_vs_full_g3']['u_pct']:.6f}/{result['cases']['slender']['comparisons']['full_g2_vs_full_g3']['s_pct']:.6f}%。它们是参考敏感性诊断，不是严格的真实误差上界；后续方法差异若接近这个尺度，需要进一步参考核验。

共12项Abaqus求解均完成，反力、面积、边界、本构重构、正积分权重与应变能/外功核查通过；已知的长宽比输入警告继续保留，没有求解过程警告或错误。这里完成的是跨洞型参考解准备，尚未证明候选方法在这两种洞型的精度与自动适应能力。
'''
    report=ROOT/'方形与狭长洞参考解.md';text=report.read_text('utf-8')
    text=text.split('## 完全积分单元的独立复核')[0].rstrip()+'\n\n'+detail
    report.write_text(text,encoding='utf-8')
    archive=ROOT/'checks/shape_reference_full_research.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for file in [Path(__file__),Path(__file__).with_name('build_shape_references.py'),Path(__file__).with_name('export_shape_references.py'),
            Path(__file__).with_name('check_shape_references.py'),Path(__file__).with_name('export_full_shape_references.py'),report,
            ROOT/'checks/shape_reference_verification.json',ROOT/'checks/shape_full_integration_verification.json']:
            z.write(file,file.name)
        for batch,label in [(BATCH,'reduced'),(FULL,'full')]:
            for name in ['protocol.json','mesh_manifest.json']:z.write(batch/name,label+'/'+name)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    (FULL/'completion.json').write_text(json.dumps(dict(passed=True,completed_abaqus_jobs=12,
        reference_preparation_complete=True,new_method_cross_shape_accuracy_verified=False,
        report_sha256=sha(report),research_archive_sha256=sha(archive),verification_sha256=sha(ROOT/'checks/shape_full_integration_verification.json'),
        timing_valid_for_comparison=False),indent=2),encoding='utf-8')


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
