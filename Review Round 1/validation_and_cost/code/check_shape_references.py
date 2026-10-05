"""Independent mesh-coordinate lookup, constitutive and refinement checks."""
import os
os.environ['MKL_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import sys,json,hashlib,zipfile,re
from datetime import datetime,timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BATCH=ROOT/'shape_references'
sys.path.insert(0,str(ROOT.parent/'controlled_pinn/code'))
from fem_interpolation import shape


def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def evaluate(mesh,data,points):
    x,y=mesh['x'],mesh['y'];u=data['u'];uu=np.zeros((len(points),2));ss=np.zeros((len(points),3))
    E,nu=1.333,.3333;lam=E*nu/((1+nu)*(1-2*nu));mu=E/(2*(1+nu))
    for start in range(0,len(points),32768):
        sl=slice(start,start+32768);xy=points[sl]
        i=np.searchsorted(x,xy[:,0],side='right')-1;j=np.searchsorted(y,xy[:,1],side='right')-1
        assert (i>=0).all() and (i<len(x)-1).all() and (j>=0).all() and (j<len(y)-1).all()
        eid=mesh['grid_element_ids'][i,j];assert (eid>0).all()
        dx,dy=x[i+1]-x[i],y[j+1]-y[j]
        rs=np.column_stack([2*(xy[:,0]-x[i])/dx-1,2*(xy[:,1]-y[j])/dy-1])
        assert np.max(np.abs(rs))<1+1e-10
        N,dN=shape('CPE8R',rs);dN[:,0]*=(2/dx)[:,None];dN[:,1]*=(2/dy)[:,None]
        un=u[mesh['connectivity'][eid-1]-1]
        gu=np.einsum('nki,nij->nkj',dN,un)
        uu[sl]=np.einsum('ni,nij->nj',N,un)
        xx,yy=gu[:,0,0],gu[:,1,1];tr=xx+yy
        ss[sl]=np.column_stack([lam*tr+2*mu*xx,lam*tr+2*mu*yy,mu*(gu[:,0,1]+gu[:,1,0])])
    return uu,ss


def relative(a,b,w):return float(100*np.sqrt(np.sum(w[:,None]*(a-b)**2)/np.sum(w[:,None]*b*b)))


def main():
    manifest=read(BATCH/'mesh_manifest.json');out=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        passed=True,reference_preparation_complete=True,new_method_cross_shape_accuracy_verified=False,
        nodal_precision='ODB values read with dataDouble where available; exact input coordinates used; displacement precision is checked through stress reconstruction.',
        metrics='Area-weighted relative L2 at the same finest-mesh integration points; all four meshes reconstructed from their own nodal displacements.',
        strict_true_error_bound=False,timing_valid_for_comparison=False,cases={},sources={})
    for geom in ['square','slender']:
        records=[r for r in manifest if r['geometry']==geom];finest=records[-1]
        fm=dict(np.load(finest['mesh_file']));fd=dict(np.load(BATCH/(finest['job']+'_Load.npz')))
        points,w=fd['xy_s'],fd['volume'];uf,sf=evaluate(fm,fd,points)
        a,b=(.1,.1) if geom=='square' else (.2,.015)
        corners=np.array([[-a,-b],[a,-b],[a,b],[-a,b]])
        distance=np.sqrt(np.min(np.sum((points[:,None,:]-corners[None,:,:])**2,axis=2),axis=1))
        away=distance>=.02
        rows=[];previous=None
        for rec in records:
            job=rec['job'];inp=Path(rec['input_file']);folder=inp.parent
            assert sha(inp)==rec['input_sha256'] and sha(rec['mesh_file'])==rec['mesh_sha256']
            assert 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in (folder/(job+'.sta')).read_text()
            msg=(folder/(job+'.msg')).read_text()
            for phrase in ['WARNING MESSAGES DURING ANALYSIS','ERROR MESSAGES']:
                found=re.findall(r'(\d+)\s+'+phrase,msg);assert found and all(int(v)==0 for v in found),(job,phrase,found)
            input_warnings=int(re.findall(r'(\d+)\s+WARNING MESSAGES DURING USER INPUT PROCESSING',msg)[-1])
            dat=(folder/(job+'.dat')).read_text()
            aspect_counts=re.findall(r'\*\*\*WARNING: The aspect ratio for (\d+) elements exceeds 100 to 1',dat)
            assert input_warnings==len(aspect_counts) and dat.count('***WARNING:')==len(aspect_counts)
            aspect_warning_elements=sum(map(int,aspect_counts))
            data=dict(np.load(BATCH/(job+'_Load.npz')));mesh=dict(np.load(rec['mesh_file']));meta=read(BATCH/(job+'.json'))
            assert np.array_equal(mesh['connectivity']-1,data['connectivity'])
            assert np.array_equal(mesh['xy'],data['xy_u']) and (data['volume']>0).all()
            assert abs(data['volume'].sum()-(1-4*a*b))<1e-6
            reaction=data['rf'].sum(0);assert np.linalg.norm(reaction-[0,5])<1e-6
            bottom=mesh['xy'][:,1]==-.5;pin=bottom&(mesh['xy'][:,0]==0)
            assert np.max(np.abs(data['u'][bottom,1]))<1e-12 and abs(data['u'][pin,0][0])<1e-12
            _,reconstructed=evaluate(mesh,data,data['xy_s'])
            defect=relative(reconstructed,data['s'],data['volume'])
            assert defect<.02,(job,'IP constitutive reconstruction percent',defect)
            uc,sc=evaluate(mesh,data,points)
            energy=meta['steps']['Load']['energies'];assert abs(energy['ALLSE']-energy['ALLWK'])<1e-6*energy['ALLSE']
            row=dict(job=job,level=rec['level'],nodes=rec['nodes'],elements=rec['elements'],area=float(data['volume'].sum()),
                input_aspect_ratio_warnings=input_warnings,aspect_warning_elements=aspect_warning_elements,analysis_warnings=0,
                summed_reaction=reaction.tolist(),strain_energy=energy['ALLSE'],work=energy['ALLWK'],
                stress_reconstruction_pct=defect,finest_common_points_u_pct=relative(uc,uf,w),
                finest_common_points_s_pct=relative(sc,sf,w),finest_odb_s_pct=relative(sc,fd['s'],w),
                away_from_corners_s_pct=relative(sc[away],sf[away],w[away]),
                corner_neighborhood_s_pct=relative(sc[~away],sf[~away],w[~away]))
            if previous is not None:
                row['consecutive_u_pct']=relative(previous[0],uc,w);row['consecutive_s_pct']=relative(previous[1],sc,w)
            rows.append(row);previous=(uc,sc)
            for file in [inp,folder/(job+'.odb'),folder/(job+'.sta'),folder/(job+'.msg'),folder/(job+'.dat'),
                         Path(rec['mesh_file']),BATCH/(job+'_Load.npz'),BATCH/(job+'.json')]:out['sources'][str(file.resolve())]=sha(file)
        out['cases'][geom]=dict(meshes=rows,comparison_points=len(points),last_refinement=rows[-1],
            finest_odb_stress_reconstruction_pct=relative(sf,fd['s'],w),
            strain_energy_last_relative_pct=100*abs(rows[-1]['strain_energy']-rows[-2]['strain_energy'])/rows[-1]['strain_energy'],
            corner_distance_for_split=.02,
            convergence='Mesh differences quantify reference sensitivity, not rigorous true-solution error bounds. Pointwise corner peak stress is not a convergence criterion.')
        print(json.dumps(dict(geometry=geom,last=rows[-1],energy_change_pct=out['cases'][geom]['strain_energy_last_relative_pct']),ensure_ascii=False),flush=True)
    source_names=['build_shape_references.py','export_shape_references.py','check_shape_references.py']
    for n in source_names:out['sources'][str(Path(__file__).with_name(n).resolve())]=sha(Path(__file__).with_name(n))
    out['sources'][str((ROOT.parent/'controlled_pinn/code/fem_interpolation.py').resolve())]=sha(ROOT.parent/'controlled_pinn/code/fem_interpolation.py')
    (ROOT/'checks/shape_reference_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=['| 洞型 | 层级 | 单元数 | 与最细网格U差/% | 与最细网格S差/% | 距角点至少0.02的S差/% | 应变能 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,c in out['cases'].items():
        for r in c['meshes']:
            rows.append(f"| {name} | {r['level']} | {r['elements']} | {r['finest_common_points_u_pct']:.6f} | {r['finest_common_points_s_pct']:.6f} | {r['away_from_corners_s_pct']:.6f} | {r['strain_energy']:.8f} |")
    details=[]
    for name,c in out['cases'].items():
        last=c['last_refinement'];details.append(f"{name}最后一次加密的U/S差为{last['consecutive_u_pct']:.6f}/{last['consecutive_s_pct']:.6f}%，应变能相对变化{c['strain_energy_last_relative_pct']:.6f}%；最细网格位移重构应力与ODB积分点应力差为{c['finest_odb_stress_reconstruction_pct']:.6f}%。")
    report=f'''# 方形洞与狭长洞：Abaqus参考解及收敛检查

两种洞型各四级嵌套网格、共8个线弹性平面应变求解及积分点结果提取完成。它们补齐实际荷载的参考数据；当前尚未评价任何候选方法在这两种洞型上的精度，不能称为跨洞型自动适应已经通过。

外边界为边长1的正方形。方洞半尺寸0.1×0.1；狭长矩形洞半尺寸0.2×0.015，长宽比约13.33。弹性模量1.333、泊松比0.3333，两侧压力1、顶部压力5，洞壁自由；底部法向位移为零、底中点水平位移为零。所有结果采用小应变、几何线性设置，单位厚度。

两种洞型共用同一分段余弦加密张量网格规则，逐级把每段单元数加倍，使用CPE8R。几何边界精确对齐；全部单元面积及积分权重为正。最细方洞153600单元，狭长洞189440单元；最大长宽比分别约101.86和203.74。Abaqus对最细方洞400个、最细狭长洞1128个单元给出长宽比超过100的输入警告，完整计数保留在检查JSON中；没有求解过程警告或错误。这里不忽略网格警告，以实际位移、应力和能量收敛量化其影响；仍不能把网格间差异当作真实误差界。

{chr(10).join(rows)}

{chr(10).join(details)}

所有误差采用各洞型相同的最细网格积分点与面积权重。位移由各自网格的形函数插值，应力由位移梯度及相同本构关系重构；另与实际ODB积分点应力交叉验证。重构使用输入网格的双精度坐标，并核查其与ODB坐标的舍入差，避免细小单元的坐标舍入放大梯度误差。0.02角点排除半径只用于附加诊断，全域指标仍包括这些邻域。

八项反力总和均核对为(0,5)，底部与规约位移满足边界条件，应变能与外功一致。网格间差值是参考解分辨率诊断，不是真实误差的严格上界；方角附近的峰值应力不用于宣称收敛。候选方法的最终判优尺度应与这些参考解差异一起判断。

原始网格、ODB导出的积分点数据、输入文件及哈希均已保留。此次参考解使用已安装Abaqus并通过MCP提交，未安装软件或调整conda环境；与神经训练并行，任何耗时均不用于方法效率比较。

下一步仍需在不使用这些参考场进行选点、调参或停止的条件下，按统一规则生成物理与神经表示，冻结候选终点后再评价；目前仅完成参考解准备。
'''
    (ROOT/'方形与狭长洞参考解.md').write_text(report,encoding='utf-8')
    archive=ROOT/'checks/shape_reference_research.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for n in source_names:z.write(Path(__file__).with_name(n),n)
        for p in [ROOT/'方形与狭长洞参考解.md',ROOT/'checks/shape_reference_verification.json',BATCH/'protocol.json',BATCH/'mesh_manifest.json']:z.write(p,p.name)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    (BATCH/'completion.json').write_text(json.dumps(dict(passed=True,reference_preparation_complete=True,
        new_method_cross_shape_accuracy_verified=False,completed_abaqus_jobs=8,research_archive_sha256=sha(archive),
        report_sha256=sha(ROOT/'方形与狭长洞参考解.md'),verification_sha256=sha(ROOT/'checks/shape_reference_verification.json'),
        timing_valid_for_comparison=False),indent=2),encoding='utf-8')


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
