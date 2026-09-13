"""只读的结构假设诊断程序，不是新的干燥预测或质量平衡模型。

条件性质量检验将题给密度关系解释为湿表观密度，并假定长度固定为0.25米。
若条件性质量不闭合，只说明这些解释不能同时成立；
所得表观质量不能作为实测质量或预测质量报告。"""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np
import openpyxl

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
# 按题给经验关系计算密度；真实湿表观密度解释仅用于条件性相容诊断
def rho(q,c):return (650+128*c) if q==3 else (760+90*c)
# ==================== 条件性干物质与水质量诊断 ====================
def diagnostic(q,data):
    state=data['final'];c=np.asarray(state['C']);x=np.asarray(state['x'])
    # 以圆环控制体权重积分含水率相关密度，并采用固定长度的条件性解释
    edges=np.r_[0.,(x[1:]+x[:-1])/2,1.]
    weights=np.diff(edges**2)/2
    radius=state['radius_cm']/100
    # 由干基含水率关系换算条件性干物质密度，不能冒充实测质量
    dry=rho(q,c)/(1+c)
    scale=2*math.pi*.25*radius**2
    # 比较初始均匀状态与保存终点的条件性质量，识别假设不相容
    initial=math.pi*.25*.02**2*rho(q,2.55)/3.55
    apparent=scale*float(weights@dry)
    water=scale*float(weights@(dry*c))
    hours=state['t']/3600
    return {'endpoint_h':hours,'unobserved_environment_hours':hours-4,
      'unobserved_environment_fraction':(hours-4)/hours,
      'conditional_initial_dry_mass_kg':initial,
      'conditional_final_dry_mass_kg':apparent,
      'conditional_final_water_mass_kg':water,
      'conditional_dry_mass_ratio':apparent/initial,
      'conditional_dry_mass_drift_percent':100*(apparent/initial-1),
      'conditional_endpoint_bound':{
        'type':'lower' if q==3 else 'upper',
        'dry_mass_ratio':(rho(3,.15)/1.15)/(rho(3,2.55)/3.55) if q==3 else
          (radius/.02)**2*rho(4,0)/(rho(4,2.55)/3.55)},
      'radius_cm':state['radius_cm'],'mean_dry_basis_C':float(2*weights@c),
      'volume_ratio_under_fixed_length':(radius/.02)**2}
# ==================== 环境观测覆盖与情景分析 ====================
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    source=ROOT/'model.py'
    inputs=[ROOT.parent/'附件'/f'附件{i}.xlsx' for i in (1,2)]
    wb=openpyxl.load_workbook(inputs[0],read_only=True,data_only=True)
    rows=list(wb.active.values);wb.close()
    air=np.asarray(rows[1:],dtype=float)
    assert air.shape==(241,3) and air[-1,0]==14400
    # 末小时均值只概括已观测阶段；不推断观测范围外的环境稳定性
    last=air[-1,1:];last_hour=air[air[:,0]>=10800,1:].mean(axis=0)
    cases={};run_hashes={}
    # 从既有终点读取条件性诊断所需状态，核对其源码及输入指纹
    for q in (3,4):
        p=ROOT/f'p2/runs/q{q}_h1_hm1.json'
        d=json.loads(p.read_text(encoding='utf-8'))
        assert d['signature']['source_sha256']['model.py']==sha(source)
        for file in inputs:assert d['signature']['input_sha256'][file.name]==sha(file)
        cases[str(q)]=diagnostic(q,d);run_hashes[p.name]=sha(p)
    # 汇总3×3有限情景的时长范围，区分换热与传质单独扰动
    scans={}
    for q in (3,4):
        matrix=[]
        for h in (.5,1.,2.):
            row=[]
            for m in (.5,1.,2.):
                p=ROOT/f'p2/runs/q{q}_h{h:g}_hm{m:g}.json'
                d=json.loads(p.read_text(encoding='utf-8'))
                assert d['status']=='endpoint_reached'
                assert d['signature']['source_sha256']['model.py']==sha(source)
                for file in inputs:assert d['signature']['input_sha256'][file.name]==sha(file)
                row.append(d['final']['t']/3600);run_hashes[p.name]=sha(p)
            matrix.append(row)
        a=np.asarray(matrix)
        scans[str(q)]={'hours':matrix,'min_h':float(a.min()),'max_h':float(a.max()),
          'heat_only_range_h':float(np.ptp(a[:,1])), 'mass_only_range_h':float(np.ptp(a[1,:])),
          'interpretation':'deterministic scenarios; no probability model or confidence coverage'}
    # ==================== 假设审查结果输出 ====================
    out={'status':'diagnostics_completed_not_physical_validation',
      'assumptions':['rho(C) conditionally treated as wet bulk density','fixed length 0.25 m',
        'nodal annular quadrature','no new time integration or result overwrite'],
      'environment':{'last_observed_s':14400,'last_values':last.tolist(),
        'last_hour_61_point_mean':last_hour.tolist(),
        'alternative_extension_computed':False},
      'cases':cases,'scans':scans,
      'q3_minus_q4_h':cases['3']['endpoint_h']-cases['4']['endpoint_h'],
      'attribution':'Both material laws and geometry differ; difference is not a pure shrinkage effect.',
      'source_sha256':sha(Path(__file__)), 'model_sha256':sha(source),
      'input_sha256':{p.name:sha(p) for p in inputs},'evidence_sha256':run_hashes}
    # 只写出诊断报告，不重算干燥轨迹或覆盖原结果
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
