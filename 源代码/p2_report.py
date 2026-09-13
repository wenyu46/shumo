"""汇总P2数值证据；算例未完成或检验未通过时不得声明最终验收通过。"""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
# ==================== 数值检验基准与算例合同 ====================
BASE={3:205810.53125,4:182967.919921875}
EXPECTED=[f'q{q}_{suffix}' for q in (3,4) for suffix in
          ['tight','reference']+[f'h{h:g}_hm{m:g}' for h in (.5,1.,2.) for m in (.5,1.,2.)]]

# ==================== 数值证据回读与一致性检查 ====================
def main():
    runs={p.stem:json.loads(p.read_text(encoding='utf-8')) for p in (ROOT/'p2/runs').glob('*.json')}
    missing=[name for name in EXPECTED if name not in runs]
    lines=['# P2 数值终检记录','', '模型 v0.4；发布 v0.6-submit。161 节点，表面加密指数 2，事件区间不宽于 0.02 s。',
        '问题 3 全程逐秒落点；问题 4 每分钟落点、前 3 h 上限 1 s、后期上限 15 s。',
        '诊断只减少状态存储，不改变积分落点。容差与独立对照的终点差验收线为 1 s；独立状态每 6 h 对照，温差线 1e-4 °C、含水率差线 1e-7 kg/kg。','',
        '| 问题 | 基准/h | 收紧 η/h | 终点差/s | 独立 BDF2/h | 终点差/s |','| --- | --- | --- | --- | --- | --- |']
    checks=[]; metrics={}
    # Evidence is accepted only while its producing code and input bytes match.
    # 核对产生数值证据的源码和题给输入指纹，注释变更也会使字节指纹改变
    for label,r in runs.items():
        if label not in EXPECTED: continue
        for name,digest in r['signature']['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,(label,name)
        for name,digest in r['signature']['input_sha256'].items():
            assert hashlib.sha256((ROOT.parent/'附件'/name).read_bytes()).hexdigest()==digest,(label,name)
    # 比较收紧容差与独立参考的终点时间及共同采样时刻的温湿状态
    for q in (3,4):
        tight=runs.get(f'q{q}_tight'); ref=runs.get(f'q{q}_reference')
        b=runs.get(f'q{q}_h1_hm1')
        fmt=lambda r: f'{r["final"]["t"]/3600:.9f}' if r else '待完成'
        diff=lambda r: f'{r["final"]["t"]-BASE[q]:+.6f}' if r else '待完成'
        lines.append(f'| {q} | {BASE[q]/3600:.9f} | {fmt(tight)} | {diff(tight)} | {fmt(ref)} | {diff(ref)} |')
        if tight: checks.append(abs(tight['final']['t']-BASE[q])<=1.)
        if ref: checks.append(abs(ref['final']['t']-BASE[q])<=1.)
        if b:
            checks.append(abs(b['final']['t']-BASE[q])<1e-7)
        if b and ref:
            times=sorted(set(b['summaries'])&set(ref['summaries']),key=int)
            errs={f:max(float(np.max(abs(np.asarray(b['summaries'][t][f])-ref['summaries'][t][f]))) for t in times) for f in ('T','C')}
            metrics[str(q)]={'time_difference_s':ref['final']['t']-BASE[q],
                             'sample_times_s':list(map(int,times)), 'max_abs_difference':errs}
            checks += [errs['T']<=1e-4,errs['C']<=1e-7]
    lines += ['','## 3×3 边界扫描','', '| h/h0 | hm/hm0 | 问题3/h | 问题4/h |','| --- | --- | --- | --- |']
    # 汇总九种系数组合的干燥时长，不将情景范围解释为置信区间
    for h in (.5,1.,2.):
        for m in (.5,1.,2.):
            vals=[]
            for q in (3,4):
                r=runs.get(f'q{q}_h{h:g}_hm{m:g}')
                vals.append(('>' if r['status']=='right_censored' else '')+f'{r["final"]["t"]/3600:.6f}' if r else '待完成')
            lines.append(f'| {h:g} | {m:g} | {vals[0]} | {vals[1]} |')
    for name,r in runs.items():
        if name not in EXPECTED: continue
        e=r.get('endpoint'); s=r['stats']
        checks += [s.get('min_C',1)>0,s.get('max_mass_balance_residual',1)<1e-10]
        if e: checks += [e['left_max_C']>=.15,e['right_max_C']<.15,e['right_s']-e['left_s']<=.020001]
    # 合并完整性、正性、有效浓度残差与严格阈值事件区间的验收条件
    complete=not missing and all(checks)
    report={'complete':complete,'cases_present':len(set(EXPECTED)&set(runs)),
            'missing':missing,'checks_passed':all(checks),'reference_comparison':metrics}
    lines+=['','## 验收范围','',f'预期 22 个全程算例，已完成 {report["cases_present"]} 个。数值验收：'+('通过。' if complete else '尚未完成或存在未通过项。'),
        '独立参考程序不调用主程序的求解、通量、BDF 系数、误差控制或事件定位代码；共享题给输入插值、配置及 NumPy/SciPy 线性代数库。',
        '残差检查针对有效浓度的离散方程，不证明实际干物质/水质量闭合。参数扫描是情景分析，不是统计置信区间。',
        '本记录只裁定数值 P2；论文、结果表、PDF 全页视觉检查与打包验收另记于发布验收记录。',
        '', '缺少算例：'+(', '.join(missing) if missing else '无'), '',
        '独立状态比较：', '```json',json.dumps(metrics,ensure_ascii=False,indent=2),'```']
    # ==================== 检验记录与情景图表输出 ====================
    (ROOT/'p2/summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT.parent/('P2_final_record.md' if complete else 'P2_progress.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # 仅在所有数值检验通过后生成情景热图和论文对照表
    if complete:
        (ROOT.parent/'完整论文-LaTeX/figs').mkdir(parents=True,exist_ok=True)
        fig,axes=plt.subplots(1,2,figsize=(8.2,3.6),layout='constrained')
        for ax,q in zip(axes,(3,4)):
            a=np.array([[runs[f'q{q}_h{h:g}_hm{m:g}']['final']['t']/3600 for m in (.5,1.,2.)] for h in (.5,1.,2.)])
            im=ax.imshow(a,cmap='Blues',aspect='equal')
            ax.set(xticks=range(3),xticklabels=['0.5','1','2'],yticks=range(3),yticklabels=['0.5','1','2'],xlabel='Mass transfer multiplier',ylabel='Heat transfer multiplier',title=f'Question {q}: drying time (h)')
            for i in range(3):
                for j in range(3): ax.text(j,i,f'{a[i,j]:.3f}',ha='center',va='center',color='white' if a[i,j]>(a.max()+a.min())/2 else 'black',fontsize=11)
        for ext in ('pdf','png'):
            fig.savefig(ROOT.parent/f'完整论文-LaTeX/figs/p2_boundary_sensitivity.{ext}',dpi=180)
        plt.close(fig)
        rows=[]
        for q in (3,4):
            tight=runs[f'q{q}_tight'];ref=runs[f'q{q}_reference']
            rows.append(f"{q}&{BASE[q]/3600:.4f}&{tight['final']['t']/3600:.4f}&{tight['final']['t']-BASE[q]:+.4f}&{ref['final']['t']-BASE[q]:+.4f}\\\\")
        tex='\n'.join([r'\begin{table}[H]\centering\small\caption{P2 容差及独立 BDF2 终点对照}\label{tab:p2}',
          r'\begin{tabular}{rrrrr}\toprule 问题&基准/h&收紧容差/h&收紧差/s&独立差/s\\\midrule']+rows+[r'\bottomrule\end{tabular}\end{table}'])
        (ROOT.parent/'完整论文-LaTeX/p2_table.tex').write_text(tex,encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
