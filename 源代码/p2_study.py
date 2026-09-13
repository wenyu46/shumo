"""冻结v0.4模型的P2数值终检程序，运行入口为python p2_study.py。

仅通过边界系数倍数及诊断存储扩展主求解器。
输出时刻、时间积分、网格及终点判定合同保持不变。"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from model import Config, Inputs, Solver

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'p2'


# ==================== 边界系数情景模型 ====================
class BoundaryStudy(Solver):
    def __init__(self, cfg, inputs=None, h_factor=1., hm_factor=1.):
        super().__init__(cfg, inputs)
        self.h_factor, self.hm_factor = h_factor, hm_factor

    # 分别施加换热与传质系数倍数，仅用于给定情景比较
    def scaled_boundary(self, value):
        if value == 25.:
            return value * self.h_factor
        if value == 8e-7:
            return value * self.hm_factor
        raise ValueError('unexpected boundary coefficient')

    def _linear_general(self, old, previous, faces, capacity, boundary,
                        ambient, radius, dt, a0, a1, a2):
        return super()._linear_general(old, previous, faces, capacity,
            self.scaled_boundary(boundary), ambient, radius, dt, a0, a1, a2)

    def _residual(self, old, previous, new, faces, capacity, boundary,
                  ambient, radius, dt, a0, a1, a2):
        return super()._residual(old, previous, new, faces, capacity,
            self.scaled_boundary(boundary), ambient, radius, dt, a0, a1, a2)

    # 执行候选时间步并以情景传质系数核对有效浓度离散平衡
    def _attempt(self, t, old_t, old_c, prev_t, prev_c, previous_dt,
                 dt, force_method=None):
        old_balance = self.stats['max_mass_balance_residual']
        answer = super()._attempt(t, old_t, old_c, prev_t, prev_c,
                                  previous_dt, dt, force_method)
        # The frozen kernel's scalar diagnostic uses the nominal hm. Correct
        # that diagnostic as well as the two assembled boundary conditions.
        a0, a1, a2, _ = self._bdf_coefficients(
            dt, previous_dt, prev_c is not None, answer[2])
        c = answer[1]
        ca = self.inputs.boundary(t + dt, self.cfg.environment)[1]
        radius = self.inputs.radius(t + dt, self.cfg.shrink)
        balance = abs(float(self.volume @ (a0*c + a1*old_c +
            a2*(old_c if prev_c is None else prev_c)) +
            8e-7*self.hm_factor/radius*(c[-1]-ca)))
        self.stats['max_mass_balance_residual'] = max(old_balance, balance)
        return answer

    # 减少诊断状态存储，保留求解器原有强制时间落点规则
    def _record(self, t, temp, c, short, long, summary, initial=False):
        # Storage only: cfg.output_stride still controls the same 1 s/60 s
        # mandatory landing grid inside Solver.run, including Q3 after 3 h.
        self.last_state = (t, temp, c)
        if initial or abs(t/21600-round(t/21600)) < 1e-9:
            summary[str(round(t))] = self.snapshot(t, temp, c)


# 设置问题3/4共同的161节点与终点合同，容差用于时间步收敛对照
def config_for(question, eta=1e-4):
    moving = question == 4
    return Config(material=4 if moving else 3, shrink=moving, n=160,
        grid_power=2., early_dt=1., late_dt=15. if moving else 1.,
        output_stride=60 if moving else 1, step_tolerance=eta,
        keep_short=False, compact_output=not moving,
        max_time=259200. if moving else 604800.)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ==================== 情景算例与独立参考计算 ====================
def run_case(spec):
    label, question, eta, hf, mf, independent = spec
    cfg = config_for(question, eta)
    sources = ['model.py', 'p2_study.py']
    # 选取独立实现或主求解器边界扩展；一致性只验证实现而非物理真实性
    if independent:
        sources.append('p2_reference.py')
    # 绑定系数、配置及源码和附件指纹，避免错误复用旧情景记录
    signature = {'config': asdict(cfg), 'h_factor': hf, 'hm_factor': mf,
        'source_sha256': {p: sha(ROOT/p) for p in sources},
        'input_sha256': {p: sha(ROOT.parent/'附件'/p)
                         for p in ('附件1.xlsx', '附件2.xlsx')}}
    path = OUT / 'runs' / (label + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        cached = json.loads(path.read_text(encoding='utf-8'))
        if cached.get('signature') == signature:
            return label, cached['final']['t']/3600, 'cache'
    started = time.perf_counter()
    if independent:
        from p2_reference import IndependentBDF2
        solver = IndependentBDF2(cfg, Inputs())
    else:
        solver = BoundaryStudy(cfg, h_factor=hf, hm_factor=mf)
    # 计算到首次全域达标；允许将观测期限内未达标的非独立算例记录为右删失
    try:
        result = solver.run()
        result['status'] = 'endpoint_reached'
    except RuntimeError as error:
        if str(error) != 'NO_ENDPOINT' or independent:
            raise
        t, temp, c = solver.last_state
        result = {'status': 'right_censored', 'endpoint': None,
                  'final': solver.snapshot(t, temp, c), 'stats': solver.stats}
    # ==================== 诊断记录输出 ====================
    result.update(signature=signature, label=label,
                  wall_s=time.perf_counter()-started)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)
    return label, result['final']['t']/3600, result['status']


# 组织收紧容差、九种边界情景及独立BDF2参考三组数值检验
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--group', choices=['scan', 'tight', 'reference', 'all'], default='all')
    args = parser.parse_args()
    cases = []
    if args.group in ('tight', 'all'):
        cases += [(f'q{q}_tight', q, 1e-5, 1., 1., False) for q in (3, 4)]
    # 3×3倍数网格是有限情景分析，不提供统计置信区间或全局误差界
    if args.group in ('scan', 'all'):
        cases += [(f'q{q}_h{h:g}_hm{m:g}', q, 1e-4, h, m, False)
                  for q in (3, 4) for h in (.5, 1., 2.) for m in (.5, 1., 2.)]
    if args.group in ('reference', 'all'):
        cases += [(f'q{q}_reference', q, 1e-4, 1., 1., True) for q in (3, 4)]
    print(f'P2: {len(cases)} cases, {args.workers} workers', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, spec): spec[0] for spec in cases}
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == '__main__':
    main()
