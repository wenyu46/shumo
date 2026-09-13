"""P2数值终检的独立BDF2参考实现，不导入主求解器或其数值内核。

共享模型配置、题给数据插值以及NumPy/SciPy线性代数库。
独立实现材料物性、物理圆环矩阵、界面积分、非线性迭代、
步长倍增、BDF历史管理、事件二分及诊断状态构造。
对照只检查实现的一致性，不验证共同物理假设和输入数据的真实性。"""
from dataclasses import asdict
import time
import numpy as np
from scipy.linalg import solve_banded


class ReferenceFailure(RuntimeError):
    pass


# ==================== 独立圆柱有限体积参考模型 ====================
class IndependentBDF2:
    # 建立表面加密节点、圆环控制体权重与界面积分节点
    def __init__(self, cfg, inputs):
        self.cfg, self.inputs = cfg, inputs
        u = np.arange(cfg.n + 1) / cfg.n
        self.x = 1 - (1-u)**cfg.grid_power
        self.mid = .5*(self.x[1:] + self.x[:-1])
        self.edge = np.concatenate(([0.], self.mid, [1.]))
        self.dv = .5*(self.edge[1:]**2 - self.edge[:-1]**2)
        nodes, weights = np.polynomial.legendre.leggauss(cfg.quadrature)
        self.q, self.w = (nodes+1)/2, weights/2
        self.stats = {'steps': 0, 'rejected_steps': 0, 'bdf2_steps': 0,
                      'euler_steps': 0, 'iterations': 0,
                      'max_mass_balance_residual': 0.,
                      'max_moisture_equation_residual': 0.,
                      'max_heat_equation_residual': 0.}

    # 独立计算材料3/4的体积热容、导热系数与扩散模型参数
    def material(self, concentration):
        c = concentration
        if self.cfg.material == 3:
            return ((650+128*c)*(1450+2736*c/(1+c)),
                    .21+.38*c/(1+c), 2.4e-3, .45)
        if self.cfg.material == 4:
            return ((760+90*c)*(1850+2150*c/(1+c)),
                    .12+.20*c/(1+c), 4.2e-4, .30)
        raise ValueError('P2 reference covers questions 3 and 4')

    # 沿相邻节点含水率区间积分扩散系数，计算界面等效传递能力
    def face_diffusion(self, c, temp):
        _, _, prefactor, activation = self.material(c)
        cq = (1-self.q)*c[:-1, None] + self.q*c[1:, None]
        tf = .5*(temp[1:]+temp[:-1]) + 273.15
        return (prefactor*np.exp(-activation/cq - 3850/tf[:, None])) @ self.w

    # 根据相邻时间步比确定BDF2系数；历史不足或步长比越界时用隐式Euler
    @staticmethod
    def bdf(dt, last_dt, history, force_euler=False):
        if force_euler or not history or last_dt is None:
            return (1/dt, -1/dt, 0.), 'euler'
        r = dt/last_dt
        if not .5-1e-12 <= r <= 2+1e-12:
            return (1/dt, -1/dt, 0.), 'euler'
        return ((1+2*r)/(1+r)/dt, -(1+r)/dt, r*r/(1+r)/dt), 'bdf2'

    # 将无量纲圆环换算为当前半径下的容量、界面导通量与表面交换系数
    def geometry(self, coefficient, capacity, radius, boundary):
        # Physical annuli, with the common 2*pi*length factor cancelled.
        mass = capacity*self.dv*radius**2
        conductance = self.mid*coefficient/np.diff(self.x)
        return mass, conductance, radius*boundary

    # 组装三对角离散方程，耦合历史状态与环境Robin边界
    def solve(self, old, older, face, capacity, radius, boundary, ambient, bdf):
        mass, conduct, outer = self.geometry(face, capacity, radius, boundary)
        a0, a1, a2 = bdf
        diagonal = a0*mass
        diagonal[:-1] += conduct
        diagonal[1:] += conduct
        diagonal[-1] += outer
        band = np.zeros((3, len(old)))
        band[0, 1:] = -conduct
        band[1] = diagonal
        band[2, :-1] = -conduct
        rhs = -mass*(a1*old+a2*older)
        rhs[-1] += outer*ambient
        return solve_banded((1, 1), band, rhs, check_finite=False)

    # 由圆环蓄积项及界面通量差检查离散方程残差
    def residual(self, new, old, older, face, capacity, radius, boundary, ambient, bdf):
        mass, conduct, outer = self.geometry(face, capacity, radius, boundary)
        flows = np.zeros(len(new)+1)
        flows[1:-1] = conduct*np.diff(new)
        flows[-1] = outer*(ambient-new[-1])
        a0, a1, a2 = bdf
        return (mass*(a0*new+a1*old+a2*older)-np.diff(flows))/radius**2

    # ==================== 温湿耦合非线性迭代 ====================
    def attempt(self, t, temp, c, older, last_dt, dt, euler=False):
        cfg = self.cfg
        bdf, method = self.bdf(dt, last_dt, older is not None, euler)
        ot, oc = (temp, c) if older is None else older
        ta, ca = self.inputs.boundary(t+dt, cfg.environment)
        radius = self.inputs.radius(t+dt, cfg.shrink)
        guess_t, guess_c = temp.copy(), c.copy()
        before, damping = np.inf, 1.
        # 更新含水率相关物性，交替求解温度与含水率直至状态和残差同时收敛
        for iteration in range(1, cfg.max_iterations+1):
            cap, k, _, _ = self.material(guess_c)
            nt = self.solve(temp, ot, .5*(k[1:]+k[:-1]), cap,
                            radius, 25., ta, bdf)
            nc = self.solve(c, oc, self.face_diffusion(guess_c, nt),
                            np.ones(len(c)), radius, 8e-7, ca, bdf)
            if not np.isfinite(nt).all() or not np.isfinite(nc).all() or min(nc) <= 0:
                raise ReferenceFailure('invalid state')
            error = max(np.max(abs(nt-guess_t))/50, np.max(abs(nc-guess_c)))
            if error <= cfg.tolerance:
                cap, k, _, _ = self.material(nc)
                rt = self.residual(nt, temp, ot, .5*(k[1:]+k[:-1]), cap,
                                   radius, 25., ta, bdf)
                rc = self.residual(nc, c, oc, self.face_diffusion(nc, nt),
                                   np.ones(len(c)), radius, 8e-7, ca, bdf)
                if max(np.max(abs(rt))/50, np.max(abs(rc))) <= cfg.residual_tolerance:
                    if np.max(np.diff(nc)) > 1e-10*max(1., np.max(nc)):
                        raise ReferenceFailure('radial order')
                    self.stats['iterations'] += iteration
                    for name, value in [('heat', np.max(abs(rt))), ('moisture', np.max(abs(rc)))]:
                        key = f'max_{name}_equation_residual'
                        self.stats[key] = max(self.stats[key], float(value))
                    self.stats['max_mass_balance_residual'] = max(
                        self.stats['max_mass_balance_residual'], abs(float(sum(rc))))
                    return nt, nc, method
            # 迭代误差增大时降低更新幅度，抑制非线性迭代振荡
            if error > 1.05*before:
                damping = .5
            guess_t += damping*(nt-guess_t)
            guess_c += damping*(nc-guess_c)
            before = error
        raise ReferenceFailure('nonlinear iteration')

    # 非线性候选步失败时以隐式Euler回退，保留原状态重新计算
    def advance(self, *args):
        try:
            return self.attempt(*args)
        except ReferenceFailure:
            return self.attempt(*args, euler=True)

    # 比较整步与两次半步状态，超出误差容限或可行性约束时缩短步长
    def controlled_step(self, t, temp, c, older, last_dt, dt):
        cfg = self.cfg
        for retry in range(cfg.max_retries+1):
            try:
                nt, nc, method = self.advance(t, temp, c, older, last_dt, dt)
                if older is None:
                    return nt, nc, method, dt
                ht, hc, _ = self.advance(t, temp, c, older, last_dt, dt/2)
                et, ec, _ = self.advance(t+dt/2, ht, hc, (temp, c), dt/2, dt/2)
                limit = 1e-10*max(1., float(max(c)))
                if max(hc)>max(c)+limit or max(ec)>max(hc)+limit:
                    raise ReferenceFailure('temporal order')
                error = max(np.max(abs(nt-et))/50, np.max(abs(nc-ec)))
                if error <= cfg.step_tolerance:
                    return nt, nc, method, dt
                if dt <= cfg.min_dt*1.0000001:
                    raise ReferenceFailure('error at minimum step')
                dt = max(cfg.min_dt, min(dt/2, .9*dt*np.sqrt(cfg.step_tolerance/error)))
                self.stats['rejected_steps'] += 1
            except ReferenceFailure:
                if retry == cfg.max_retries or dt <= cfg.min_dt*1.0000001:
                    raise
                dt = max(cfg.min_dt, dt/2)
                self.stats['rejected_steps'] += 1
        raise ReferenceFailure('retry budget')

    # 首次跨越严格干燥阈值后用历史状态重算并二分定位事件区间
    def endpoint(self, t, temp, c, older, last_dt, dt, nt, nc):
        lo, hi, clo = 0., dt, float(max(c))
        while hi-lo > self.cfg.endpoint_tolerance:
            middle = (lo+hi)/2
            mt, mc, _ = self.advance(t, temp, c, older, last_dt, middle)
            if max(mc)>max(c)+1e-10*max(1., max(c)):
                raise ReferenceFailure('event order')
            if max(mc)<.15:
                hi, nt, nc = middle, mt, mc
            else:
                lo, clo = middle, float(max(mc))
        return nt, nc, {'left_s': t+lo, 'right_s': t+hi,
                         'left_max_C': clo, 'right_max_C': float(max(nc))}

    # ==================== 状态采样与过程指标 ====================
    def snapshot(self, t, temp, c):
        # 按分段线性径向含水率计算未达标截面积比例，保留阈值相等区域
        wet = 0.
        for j in range(len(c)-1):
            left, right, a, b = self.x[j], self.x[j+1], c[j], c[j+1]
            if a >= .15 and b >= .15:
                wet += right**2-left**2
            elif (a >= .15) != (b >= .15):
                crossing = left+(.15-a)*(right-left)/(b-a)
                wet += crossing**2-left**2 if a >= .15 else right**2-crossing**2
        # 返回终点、温湿场、过程指标与数值诊断，供实现一致性检验
        return dict(t=float(t), radius_cm=100*self.inputs.radius(t, self.cfg.shrink),
                    x=self.x.tolist(), T=temp.tolist(), C=c.tolist(),
                    max_C=float(max(c)), mean_C=float(2*self.dv@c), wet_fraction=float(wet))

    # ==================== 主时间推进过程 ====================
    def run(self):
        cfg = self.cfg
        started = time.perf_counter()
        t, last_dt, older, endpoint = 0., None, None, None
        temp, c = np.full(cfg.n+1, 28.), np.full(cfg.n+1, 2.55)
        summaries = {'0': self.snapshot(0, temp, c)}
        desired = (100, 300, 600, 900, 1200, 1500, 1800, 3600, 5400, 7200, 9000, 10800)
        min_c, max_c, min_t, max_t, radial = 2.55, 2.55, 28., 28., 0.
        # 逐步逼近强制采样时刻并监测全域最大含水率的首次严格达标
        while t < cfg.max_time-1e-8:
            stride = 1. if cfg.output_stride == 1 else 60.
            landing = (np.floor((t+1e-8)/stride)+1)*stride
            dt = min(cfg.early_dt if t<10800-1e-8 else cfg.late_dt,
                     landing-t, cfg.max_time-t,
                     min([s-t for s in desired if s>t+1e-8] or [cfg.max_time]))
            nt, nc, method, dt = self.controlled_step(t, temp, c, older, last_dt, dt)
            if cfg.stop_dry and max(nc)<.15:
                nt, nc, endpoint = self.endpoint(t, temp, c, older, last_dt, dt, nt, nc)
                dt = endpoint['right_s']-t
            # 接受新时间层状态，更新BDF历史及实际步长
            older, temp, c, last_dt = (temp, c), nt, nc, dt
            t += dt
            if abs(t-landing)<=1e-6:
                t = float(landing)
            self.stats['steps'] += 1
            self.stats[method+'_steps'] += 1
            min_c, max_c = min(min_c, float(min(c))), max(max_c, float(max(c)))
            min_t, max_t = min(min_t, float(min(temp))), max(max_t, float(max(temp)))
            radial = max(radial, float(max(np.diff(c))))
            if abs(t/21600-round(t/21600))<1e-9:
                summaries[str(round(t))] = self.snapshot(t, temp, c)
            if endpoint is not None:
                break
        if cfg.stop_dry and endpoint is None:
            raise RuntimeError('NO_ENDPOINT')
        self.stats.update(runtime_s=time.perf_counter()-started, min_C=min_c,
                          max_C=max_c, min_T=min_t, max_T=max_t,
                          max_radial_monotonicity_violation=radial)
        return dict(config=asdict(cfg), final=self.snapshot(t, temp, c),
                    endpoint=endpoint, stats=self.stats, summaries=summaries)
