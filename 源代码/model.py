"""2026年全国大学生数学建模竞赛A题的非线性径向温湿传递求解器。

v0.4采用表面加密的非均匀有限体积网格、变步长BDF2时间格式、
隐式Euler回退、精确输出时刻落点及干燥终点保护。
物理解释沿用题设的有效温湿传递模型，
未加入汽化潜热或真实干物质质量闭合项。"""
# ==================== 依赖与路径配置 ====================
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
import time
import numpy as np
import openpyxl
from scipy.linalg import solve_banded

# 当前脚本所在目录：用于定位题目附件数据
ROOT = Path(__file__).resolve().parent


# ==================== 数值积分与材料物性模型 ====================
# 缓存高斯-勒让德积分节点与权重，供后续界面扩散系数积分使用
@lru_cache(maxsize=None)
def gauss(order):
    z, w = np.polynomial.legendre.leggauss(order)
    return (z + 1.0) / 2.0, w / 2.0


# 根据材料类型、含水率与温度计算体积热容、导热系数和水分扩散系数
def properties(material, c, t):
    c = np.asarray(c, dtype=float)
    t = np.asarray(t, dtype=float)
    if material == 1:
        return (np.full_like(c, 820.0 * 2600.0), np.full_like(c, 0.36),
                7e-9 * np.exp(-0.89 / c))
    if material == 3:
        return ((650.0 + 128.0 * c) * (1450.0 + 2736.0 * c / (c + 1.0)),
                0.21 + 0.38 * c / (c + 1.0),
                2.4e-3 * np.exp(-0.45 / c - 3850.0 / (t + 273.15)))
    if material == 4:
        return ((760.0 + 90.0 * c) * (1850.0 + 2150.0 * c / (c + 1.0)),
                0.12 + 0.20 * c / (c + 1.0),
                4.2e-4 * np.exp(-0.30 / c - 3850.0 / (t + 273.15)))
    raise ValueError(material)


# 对相邻控制体含水率区间上的扩散系数进行高斯积分，得到界面有效扩散系数
def integral_diffusivity(material, left, right, tface, order=8):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    tface = np.asarray(tface, dtype=float)
    z, w = gauss(order)
    c = left[:, None] + (right - left)[:, None] * z
    if material == 1:
        vals = 7e-9 * np.exp(-0.89 / c)
    elif material == 3:
        vals = 2.4e-3 * np.exp(-0.45 / c - 3850.0 / (tface[:, None] + 273.15))
    elif material == 4:
        vals = 4.2e-4 * np.exp(-0.30 / c - 3850.0 / (tface[:, None] + 273.15))
    else:
        raise ValueError(material)
    return vals @ w



# ==================== 模型与数值求解参数 ====================
@dataclass
class Config:
    # 物理模型与空间网格参数
    material: int = 3
    shrink: bool = False
    n: int = 80
    grid_power: float = 2.0
    # 时间离散与数值积分参数
    early_dt: float = 1.0
    late_dt: float = 15.0
    quadrature: int = 8
    method: str = 'bdf2'
    # 非线性迭代、方程残差与自适应步长控制参数
    tolerance: float = 1e-9
    residual_tolerance: float = 2e-8
    step_tolerance: float = 1e-4
    max_iterations: int = 40
    endpoint_tolerance: float = 0.02
    min_dt: float = 0.01
    max_retries: int = 8
    # 仿真终止条件与结果输出参数
    max_time: float = 7 * 86400.0
    stop_dry: bool = True
    keep_short: bool = False
    compact_output: bool = False
    output_stride: int = 60
    environment: str = 'last'



# ==================== 输入数据与边界条件 ====================
class Inputs:
    def __init__(self):
        # 读取空气温湿度与半径变化附件，并检查时间序列有效性
        data = []
        for name in ('附件1.xlsx', '附件2.xlsx'):
            path = ROOT.parent / '附件' / name
            wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
            rows = list(wb.active.values)
            wb.close()
            a = np.asarray(rows[1:], dtype=float)
            if not np.isfinite(a).all() or not np.all(np.diff(a[:, 0]) > 0):
                raise ValueError(f'invalid input: {name}')
            data.append(a)
        self.air, self.radii = data

    def boundary(self, t, environment='last'):
        # 根据当前时刻插值得到环境温度与环境含水边界值
        a = self.air
        # 超出附件时域时，可采用最后一小时均值作为延拓边界
        if t > a[-1, 0] and environment == 'last_hour_mean':
            block = a[a[:, 0] >= a[-1, 0] - 3600.0, 1:]
            target = block.mean(axis=0)
            if t <= a[-1, 0] + 60.0:
                w = (t - a[-1, 0]) / 60.0
                return tuple((1.0 - w) * a[-1, 1:] + w * target)
            return tuple(target)
        return (float(np.interp(t, a[:, 0], a[:, 1])),
                float(np.interp(t, a[:, 0], a[:, 2])))

    def radius(self, t, shrink):
        # 未考虑收缩时保持初始半径不变
        if not shrink:
            return 0.02
        # 考虑收缩时仅在附件给定时域内插值，禁止外推
        if t > self.radii[-1, 0] + 1e-8:
            raise ValueError('Shrinkage data extrapolation is not allowed')
        return 0.01 * float(np.interp(t, self.radii[:, 0], self.radii[:, 1]))



# 数值推进失败的统一异常类型，用于触发方法回退与步长缩减
class StepFailure(RuntimeError):
    pass



# ==================== 热湿耦合有限体积求解器 ====================
class Solver:
    def __init__(self, config, inputs=None):
        # 检查网格与输出参数是否合法
        cfg = config
        if cfg.n < 2 or cfg.grid_power < 1:
            raise ValueError('invalid grid')
        if cfg.keep_short and cfg.output_stride < 1:
            raise ValueError('output_stride must be positive')
        # 保存求解配置并加载边界输入
        self.cfg = cfg
        self.inputs = inputs or Inputs()
        # 构造径向非均匀网格、控制体边界及归一化控制体体积
        self.x = 1.0 - (1.0 - np.arange(cfg.n + 1, dtype=float) / cfg.n) ** cfg.grid_power
        self.face = (self.x[:-1] + self.x[1:]) / 2.0
        self.edges = np.r_[0.0, self.face, 1.0]
        self.volume = np.diff(self.edges ** 2) / 2.0
        self.dx = np.diff(self.x)
        # 预设结果采样距离，并初始化数值稳定性与收敛统计量
        self.output_distances_cm = np.arange(0.0, 2.001, 0.1)
        self.stats = {
            'steps': 0, 'iterations': 0, 'rejected_steps': 0,
            'max_iterations': 0, 'max_mass_balance_residual': 0.0,
            'max_moisture_equation_residual': 0.0,
            'max_heat_equation_residual': 0.0,
            'bdf2_steps': 0, 'euler_steps': 0, 'event_fallbacks': 0,
            'step_failures': 0, 'max_step_error': 0.0,
        }

    # 组装有限体积隐式离散的三对角带状矩阵及表面对流边界项
    def coefficients(self, face_coefficient, capacity, boundary, radius, dt):
        face_coefficient = np.asarray(face_coefficient, dtype=float)
        capacity = np.asarray(capacity, dtype=float)
        g = self.face * face_coefficient / (self.dx * radius ** 2)
        left = dt * np.r_[0.0, g] / (self.volume * capacity)
        right = dt * np.r_[g, 0.0] / (self.volume * capacity)
        conv = dt * boundary / (radius * self.volume[-1] * capacity[-1])
        ab = np.zeros((3, len(capacity)), dtype=float)
        ab[0, 1:] = -right[:-1]
        ab[1, :] = 1.0 + left + right
        ab[1, -1] += conv
        ab[2, :-1] = -left[1:]
        return ab, conv

    # 求解一次固定系数的隐式线性扩散/传热方程
    def linear_step(self, u, face_coefficient, capacity, boundary, ambient, radius, dt):
        ab, conv = self.coefficients(face_coefficient, capacity, boundary, radius, dt)
        rhs = np.asarray(u, dtype=float).copy()
        rhs[-1] += conv * ambient
        return solve_banded((1, 1), ab, rhs, check_finite=False)

    # 根据相邻时间步长比构造变步长 BDF2 系数；条件不满足时退化为隐式 Euler
    def _bdf_coefficients(self, dt, previous_dt, has_history, method=None):
        method = method or self.cfg.method
        if method != 'bdf2' or not has_history or previous_dt is None:
            return 1.0 / dt, -1.0 / dt, 0.0, 'euler'
        ratio = dt / previous_dt
        if ratio < 0.5 - 1e-12 or ratio > 2.0 + 1e-12:
            return 1.0 / dt, -1.0 / dt, 0.0, 'euler'
        return ((1.0 + 2.0 * ratio) / (dt * (1.0 + ratio)),
                -(1.0 + ratio) ** 2 / (dt * (1.0 + ratio)),
                ratio ** 2 / (dt * (1.0 + ratio)), 'bdf2')

    # 按一般 BDF 时间离散形式组装并求解三对角线性系统
    def _linear_general(self, old, previous, face_coefficient, capacity,
                        boundary, ambient, radius, dt, a0, a1, a2):
        capacity = np.asarray(capacity, dtype=float)
        g = self.face * np.asarray(face_coefficient, dtype=float) / (self.dx * radius ** 2)
        scale = 1.0 / a0
        left = scale * np.r_[0.0, g] / (self.volume * capacity)
        right = scale * np.r_[g, 0.0] / (self.volume * capacity)
        conv = scale * boundary / (radius * self.volume[-1] * capacity[-1])
        ab = np.zeros((3, len(capacity)), dtype=float)
        ab[0, 1:] = -right[:-1]
        ab[1, :] = 1.0 + left + right
        ab[1, -1] += conv
        ab[2, :-1] = -left[1:]
        history_rhs = -(a1 * old + a2 * previous) / a0
        history_rhs = np.asarray(history_rhs, dtype=float)
        history_rhs[-1] += conv * ambient
        return solve_banded((1, 1), ab, history_rhs, check_finite=False)

    # 计算离散守恒方程残差，用于判定非线性迭代是否真正收敛
    def _residual(self, old, previous, new, face_coefficient, capacity,
                  boundary, ambient, radius, dt, a0, a1, a2):
        capacity = np.asarray(capacity, dtype=float)
        g = self.face * np.asarray(face_coefficient, dtype=float) / (self.dx * radius ** 2)
        flux_left = np.r_[0.0, g] * (new - np.r_[new[0], new[:-1]])
        flux_right = np.r_[g, 0.0] * (np.r_[new[1:], new[-1]] - new)
        flux_right[-1] = -boundary * (new[-1] - ambient) / radius
        # 离散守恒方程：容量项 -（右侧通量 - 左侧通量）= 0
        lhs = capacity * self.volume * (a0 * new + a1 * old + a2 * previous)
        return lhs - (flux_right - flux_left)

    # 计算控制体界面水分扩散系数：可选算术平均或积分平均
    def diffusion_faces(self, c, temp):
        if self.cfg.method == 'arithmetic':
            d = properties(self.cfg.material, c, temp)[2]
            return (d[:-1] + d[1:]) / 2.0
        return integral_diffusivity(
            self.cfg.material, c[:-1], c[1:],
            (temp[:-1] + temp[1:]) / 2.0, self.cfg.quadrature)


    # 执行一个候选时间步：耦合求解温度场与含水率场，并检查收敛与物理约束
    def _attempt(self, t, old_t, old_c, prev_t, prev_c, previous_dt, dt, force_method=None):
        cfg = self.cfg
        # 获取新时刻的环境边界条件与当前半径
        ta, ca = self.inputs.boundary(t + dt, cfg.environment)
        radius = self.inputs.radius(t + dt, cfg.shrink)
        # 根据历史状态决定采用 BDF2 还是隐式 Euler
        has_history = prev_t is not None and prev_c is not None and previous_dt is not None
        a0, a1, a2, actual_method = self._bdf_coefficients(
            dt, previous_dt, has_history, force_method)
        # 历史状态不足时使用当前旧状态补齐，保证统一离散格式
        previous_t = old_t if prev_t is None else prev_t
        previous_c = old_c if prev_c is None else prev_c
        # 以旧状态作为 Picard 非线性迭代初值
        new_t = old_t.copy()
        new_c = old_c.copy()
        previous_error = np.inf
        damping = 1.0
        last = None
        # Picard 迭代：交替更新热传导方程与水分扩散方程
        for it in range(1, cfg.max_iterations + 1):
            # 根据当前温湿状态更新材料物性，并先求解温度场
            cap, k, _ = properties(cfg.material, new_c, new_t)
            ts = self._linear_general(
                old_t, previous_t, (k[:-1] + k[1:]) / 2.0, cap,
                25.0, ta, radius, dt, a0, a1, a2)
            # 基于更新后的温度场计算扩散系数，再求解含水率场
            dface = self.diffusion_faces(new_c, ts)
            cs = self._linear_general(
                old_c, previous_c, dface, np.ones_like(new_c),
                8e-7, ca, radius, dt, a0, a1, a2)
            # 非有限数值表明当前时间步失效
            if not np.isfinite(ts).all() or not np.isfinite(cs).all():
                raise StepFailure('nonfinite state')
            # 以温度尺度归一化后的最大改变量作为 Picard 收敛指标
            error = max(float(np.max(np.abs(ts - new_t))) / 50.0,
                        float(np.max(np.abs(cs - new_c))))
            last = (ts, cs)
            # 达到迭代容差后进一步检查离散方程残差与物理可行性
            if error <= cfg.tolerance:
                cap2, k2, _ = properties(cfg.material, cs, ts)
                dface2 = self.diffusion_faces(cs, ts)
                rc = self._residual(
                    old_c, previous_c, cs, dface2, np.ones_like(cs),
                    8e-7, ca, radius, dt, a0, a1, a2)
                rt = self._residual(
                    old_t, previous_t, ts, (k2[:-1] + k2[1:]) / 2.0,
                    cap2, 25.0, ta, radius, dt, a0, a1, a2)
                rn_c = float(np.max(np.abs(rc)))
                rn_t = float(np.max(np.abs(rt))) / 50.0
                if max(rn_c, rn_t) <= cfg.residual_tolerance:
                    # 含水率必须保持为正，并满足径向单调性约束
                    if cs.min() <= 0.0:
                        raise StepFailure('nonpositive state')
                    monotonic_limit = 1e-10 * max(1.0, float(cs.max()))
                    if float(np.diff(cs).max()) > monotonic_limit:
                        raise StepFailure('radial monotonicity violation')
                    # 记录收敛迭代次数、方程残差与质量守恒残差
                    self.stats['iterations'] += it
                    self.stats['max_iterations'] = max(self.stats['max_iterations'], it)
                    self.stats['max_moisture_equation_residual'] = max(
                        self.stats['max_moisture_equation_residual'], rn_c)
                    self.stats['max_heat_equation_residual'] = max(
                        self.stats['max_heat_equation_residual'], rn_t * 50.0)
                    balance = abs(float(self.volume @ (
                        a0 * cs + a1 * old_c + a2 * previous_c)
                        + 8e-7 / radius * (cs[-1] - ca)))
                    self.stats['max_mass_balance_residual'] = max(
                        self.stats['max_mass_balance_residual'], balance)
                    return ts, cs, actual_method, error
            # 若误差出现增长，则减小松弛因子以增强非线性迭代稳定性
            if error > previous_error * 1.05:
                damping = 0.5
            new_t += damping * (ts - new_t)
            new_c += damping * (cs - new_c)
            previous_error = error
        raise StepFailure(f'{actual_method} iteration did not converge')

    # 单步推进失败时优先回退到隐式 Euler，再将失败交给外层自适应步长处理
    def _advance_once(self, *args, **kwargs):
        try:
            return self._attempt(*args, **kwargs)
        except StepFailure:
            force = kwargs.get('force_method')
            if force != 'euler':
                try:
                    self.stats['event_fallbacks'] += 1 if force == 'bdf2' else 0
                    return self._attempt(*args, **{**kwargs, 'force_method': 'euler'})
                except StepFailure:
                    pass
            raise


    # 采用“整步—两个半步”比较估计时间离散误差，并据此自适应调整步长
    def _step_with_error(self, t, old_t, old_c, prev_t, prev_c, previous_dt, dt):
        cfg = self.cfg
        # 对当前时间步进行有限次数重试
        for retry in range(cfg.max_retries + 1):
            try:
                # 先完成一个整步推进
                full_t, full_c, method, picard_error = self._advance_once(
                    t, old_t, old_c, prev_t, prev_c, previous_dt, dt)
                # 无 BDF2 历史时不进行二阶步长误差估计
                if cfg.method != 'bdf2' or prev_t is None or previous_dt is None:
                    return full_t, full_c, dt, method, 0.0, retry
                # 再执行两个半步，用于与整步结果比较
                half = dt / 2.0
                mid_t, mid_c, _, _ = self._advance_once(
                    t, old_t, old_c, prev_t, prev_c, previous_dt, half)
                end_t, end_c, _, _ = self._advance_once(
                    t + half, mid_t, mid_c, old_t, old_c, half, half)
                # 检查端点判据的单调性，避免错误跨越干燥终点
                limit = 1e-10 * max(1.0, float(old_c.max()))
                if (float(mid_c.max()) > float(old_c.max()) + limit
                        or float(end_c.max()) > float(mid_c.max()) + limit):
                    raise StepFailure('endpoint indicator is not monotone')
                # 以整步与双半步之差估计当前时间步误差
                err = max(float(np.max(np.abs(full_t - end_t))) / 50.0,
                          float(np.max(np.abs(full_c - end_c))))
                self.stats['max_step_error'] = max(self.stats['max_step_error'], err)
                # 误差满足要求则接受整步结果
                if err <= cfg.step_tolerance:
                    return full_t, full_c, dt, method, err, retry
                # 已达到最小步长仍不满足误差要求时判定步进失败
                if dt <= cfg.min_dt * 1.0000001:
                    self.stats['step_failures'] += 1
                    raise StepFailure('step error remains above tolerance at min_dt')
                # 根据误差大小缩短时间步并重新计算
                dt = max(cfg.min_dt, min(dt / 2.0, dt * 0.9 *
                         np.sqrt(cfg.step_tolerance / max(err, 1e-300))))
                self.stats['rejected_steps'] += 1
            # 数值求解失败时直接折半时间步后重试
            except StepFailure:
                if retry >= cfg.max_retries or dt <= cfg.min_dt * 1.0000001:
                    self.stats['step_failures'] += 1
                    raise
                dt = max(cfg.min_dt, dt / 2.0)
                self.stats['rejected_steps'] += 1
        raise StepFailure('unreachable retry state')


    # 构造紧凑输出快照：仅保存指定径向位置的温度和含水率采样值
    def _compact_snapshot(self, t, temp, c):
        radius_cm = self.inputs.radius(t, self.cfg.shrink) * 100.0
        grid_cm = self.x * radius_cm
        out = {'t': float(t), 'radius_cm': float(radius_cm),
               'max_C': float(c.max()), 'wet_fraction': float(self._wet(c)),
               'mean_C': float(2.0 * self.volume @ c),
               'sample_distances_cm': self.output_distances_cm.tolist()}
        for field, values in (('T', temp), ('C', c)):
            out[field] = [float(np.interp(d, grid_cm, values))
                          if d <= radius_cm + 1e-10 else None
                          for d in self.output_distances_cm]
        return out

    # 根据 C >= 0.15 的径向区间计算仍处于湿润状态的面积比例
    def _wet(self, c):
        wet = 0.0
        for i in range(len(c) - 1):
            lo, hi = self.x[i], self.x[i + 1]
            a, b = c[i], c[i + 1]
            if a >= 0.15 and b >= 0.15:
                wet += hi * hi - lo * lo
            elif (a >= 0.15) != (b >= 0.15):
                crossing = lo + (0.15 - a) / (b - a) * (hi - lo)
                wet += crossing ** 2 - lo ** 2 if a >= 0.15 else hi ** 2 - crossing ** 2
        return wet

    # 生成当前时刻完整或紧凑的温湿场结果快照
    def snapshot(self, t, temp, c, compact=False):
        if compact:
            return self._compact_snapshot(t, temp, c)
        radius = self.inputs.radius(t, self.cfg.shrink)
        return {'t': float(t), 'radius_cm': float(radius * 100.0),
                'T': temp.tolist(), 'C': c.tolist(),
                'max_C': float(c.max()), 'wet_fraction': float(self._wet(c)),
                'mean_C': float(2.0 * self.volume @ c),
                'x': self.x.tolist()}

    # 按短周期、分钟级和关键时刻三种频率记录仿真结果
    def _record(self, t, temp, c, short, long, summary, initial=False):
        cfg = self.cfg
        compact = cfg.compact_output
        if cfg.keep_short and (initial or abs(t / cfg.output_stride -
                                             round(t / cfg.output_stride)) < 1e-8):
            short.append(self.snapshot(t, temp, c, compact=compact))
        if initial or abs(t / 60.0 - round(t / 60.0)) < 1e-8:
            long.append(self.snapshot(t, temp, c, compact=False))
        desired = {100, 300, 600, 900, 1200, 1500, 1800, 3600, 5400,
                   7200, 9000, 10800}
        if any(abs(t - s) < 1e-8 for s in desired) or abs(t / 21600.0 -
                                                        round(t / 21600.0)) < 1e-8:
            summary[str(round(t))] = self.snapshot(t, temp, c, compact=False)


    # 在首次跨越干燥阈值的时间步内用二分法定位终止时刻
    def _locate_endpoint(self, t, old_t, old_c, prev_t, prev_c, previous_dt,
                         right_dt, right_t, right_c):
        left = 0.0
        right = right_dt
        left_max = float(old_c.max())
        rt, rc = right_t, right_c
        limit = 1e-10 * max(1.0, float(old_c.max()))
        # 逐步缩小“未干燥—已干燥”时间区间，直至满足端点时间容差
        while right - left > self.cfg.endpoint_tolerance:
            mid = (left + right) / 2.0
            mt, mc, _, _ = self._advance_once(
                t, old_t, old_c, prev_t, prev_c, previous_dt, mid)
            if float(mc.max()) > float(old_c.max()) + limit:
                raise StepFailure('endpoint indicator is not monotone')
            if mc.max() < 0.15:
                right, rt, rc = mid, mt, mc
            else:
                left, left_max = mid, float(mc.max())
        return rt, rc, {'left_s': float(t + left), 'right_s': float(t + right),
                        'left_max_C': left_max, 'right_max_C': float(rc.max())}


    # ==================== 主时间推进过程 ====================
    # 从初始温湿状态出发推进至干燥终点或最大模拟时间，并汇总全过程结果
    def run(self):
        cfg = self.cfg
        started = time.perf_counter()
        # 初始化时间、温度场、含水率场及 BDF 历史状态
        t = 0.0
        temp = np.full(cfg.n + 1, 28.0)
        c = np.full(cfg.n + 1, 2.55)
        prev_t = prev_c = None
        previous_dt = None
        # 初始化多时间尺度输出容器与首个状态快照
        short, long, summary = [], [], {}
        self._record(t, temp, c, short, long, summary, initial=True)
        # 初始化全过程极值、单调性、终点与时间步统计量
        minc = maxc = 2.55
        mint = maxt = 28.0
        radial_violation = 0.0
        endpoint = None
        accepted = 0
        min_dt_seen = np.inf
        max_dt_seen = 0.0
        # 主时间循环：根据阶段、输出时刻和关键时刻共同确定下一步步长
        while t < cfg.max_time - 1e-8:
            # 早期采用小步长、后期采用大步长，并保证精确落在输出网格上
            base = cfg.early_dt if t < 10800.0 - 1e-8 else cfg.late_dt
            stride = 1.0 if cfg.output_stride == 1 else 60.0
            next_output = (np.floor((t + 1e-8) / stride) + 1.0) * stride
            dt = min(base, cfg.max_time - t, next_output - t)
            # 强制时间步精确命中题目关注的关键时刻
            targets = [s - t for s in (100, 300, 600, 900, 1200, 1500, 1800,
                                       3600, 5400, 7200, 9000, 10800)
                       if s > t + 1e-8]
            if targets:
                dt = min(dt, min(targets))
            # 完成一次带误差控制的自适应时间推进
            try:
                nt, nc, used_dt, method, step_error, _ = self._step_with_error(
                    t, temp, c, prev_t, prev_c, previous_dt, dt)
            except StepFailure:
                raise
            # 首次达到整体含水率阈值时，进一步定位精确干燥终点
            if cfg.stop_dry and nc.max() < 0.15:
                nt, nc, endpoint = self._locate_endpoint(
                    t, temp, c, prev_t, prev_c, previous_dt, used_dt, nt, nc)
                used_dt = endpoint['right_s'] - t
            # 接受新状态并更新 BDF 历史信息
            prev_t, prev_c, previous_dt = temp, c, used_dt
            temp, c = nt, nc
            t += used_dt
            # 消除浮点误差，使输出时刻严格落在预定时间网格
            if abs(t - next_output) <= 1e-6:
                t = float(next_output)
            # 更新步数、时间离散方法及时间步范围统计
            accepted += 1
            self.stats['steps'] += 1
            if method == 'bdf2':
                self.stats['bdf2_steps'] += 1
            else:
                self.stats['euler_steps'] += 1
            min_dt_seen = min(min_dt_seen, used_dt)
            max_dt_seen = max(max_dt_seen, used_dt)
            # 更新温湿场全过程极值与径向单调性指标
            minc = min(minc, float(c.min()))
            maxc = max(maxc, float(c.max()))
            mint = min(mint, float(temp.min()))
            maxt = max(maxt, float(temp.max()))
            radial_violation = max(radial_violation, float(np.diff(c).max()))
            # 按预设频率保存当前状态；达到干燥终点后结束推进
            self._record(t, temp, c, short, long, summary)
            if endpoint:
                break
        # 构造最终状态并检查是否成功找到题设干燥终点
        final = self.snapshot(t, temp, c, compact=False)
        if cfg.stop_dry and endpoint is None:
            raise RuntimeError('NO_ENDPOINT')
        # 确保最终时刻被写入短周期和分钟级结果序列
        if cfg.keep_short and (not short or short[-1]['t'] != t):
            short.append(self.snapshot(t, temp, c, compact=cfg.compact_output))
        if not long or long[-1]['t'] != t:
            long.append(final)
        # 汇总运行时间、状态极值、稳定性指标及实际时间步统计
        self.stats.update({'runtime_s': time.perf_counter() - started,
                           'min_C': minc, 'max_C': maxc, 'min_T': mint,
                           'max_T': maxt,
                           'max_radial_monotonicity_violation': radial_violation,
                           'accepted_macro_steps': accepted,
                           'actual_min_macro_dt': min_dt_seen,
                           'actual_max_macro_dt': max_dt_seen})
        # 返回配置、最终状态、终点区间、统计量及各时间尺度结果
        return {'config': asdict(cfg), 'final': final, 'endpoint': endpoint,
                'stats': self.stats, 'short': short, 'long': long,
                'summaries': summary}



# ==================== 结果后处理 ====================
# 在给定快照中按实际半径插值提取指定径向距离处的温度或含水率
def sample(snapshot, field, distances):
    distances = np.asarray(distances, dtype=float)
    # 紧凑快照直接使用预设采样网格进行二次插值
    if 'sample_distances_cm' in snapshot:
        raw_grid = np.asarray(snapshot['sample_distances_cm'], dtype=float)
        raw_values = np.asarray(snapshot[field], dtype=object)
        valid = np.array([v is not None and np.isfinite(float(v)) for v in raw_values])
        if valid.sum() < 2:
            return [None for _ in distances]
        grid = raw_grid[valid]
        values = np.asarray([float(v) for v in raw_values[valid]], dtype=float)
        return [None if d > snapshot['radius_cm'] + 1e-10 else float(np.interp(d, grid, values))
                for d in distances]
    # 完整快照根据归一化径向坐标恢复实际距离后插值
    radius = snapshot['radius_cm']
    x = np.asarray(snapshot.get('x', np.linspace(0.0, 1.0, len(snapshot[field]))))
    grid = x * radius
    return [float(np.interp(d, grid, snapshot[field]))
            if d <= radius + 1e-10 else None for d in distances]
