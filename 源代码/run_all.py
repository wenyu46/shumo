"""v0.4正式算例的统一配置与运行入口。

问题2全过程轨迹同时作为问题3固定半径轨迹：
一次求解提供问题2逐秒输出及问题3每分钟的过程状态。"""
# ==================== 模型配置与求解入口 ====================
try:
    from .model import Config
    from .run_study import run_case
except ImportError:  # direct execution from 改进模型/ remains supported
    from model import Config
    from run_study import run_case


# 配置四问的材料类型、径向网格、收缩开关及强制输出时刻
def configurations():
    return {
        # 问题1：固定半径预热阶段，计算至1800秒并逐秒保存
        'q1_final': Config(material=1, n=160, grid_power=2.0,
                           early_dt=1.0, late_dt=1.0, max_time=1800.0,
                           stop_dry=False, keep_short=True,
                           compact_output=False, output_stride=1),
        # 问题2与问题3共用固定半径全过程轨迹，终点由全域含水率阈值确定
        'q2_full': Config(material=3, n=160, grid_power=2.0,
                           early_dt=1.0, late_dt=1.0,
                           stop_dry=True, keep_short=True,
                           compact_output=True, output_stride=1,
                           environment='last'),
        # 问题4：采用半径观测与材料4物性，按分钟保存状态
        'q4_final': Config(material=4, shrink=True, n=160, grid_power=2.0,
                           early_dt=1.0, late_dt=15.0,
                           stop_dry=True, keep_short=False,
                           compact_output=False, output_stride=60,
                           environment='last'),
        # 粗网格对照配置用于检查空间离散影响，不替代正式结果
        'q3_coarse': Config(material=3, n=40, grid_power=2.0,
                            early_dt=1.0, late_dt=30.0, stop_dry=True),
        'q4_coarse': Config(material=4, shrink=True, n=40, grid_power=2.0,
                            early_dt=1.0, late_dt=30.0, stop_dry=True),
    }


# ==================== 按算例执行数值计算 ====================
if __name__ == '__main__':
    for name, cfg in configurations().items():
        run_case(name, cfg)
