"""从冻结的v0.4轨迹缓存生成论文中的观测数据图、过程图与径向结果图。"""
from pathlib import Path
import gzip, json, sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import shutil

from model import ROOT, sample


# 读取保存的正式数值轨迹，图像生成不重新执行模型求解
def load(name):
    with gzip.open(ROOT / 'runs' / f'{name}.json.gz', 'rt', encoding='utf8') as f:
        return json.load(f)


# 将同一图像导出多种格式，并复制论文使用的PDF图源
def save(fig, stem):
    for extension in ('pdf','svg','png'):
        path=ROOT/'figures'/f'{stem}.{extension}'
        fig.savefig(path,dpi=180,bbox_inches='tight')
        if extension=='pdf':
            target=ROOT.parent/'完整论文-LaTeX/figs'
            target.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target/path.name)
    plt.close(fig)


# ==================== 观测数据与模型背景图 ====================
def main():
    plt.rcParams.update({'font.family':['DejaVu Serif','SimSun'],
        'font.size':11,'axes.unicode_minus':False,'mathtext.fontset':'dejavuserif',
        'pdf.fonttype':42,'svg.fonttype':'path'})
    q1, q2, q4 = load('q1_final'), load('q2_full'), load('q4_final')
    q3 = q2
    figdir = ROOT / 'figures'
    figdir.mkdir(exist_ok=True)
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    # Raw data figures: observed environmental and radius data, plus derived ranges.
    # q1: two observed air variables, separate panels because units differ.
    from model import Inputs
    inp = Inputs()
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 4.2), constrained_layout=True)
    ax[0].plot(inp.air[:, 0] / 3600, inp.air[:, 1], color=colors[0])
    ax[0].set(xlabel='时间 / h', ylabel='烘房温度 / °C', title='附件1：烘房温度')
    ax[1].plot(inp.air[:, 0] / 3600, inp.air[:, 2], color=colors[2])
    ax[1].set(xlabel='时间 / h', ylabel='有效水分浓度 / (kg/kg)', title='附件1：烘房水分浓度')
    for a in ax: a.spines[['top', 'right']].set_visible(False)
    save(fig, 'raw_q1_environment')
    # q2: input range summary as a relationship between environment variables.
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(inp.air[:, 1], inp.air[:, 2], color=colors[0], lw=1.0, marker='o', ms=1.5)
    ax.set(xlabel='烘房温度 / °C', ylabel='烘房水分浓度 / (kg/kg)', title='附件1：环境状态关系')
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'raw_q2_environment_relation')
    # q3: radius-independent observed data context.
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.scatter(inp.radii[:, 0] / 3600, inp.radii[:, 1], s=12, color=colors[1], label='观测')
    ax.plot(inp.radii[:, 0] / 3600, inp.radii[:, 1], color=colors[1], alpha=.6)
    ax.set(xlabel='时间 / h', ylabel='半径 / cm', title='附件2：药材半径观测与线性插值')
    ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'raw_q3_radius')
    # q4 raw: combined normalized radius and environmental temperature.
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(inp.radii[:, 0] / 3600, inp.radii[:, 1] / inp.radii[0, 1], color=colors[1])
    ax.set(xlabel='时间 / h', ylabel='R(t) / R(0)', title='收缩几何的无量纲半径')
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'raw_q4_shrinkage')

    # Process figures.
    # ==================== 温湿状态与收缩过程图 ====================
    s1 = q1['long']
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    tt = np.array([s['t'] for s in s1]) / 3600
    ax.plot(tt, [s['T'][0] for s in s1], color=colors[0], label='中心')
    ax.plot(tt, [s['T'][-1] for s in s1], color=colors[1], label='表面')
    ax.set(xlabel='时间 / h', ylabel='温度 / °C', title='问题1：预热阶段中心—表面温度')
    ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'process_q1_temperature')
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ss = q2['long']; tt = np.array([s['t'] for s in ss]) / 3600
    ax.plot(tt, [s['C'][0] for s in ss], color=colors[0], label='中心')
    ax.plot(tt, [s['C'][-1] for s in ss], color=colors[1], label='表面')
    ax.set(xlabel='时间 / h', ylabel='干基含水率 / (kg/kg)', title='问题2—3：固定半径失水过程')
    # 低含水率阶段采用对数纵轴展示，避免中心与表面曲线差异被掩盖
    ax.set_yscale('log'); ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'process_q2_moisture')
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(tt, [100*s['wet_fraction'] for s in ss], color=colors[2])
    ax.axhline(0, color='#555', lw=.8)
    ax.set(xlabel='时间 / h', ylabel='未达标截面积比例 / %', title='问题3：未达标核心面积变化')
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'process_q3_wet_fraction')
    ss4 = q4['long']; tt4 = np.array([s['t'] for s in ss4]) / 3600
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 4.2), constrained_layout=True)
    ax[0].plot(tt4, [s['radius_cm'] for s in ss4], color=colors[1])
    ax[0].set(xlabel='时间 / h', ylabel='半径 / cm', title='收缩半径')
    ax[1].plot(tt4, [s['C'][0] for s in ss4], color=colors[0])
    ax[1].set(xlabel='时间 / h', ylabel='中心含水率 / (kg/kg)', title='中心含水率')
    for a in ax: a.spines[['top', 'right']].set_visible(False)
    save(fig, 'process_q4_shrinkage_moisture')

    # Result figures: spatial fields and endpoint profiles.
    # ==================== 径向结果与干燥终点图 ====================
    dist = np.arange(0, 2.001, .1)
    snap1 = q1['summaries']['1800']; valsT = sample(snap1, 'T', dist); valsC = sample(snap1, 'C', dist)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(dist, valsT, color=colors[0], label='温度')
    ax.set(xlabel='到中心距离 / cm', ylabel='温度 / °C', title='问题1：1800 s径向温度结果')
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'result_q1_profile')
    snap2 = q2['summaries']['10800']; vals = sample(snap2, 'C', dist)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(dist, vals, color=colors[2], marker='o', ms=2, label='3 h')
    ax.set(xlabel='到中心距离 / cm', ylabel='干基含水率 / (kg/kg)', title='问题2：3 h径向含水率结果')
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'result_q2_profile')
    vals = sample(q2['final'], 'C', dist)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(dist, vals, color=colors[0], marker='o', ms=2)
    ax.axhline(.15, color='#555', ls='--', label='阈值 0.15')
    ax.set(xlabel='到中心距离 / cm', ylabel='干基含水率 / (kg/kg)', title='问题3：首次达标时径向含水率')
    ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'result_q3_endpoint')
    # 问题4以固定物理距离展示终点剖面，域外位置不外推
    vals = sample(q4['final'], 'C', dist)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.plot(dist, vals, color=colors[1], marker='o', ms=2, label='有效域采样')
    ax.axhline(.15, color='#555', ls='--', label='阈值 0.15')
    ax.set(xlabel='固定物理距离 / cm', ylabel='干基含水率 / (kg/kg)', title='问题4：收缩终点物理距离剖面')
    ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'result_q4_endpoint')
    print('figures generated: 12 logical figures')
if __name__ == '__main__':
    main()
