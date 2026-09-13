"""温湿传递求解器的数值合同、解析退化及实现一致性测试。"""
import unittest
import importlib.util
import numpy as np

# ==================== 物理退化、解析解与数值合同检验 ====================
class TestFlux(unittest.TestCase):
    # 核对界面扩散积分与高精度基准的一致性
    def test_integrated_diffusivity(self):
        self.assertIsNotNone(importlib.util.find_spec('model'), 'integral-flux solver is not implemented yet')
        from model import integral_diffusivity
        actual=integral_diffusivity(3,np.array([.05]),np.array([.15]),np.array([50.165]),order=16)
        self.assertAlmostEqual(float(actual[0])/2.580097788700743e-10,1.,places=10)

    # 验证相邻含水率相等时界面积分退化为局部扩散系数
    def test_equal_concentrations(self):
        self.assertIsNotNone(importlib.util.find_spec('model'), 'integral-flux solver is not implemented yet')
        from model import integral_diffusivity, properties
        c=np.array([.05,.15,1.,2.55]);t=np.array([28.,35.,50.,50.165])
        ref=properties(3,c,t)[2]
        np.testing.assert_allclose(integral_diffusivity(3,c,c,t),ref,rtol=1e-13)

    # 检验无通量边界的离散有效量平衡及均匀平衡态保持
    def test_no_flux_conservation_and_equilibrium(self):
        from model import Solver,Config
        s=Solver(Config(n=40));u=1+s.x**2
        new=s.linear_step(u,np.full(40,1e-8),np.ones(41),0.,0.,.02,30.)
        self.assertLess(abs(float(s.volume@(new-u))),1e-13)
        same=s.linear_step(np.full(41,.7),np.full(40,1e-8),np.ones(41),8e-7,.7,.02,30.)
        np.testing.assert_allclose(same,.7,rtol=1e-13)

    # 以圆柱Bessel解析衰减解检查径向离散与隐式时间推进
    def test_cylindrical_bessel_decay(self):
        from scipy.special import j0,j1
        from model import Solver,Config
        s=Solver(Config(n=80));D=1e-7;R=.02;z=1.
        h=D*z*j1(z)/(R*j0(z));u=j0(z*s.x)
        for _ in range(1000):u=s.linear_step(u,np.full(80,D),np.ones(81),h,0.,R,.1)
        exact=j0(z*s.x)*np.exp(-D*z*z*100/R**2)
        self.assertLess(float(np.max(abs(u-exact))),1e-5)

    # 检验交换相邻节点后的扩散系数对称性及正性
    def test_integral_flux_symmetry_and_positivity(self):
        from model import integral_diffusivity
        a=np.array([.05,.15,1.]);b=np.array([.2,2.55,.06]);t=np.array([28.,50.,40.])
        f=integral_diffusivity(3,a,b,t,16)
        np.testing.assert_allclose(f,integral_diffusivity(3,b,a,t,16),rtol=1e-13)
        self.assertTrue(np.all(f>0))

    # 检验线性剖面阈值交点与未达标截面积比例
    def test_wet_fraction_crossing(self):
        from model import Solver,Config
        s=Solver(Config(n=20));c=.2-.1*s.x
        snap=s.snapshot(0,np.full(21,28.),c)
        self.assertAlmostEqual(snap['wet_fraction'],.25,places=12)

    # 检验多段未达标区域以及阈值相等状态的面积计数
    def test_wet_fraction_multiple_regions_and_threshold_equality(self):
        from model import Solver,Config
        s=Solver(Config(n=4,grid_power=1))
        snap=s.snapshot(0,np.full(5,28.),np.array([.2,.1,.2,.1,.2]))
        # Wet intervals: [0,1/8], [3/8,5/8], [7/8,1].
        self.assertAlmostEqual(snap['wet_fraction'],.5,places=12)
        equal=s.snapshot(0,np.full(5,28.),np.full(5,.15))
        self.assertEqual(equal['wet_fraction'],1.)

    # 检验收缩半径不变时退化为固定半径模型
    def test_zero_shrinkage_reduces_to_fixed_radius(self):
        from model import Solver,Config
        class ConstantRadiusInputs:
            def boundary(self,t,environment='last'): return 50.,.05
            def radius(self,t,shrink): return .02
        args=dict(material=4,n=10,max_time=60,stop_dry=False)
        a=Solver(Config(**args,shrink=False),ConstantRadiusInputs()).run()
        b=Solver(Config(**args,shrink=True),ConstantRadiusInputs()).run()
        np.testing.assert_array_equal(a['final']['T'],b['final']['T'])
        np.testing.assert_array_equal(a['final']['C'],b['final']['C'])

    # 检验移动域采样采用真实半径且域外结果保持为空
    def test_moving_sample_uses_physical_radius_and_outside_blanks(self):
        from model import sample
        snapshot={'radius_cm':1.2,'C':[.6,.4,.2]}
        actual=sample(snapshot,'C',[0.,.5,1.,1.2,1.3,2.])
        np.testing.assert_allclose(actual[:4],[.6,13/30,4/15,.2],atol=1e-14)
        self.assertEqual(actual[4:],[None,None])

    # 检验半径观测范围外的请求被拒绝，不静默外推
    def test_radius_data_do_not_silently_extrapolate(self):
        from model import Inputs
        inputs=Inputs()
        with self.assertRaisesRegex(ValueError,'extrapolation'):
            inputs.radius(float(inputs.radii[-1,0])+1,True)


    # 检验自适应时间推进准确落在逐秒输出时刻
    def test_full_output_lands_on_every_second(self):
        from model import Solver,Config
        result=Solver(Config(n=10,early_dt=10,keep_short=True,
                             output_stride=1,max_time=60,stop_dry=False)).run()
        self.assertEqual([s['t'] for s in result['short']],list(range(61)))

    # 检验名义边界系数情景与主求解器状态逐值一致
    def test_boundary_study_nominal_preserves_kernel(self):
        from model import Solver,Config
        from p2_study import BoundaryStudy
        for moving in (False,True):
            cfg=Config(material=4 if moving else 3,shrink=moving,n=160,
                       max_time=120,stop_dry=False,output_stride=1)
            a=Solver(cfg).run(); b=BoundaryStudy(cfg).run()
            for field in ('T','C'):
                np.testing.assert_array_equal(a['final'][field],b['final'][field])
            self.assertEqual(a['stats']['steps'],b['stats']['steps'])

    # 检验独立圆环实现的无通量离散平衡与均匀态保持
    def test_independent_reference_annulus_conservation(self):
        from model import Config,Inputs
        from p2_reference import IndependentBDF2
        s=IndependentBDF2(Config(n=40),Inputs())
        old=1+s.x**2
        new=s.solve(old,old,np.full(40,1e-8),np.ones(41),.02,0.,0.,(1/30,-1/30,0))
        self.assertLess(abs(float(s.dv@(new-old))),1e-13)
        same=s.solve(np.full(41,.7),np.full(41,.7),np.full(40,1e-8),
                     np.ones(41),.02,8e-7,.7,(1/30,-1/30,0))
        np.testing.assert_allclose(same,.7,rtol=1e-13)

    # 以真实附件对照主求解器与独立参考的短程温湿场
    def test_independent_reference_real_inputs(self):
        from model import Solver,Config,Inputs
        from p2_reference import IndependentBDF2
        for moving in (False,True):
            cfg=Config(material=4 if moving else 3,shrink=moving,n=160,
                       max_time=120,stop_dry=False,output_stride=1)
            a=Solver(cfg).run();b=IndependentBDF2(cfg,Inputs()).run()
            np.testing.assert_allclose(a['final']['T'],b['final']['T'],rtol=0,atol=1e-6)
            np.testing.assert_allclose(a['final']['C'],b['final']['C'],rtol=0,atol=1e-9)

    # 检验输出存储选项不改变有效计算轨迹
    def test_sampling_does_not_change_valid_trajectory(self):
        from model import Solver,Config
        args=dict(material=1,n=10,early_dt=.5,max_time=60,stop_dry=False)
        a=Solver(Config(**args,keep_short=False)).run()
        b=Solver(Config(**args,keep_short=True)).run()
        np.testing.assert_array_equal(a['final']['C'],b['final']['C'])
        self.assertEqual(a['stats']['steps'],b['stats']['steps'])

    # 检验输入字节变化导致缓存来源指纹变化
    def test_cache_provenance_covers_input_bytes(self):
        import tempfile
        from pathlib import Path
        from run_study import provenance
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'input';p.write_bytes(b'first')
            a=provenance([p]);p.write_bytes(b'second');b=provenance([p])
            self.assertNotEqual(a,b)

# ==================== 执行数值验证集合 ====================
if __name__=='__main__': unittest.main()
