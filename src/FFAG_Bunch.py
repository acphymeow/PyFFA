import numpy as np
from FFAG_track import FFAG_MPI
import copy
import os
import time
from FFAG_SC import find_two_points_non_uniform_vect_njit, DistributeChargeNjit, find_two_points_uniform_vectorized
from FFAG_ParasAndConversion import FFAG_GlobalParameters
import pyfftw
# 缓存避免每次都重建 FFTW 计划
pyfftw.interfaces.cache.enable()
# 设置fft线程数为1,避免线程冲突
pyfftw.config.NUM_THREADS = 1
from mpi4py import MPI
from FFAG_SC_new import (divide_arc2, merge_multiple_arcs, DistributeCharge2D_Njit, GetLocalCoordinates_2p5,
                         ComputeSliceBoundsAndCoords_merged, make_fft_kernel_open_numba, total_field_1d_numba,
                         compute_field_from_potential_numba,interpolate_fields_to_particles_numba,
                         DistributeChargeSBS_Njit, ComputeSliceBoundsAndCoordsSBS_merged, compute_field_from_potential_numba_SBS,
                         interpolate_fields_to_particles_SBS_numba, allgatherv_phi_3d, allgatherv_phi_3d_inplace, contiguous_block_for_rank, GetLocalCoordinates_2p5_rk)
from FFAG_MathTools import UpdataPreStepNjit, Bilinear_interp_2D_vect_uniform



class FFAG_ManageBunchAttribute:
    def __init__(self):
        # 初始化属性字典
        self.Attribute = dict()
        self.Attribute['r'] = int(0)
        self.Attribute['vr'] = int(1)
        self.Attribute['z'] = int(2)
        self.Attribute['vz'] = int(3)
        self.Attribute['fi'] = int(4)
        self.Attribute['Ek'] = int(5)
        self.Attribute['inj_t'] = int(6)
        self.Attribute['Inj_flag'] = int(7)
        self.Attribute['Survive'] = int(8)
        self.Attribute['RF_phase'] = int(9)
        self.Attribute['Esc_r'] = int(10)
        self.Attribute['Esc_z'] = int(11)
        self.Attribute['Esc_fi'] = int(12)
        self.Attribute['Bunch_ID'] = int(13)
        self.Attribute['Local_ID'] = int(14)
        self.Attribute['Global_ID'] = int(15)
        #

        self.AttributeFormat = dict()  # 新增的属性格式字典
        # 定义属性的保存格式
        self.AttributeFormat['r'] = '%.8e'
        self.AttributeFormat['vr'] = '%.8e'
        self.AttributeFormat['z'] = '%.8e'
        self.AttributeFormat['vz'] = '%.8e'
        self.AttributeFormat['fi'] = '%.8e'
        self.AttributeFormat['Ek'] = '%.8e'
        self.AttributeFormat['inj_t'] = '%.6e'
        self.AttributeFormat['Inj_flag'] = '%d'
        self.AttributeFormat['Survive'] = '%d'
        self.AttributeFormat['RF_phase'] = '%.6e'
        self.AttributeFormat['Esc_r'] = '%.6e'
        self.AttributeFormat['Esc_z'] = '%.6e'
        self.AttributeFormat['Esc_fi'] = '%.6e'
        self.AttributeFormat['Bunch_ID'] = '%d'
        self.AttributeFormat['Local_ID'] = '%d'
        self.AttributeFormat['Global_ID'] = '%d'

    def get_num_attributes(self):
        """
        获取当前属性的个数，即Bunch矩阵的列数。
        """
        return len(self.Attribute)

    def get_attribute_names(self):
        """
        获取所有属性名称的列表。
        """
        return list(self.Attribute.keys())


class FFAG_Bunch:

    def __init__(self, ParticlesDistribution, marcosize=1, sc_grid_size=(64, 128,128, 12.0), BunchType='Boris'):
        # marcosize is the scaling factor between
        # the simulated macro particles and the actual particle count.
        self.BunchAttribute = FFAG_ManageBunchAttribute()
        self.BunchType = BunchType # Boris RK4

        # Get the Global particle number
        self.TotalParticleNum = np.size(ParticlesDistribution, 0)
        self.marcosize = marcosize
        self.speed_c = 2.99792458e8
        self.q_charge = 1.60217662e-19
        self.E0_rest_MeV = 938.2723

        # StaticBunch represents the static initial distribution that will remain unchanged
        # GlobalBunch and LocalBunch are dynamic and will change in each step
        # (r, vr), (z, vz), (fi, dfidt), (t_inj, inj_flag,survive_flag),
        # (rf_phase, Esc_r, Esc_z, Esc_fi)
        # (Bunch_ID, Local_ID, Global_ID)

        # divide particles for threads
        PIDGlobal = ParticlesDistribution[:, self.BunchAttribute.Attribute['Global_ID']]
        PIDLocal = FFAG_MPI().DivideVariables(PIDGlobal)
        ParticlesDistribution_local = ParticlesDistribution[PIDLocal, :]

        # generate local Static ini bunch
        self.StaticBunchLocal = copy.deepcopy(ParticlesDistribution_local)
        self.TotalParticleNum_Local = np.size(self.StaticBunchLocal, 0)
        # generate the local id
        self.StaticBunchLocal[:, self.BunchAttribute.Attribute['Local_ID']] = np.arange(self.TotalParticleNum_Local)

        self.LocalBunch = copy.deepcopy(self.StaticBunchLocal)

        BunchMatrixColNum = self.BunchAttribute.get_num_attributes()  # Bunch矩阵的列数 Bunch的属性数
        self.LocalLostBunch = np.zeros((0, BunchMatrixColNum))  # 初始化Local束损Bunch矩阵
        self.GlobalLostBunch = np.zeros((0, BunchMatrixColNum))  # 初始化Global束损Bunch矩阵
        # self.LocalSurviveBunch = np.zeros((0, BunchMatrixColNum))
        self.LocalUnInjectBunch = copy.deepcopy(self.StaticBunchLocal)

        self._uninject_head = 0
        # 注入粒子数，存活粒子数，损失粒子数
        self.uninject_num = self.TotalParticleNum_Local  # local未注入粒子数
        self.InjectedNum_Local = 0  # local已注入粒子数
        self.SurvivedNum_Local = 0  # local存活粒子数
        self.LostNum_Local = 0  # local损失粒子数

        self.InjectedNum = 0  # global已注入粒子数
        self.SurvivedNum = 0  # global存活粒子数
        self.LostNum = 0  # global损失粒子数

        # SC grid size
        self.x_grid_size = sc_grid_size[1]
        self.z_grid_size = sc_grid_size[2]
        self.f_grid_size = sc_grid_size[0]
        self.sigma_ratio = sc_grid_size[3]

        # malloc 中间变量
        self.PreStep_FiModMalloc = np.zeros(self.TotalParticleNum_Local, )
        self.PostStep_FiModMalloc = np.zeros(self.TotalParticleNum_Local, )
        self.PreStep_Fi = np.zeros(self.TotalParticleNum_Local, )
        self.PostStep_Fi = np.zeros(self.TotalParticleNum_Local, )
        self.Step_SurviveFlag = np.zeros(self.TotalParticleNum_Local, dtype=np.bool_)
        self.Step_LostFlag = np.zeros(self.TotalParticleNum_Local, dtype=np.bool_)
        self.CrossingCheckBoolMalloc = np.zeros(self.TotalParticleNum_Local, dtype=np.bool_)
        self.CrossingCheckRatioMalloc = np.zeros(self.TotalParticleNum_Local, )
        self.PreStepMat = np.zeros((self.TotalParticleNum_Local, 7))

        self.Malloc_SC_Ex_Pre_Pre = np.zeros((self.TotalParticleNum_Local, ))  #前2个SC step的SC电场分量, 用于插值
        self.Malloc_SC_Ez_Pre_Pre = np.zeros((self.TotalParticleNum_Local, ))
        self.Malloc_SC_Ef_Pre_Pre = np.zeros((self.TotalParticleNum_Local, ))
        self.Malloc_SC_Ex_Pre = np.zeros((self.TotalParticleNum_Local, ))  #前1个SC step的SC电场分量, 用于插值
        self.Malloc_SC_Ez_Pre = np.zeros((self.TotalParticleNum_Local, ))
        self.Malloc_SC_Ef_Pre = np.zeros((self.TotalParticleNum_Local, ))

        self.Malloc_x_cart = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_y_cart = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_z_cart = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_vx_cart = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_vy_cart = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_vz_cart = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_Ex_cart = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_Ey_cart = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_Ez_cart = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_Bx_cart = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_By_cart = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_Bz_cart = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_RF_phase = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_mod_fi = np.zeros(self.TotalParticleNum_Local, )

        self.Malloc_x_new = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_y_new = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_z_new = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_vx_new = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_vy_new = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_vz_new = np.zeros(self.TotalParticleNum_Local, )

        self.Malloc_Bz_interp = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_Br_interp = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_Bf_interp = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_Ez_interp = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_Er_interp = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_Ef_interp = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_BzCoef_interp = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_BrCoef_interp = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_BfCoef_interp = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_RF_shift = np.zeros(self.TotalParticleNum_Local, dtype=np.int32)  # 在第几个RF gap

        self.Malloc_SC_r = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_SC_fi = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_SC_z = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_SC_gamma = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_SC_mean_gamma_interp = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_SC_r_SEO = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_SC_pr_SEO = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_SC_interp_flag = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_SC_x = np.zeros(self.TotalParticleNum_Local, )

        self.Malloc_SC_fiMod = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_SC_surviveIDX = np.zeros(self.TotalParticleNum_Local, dtype=np.int32)
        self.Malloc_SC_fiMod_arc = np.zeros(self.TotalParticleNum_Local, )

        self.Malloc_SC_slice_r2d = np.zeros((self.f_grid_size, self.TotalParticleNum_Local), dtype=np.float64)
        self.Malloc_SC_slice_z2d = np.zeros((self.f_grid_size, self.TotalParticleNum_Local), dtype=np.float64)
        self.Malloc_SC_slice_x2d = np.zeros((self.f_grid_size, self.TotalParticleNum_Local), dtype=np.float64)
        self.Malloc_SC_slice_LID = np.zeros((self.f_grid_size, self.TotalParticleNum_Local), dtype=np.int32)
        self.Malloc_SC_slice_indices = np.zeros(self.TotalParticleNum_Local, dtype=np.int32)
        self.Malloc_SC_slice_sumZ = np.zeros(self.f_grid_size, dtype=np.float64)
        self.Malloc_SC_slice_sumR = np.zeros(self.f_grid_size, dtype=np.float64)
        self.Malloc_SC_slice_sumZ2 = np.zeros(self.f_grid_size, dtype=np.float64)
        self.Malloc_SC_slice_sumR2 = np.zeros(self.f_grid_size, dtype=np.float64)

        self.Malloc_SC_Exmap = np.zeros((self.z_grid_size, self.x_grid_size), dtype=np.float32)
        self.Malloc_SC_Ezmap = np.zeros((self.z_grid_size, self.x_grid_size), dtype=np.float32)
        # self.Malloc_SC_Exmap_3D = np.zeros((self.f_grid_size,self.z_grid_size, self.x_grid_size), dtype=np.float32)
        # self.Malloc_SC_Ezmap_3D = np.zeros((self.f_grid_size,self.z_grid_size, self.x_grid_size), dtype=np.float32)

        self.Malloc_SC_Ex_p = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_SC_Ez_p = np.zeros(self.TotalParticleNum_Local, )
        self.Malloc_SC_Ef_p = np.zeros(self.TotalParticleNum_Local, )

        self.Malloc_SC_survive_flag_PrePre = np.zeros(self.TotalParticleNum_Local, dtype=bool)
        self.Malloc_SC_survive_flag_Pre = np.zeros(self.TotalParticleNum_Local, dtype=bool)


        self.Malloc_RK_dxdt = np.zeros((self.TotalParticleNum_Local, 16))

        # Add Pre_step and Post_step for tracking (x, y, z, fi)
        # self.Pre_steps = np.zeros((self.TotalParticleNum_Local, 6))  # Pre-step positions (r, z, fi, fi, survive_flag, GlobalID)
        # self.Post_steps = np.zeros((self.TotalParticleNum_Local, 6))  # Post-step positions (x, y, z, fi, survive_flag, GlobalID)
        # self.survive_mask = None

        self.point_x = None
        self.point_y = None
        self.point_z = None

        self.xmin_Local = None
        self.xmax_Local = None
        self.ymin_Local = None
        self.ymax_Local = None
        self.zmin_Local = None
        self.zmax_Local = None

        self.xmin_Global = None
        self.xmax_Global = None
        self.ymin_Global = None
        self.ymax_Global = None
        self.zmin_Global = None
        self.zmax_Global = None

        self.Xgrid = None
        self.Ygrid = None
        self.Zgrid = None
        self.Xmatrix = None
        self.Ymatrix = None
        self.Zmatrix = None

        self.charge_distribution_local = None
        self.charge_distribution_global = None

        self.SEO_fi_axis = None
        self.SEO_Ek_axis = None
        self.SEO_r_matrix = None
        self.SEO_pr_matrix = None
        self.SEO_perimeter = None


    # @profile
    def InjectParticles(self, t_threshold, t_step):
        """
        将符合注入条件的粒子注入系统,并更新相关粒子数统计,每个step调用一次。
        :param t_threshold: 当前时间阈值
        """
        inj_time_idx = self.BunchAttribute.Attribute['inj_t']
        inj_flag_idx = self.BunchAttribute.Attribute['Inj_flag']
        survive_flag_idx = self.BunchAttribute.Attribute['Survive']

        # 纳秒单位转换
        inj_threshold_high = t_threshold * 1e9
        inj_threshold_low = (t_threshold - t_step) * 1e9

        # 获取当前未注入粒子区域
        LocalUnInjectBunchValid = self.LocalUnInjectBunch[self._uninject_head:, ]
        inj_times = LocalUnInjectBunchValid[:, inj_time_idx]

        # 计算注入范围 [low, high)
        start_idx = np.searchsorted(inj_times, inj_threshold_low, side='left')
        end_idx = np.searchsorted(inj_times, inj_threshold_high, side='right')

        num_to_inject_local = end_idx - start_idx
        if num_to_inject_local > 0:
            # 计算原始数组上的绝对索引范围
            start = self._uninject_head + start_idx
            end = self._uninject_head + end_idx

            # 更新注入标记
            self.LocalBunch[start:end, inj_flag_idx] = 1
            self.LocalBunch[start:end, survive_flag_idx] = 1
            self.Step_SurviveFlag[start:end] = True

            # 更新指针
            self._uninject_head = end
            self.uninject_num -= num_to_inject_local
            self.InjectedNum_Local += num_to_inject_local
            self.SurvivedNum_Local += num_to_inject_local  # 注入即为存活

        return 0

    def restore_from_restart_matrix(self, *, assert_monotonic=True) -> None:
        """
        设计的结构：
          - LocalBunch 全局按 inj_t 升序
          - Inj_flag 形成阶跃：[1...1, 0...0]
          - _uninject_head = 第一个 Inj_flag==0 的位置（若全是1则=N；若全是0则=0）
          - uninject_num / InjectedNum_Local / SurvivedNum_Local / LostNum_Local / Step_SurviveFlag 同步恢复
        """
        A = self.BunchAttribute.Attribute
        inj_t_idx = A["inj_t"]
        inj_flag_idx = A["Inj_flag"]
        survive_idx = A["Survive"]

        N = self.LocalBunch.shape[0]
        if N == 0:
            self._uninject_head = 0
            self.uninject_num = 0
            self.InjectedNum_Local = 0
            self.SurvivedNum_Local = 0
            self.LostNum_Local = 0
            self.Step_SurviveFlag = np.zeros(0, dtype=bool)
            self.LocalUnInjectBunch = self.LocalBunch
            return

        # 2) 读 flag
        inj = (self.LocalBunch[:, inj_flag_idx] > 0.5)
        sur = (self.LocalBunch[:, survive_idx] > 0.5)

        # 3) 找阶跃边界：第一个 False（未注入）的位置
        #    - 如果全 True：head = N
        #    - 如果全 False：head = 0
        zeros = np.flatnonzero(~inj)
        head = int(zeros[0]) if zeros.size else N
        self._uninject_head = head

        # 4) inj 必须是 [True...True, False...False]
        if assert_monotonic:
            if head < N:
                # head 之前不允许出现 False；head 之后不允许出现 True
                ok = inj[:head].all() and (~inj[head:]).all()
            else:
                ok = inj.all()
            if not ok:
                # 反例1：head 后面出现 True
                bad1 = np.flatnonzero(inj[head:]) + head
                # 反例2：head 前面出现 False
                bad2 = np.flatnonzero(~inj[:head])
                raise RuntimeError(
                    "Restart bunch violates the designed structure: Inj_flag is not a step function "
                    "along inj_t-sorted LocalBunch. "
                    f"Computed head={head}, first bad-before={bad2[0] if bad2.size else None}, "
                    f"first bad-after={bad1[0] if bad1.size else None}."
                )

        # 5) 统计量（local）
        self.InjectedNum_Local = int(inj.sum())
        self.uninject_num = int(N - self.InjectedNum_Local)
        self.SurvivedNum_Local = int(sur.sum())
        self.LostNum_Local = int((inj & (~sur)).sum())

        # 6) 推进Flag：已注入且存活
        self.Step_SurviveFlag = inj & sur



    # @profile
    def UpdateParticleNumGlobal(self):
        """
        使用MPI收集所有进程的本地粒子数，并更新全局粒子数统计。
        """
        # 初始化MPI通信
        comm = MPI.COMM_WORLD

        # 汇总本地粒子数信息
        global_survived_num = comm.reduce(self.SurvivedNum_Local, op=MPI.SUM, root=0)
        global_lost_num = comm.reduce(self.LostNum_Local, op=MPI.SUM, root=0)
        global_inject_num = comm.reduce(self.InjectedNum_Local, op=MPI.SUM, root=0)

        # 仅在主进程（rank == 0）更新全局统计量
        if comm.rank == 0:
            self.InjectedNum = global_inject_num  # global已注入粒子数
            self.SurvivedNum = global_survived_num  # global存活粒子数
            self.LostNum = global_lost_num  # global损失粒子数

        # 同步全局统计信息到所有进程
        self.InjectedNum = comm.bcast(self.InjectedNum, root=0)
        self.SurvivedNum = comm.bcast(self.SurvivedNum, root=0)
        self.LostNum = comm.bcast(self.LostNum, root=0)

    # @profile
    def UpdatePreSteps(self):
        """
        每个step进行积分前调用，读取LocalBunch矩阵中的坐标，更新PreSteps。
        """
        # fi_idx = self.BunchAttribute.Attribute['fi']
        # self.PreStep_Fi = self.LocalBunch[:, fi_idx].copy()
        UpdataPreStepNjit(self.PreStep_Fi, self.PreStepMat, self.LocalBunch, self.Step_SurviveFlag)


    def UpdatePostSteps(self):
        """
        每个step进行积分后调用(在DeleteParticles后)，读取LocalBunch矩阵中的坐标，更新PostSteps。
        """
        fi_idx = self.BunchAttribute.Attribute['fi']
        self.PostStep_Fi = self.LocalBunch[:, fi_idx]


    def DeleteParticles(self, DeleteIndexLocal):
        """
        删除本地指定粒子，并将其移动到 LocalLostBunch，同时更新相关标志和统计。

        参数:
            DeleteIndexLocal : np.ndarray[int]
            指示要删除的粒子本地索引。
        """
        survive_flag_idx = self.BunchAttribute.Attribute['Survive']

        # 1. 将粒子复制到 LocalLostBunch（保留状态记录）
        self.LocalLostBunch = np.row_stack(
            (self.LocalLostBunch, self.LocalBunch[DeleteIndexLocal, :])
        )

        # 2. 设置为损失状态：survive_flag = 0
        self.LocalBunch[DeleteIndexLocal, survive_flag_idx] = 0

        # 4. 更新逐步状态标记
        self.Step_SurviveFlag[DeleteIndexLocal] = False
        self.Step_LostFlag[DeleteIndexLocal] = True

        # 5. 更新本地损失粒子计数
        self.LostNum_Local += DeleteIndexLocal.size
        self.SurvivedNum_Local -= DeleteIndexLocal.size



    def extend_matrix_y_direction(self, matrix, axis=1):
        """
        扩展一个 3D 矩阵沿 y 方向（指定 axis）左右各扩展一半范围。

        参数：
            matrix (np.ndarray): 输入的 3D 矩阵。
            axis (int): 要扩展的轴，默认为 1（沿 y 方向）。

        返回：
            np.ndarray: 沿 y 方向扩展后的 3D 矩阵。
        """
        if matrix.ndim != 3:
            raise ValueError("输入矩阵必须是 3 维矩阵")

        # 计算扩展范围
        slice_size = matrix.shape[axis] // 2

        # 获取左右两部分切片
        left_extension = np.take(matrix, range(slice_size), axis=axis) - (matrix.max() - matrix.min())
        right_extension = np.take(matrix, range(-slice_size, 0), axis=axis) + (matrix.max() - matrix.min())

        # 拼接矩阵
        extended_matrix = np.concatenate([left_extension, matrix, right_extension], axis=axis)

        return extended_matrix


    def GetLocalCoordinates(self):
        """
        返回本地 Bunch 中所有存活粒子的 x, y, z 笛卡尔坐标（n, 3 的 ndarray），
        以及 LocalID 和 GlobalID（分别为 n, 的整数类型的 ndarray）。
        """
        # 获取属性的索引
        r_idx = self.BunchAttribute.Attribute['r']
        fi_idx = self.BunchAttribute.Attribute['fi']
        z_idx = self.BunchAttribute.Attribute['z']
        survive_flag_idx = self.BunchAttribute.Attribute['Survive']
        local_id_idx = self.BunchAttribute.Attribute['Local_ID']
        global_id_idx = self.BunchAttribute.Attribute['Global_ID']

        # 筛选出本地存活的粒子survive_flag == 1, 排除已损失或未注入的粒子
        valid_particles_mask = self.LocalBunch[:, survive_flag_idx] == 1

        # 提取有效粒子的 r, fi, z 坐标
        r_polar = self.LocalBunch[valid_particles_mask, r_idx]
        fi_polar = self.LocalBunch[valid_particles_mask, fi_idx]
        z_cart = self.LocalBunch[valid_particles_mask, z_idx]

        # 将极坐标 (r, fi) 转换为笛卡尔坐标 (x, y)
        x_cart = r_polar * np.cos(fi_polar)
        y_cart = r_polar * np.sin(fi_polar)

        # 提取有效粒子的 LocalID 和 GlobalID，并将它们转换为整数
        local_id = self.LocalBunch[valid_particles_mask, local_id_idx].astype(int)
        global_id = self.LocalBunch[valid_particles_mask, global_id_idx].astype(int)

        # 将 x, y, z 组合为 (n, 3) 的 ndarray
        coordinates = np.column_stack((x_cart, y_cart, z_cart))

        # 返回笛卡尔坐标和整数类型的 LocalID、GlobalID
        return coordinates, local_id, global_id


    def GetLocalCoordinates_FlatCoordinate(self):
        """
        返回本地 Bunch 中所有存活粒子的 x, y, z 笛卡尔坐标（n, 3 的 ndarray），
        以及 LocalID 和 GlobalID（分别为 n, 的整数类型的 ndarray）。
        """
        # 获取属性的索引
        r_idx = self.BunchAttribute.Attribute['r']
        fi_idx = self.BunchAttribute.Attribute['fi']
        z_idx = self.BunchAttribute.Attribute['z']
        survive_flag_idx = self.BunchAttribute.Attribute['Survive']
        local_id_idx = self.BunchAttribute.Attribute['Local_ID']
        global_id_idx = self.BunchAttribute.Attribute['Global_ID']

        # 筛选出本地存活的粒子survive_flag == 1, 排除已损失或未注入的粒子
        valid_particles_mask = self.LocalBunch[:, survive_flag_idx] == 1

        # 提取有效粒子的 r, fi, z 坐标
        r_polar = self.LocalBunch[valid_particles_mask, r_idx]
        fi_polar = self.LocalBunch[valid_particles_mask, fi_idx]
        z_polar = self.LocalBunch[valid_particles_mask, z_idx]

        fi_mod_360 = np.mod(fi_polar, 2 * np.pi)
        if self.fmin_Global_flat is not None and self.fmin_Global_flat >= 0:
            fi_mapped = fi_mod_360
        else:
            fi_mod_180 =  fi_mod_360.copy()
            fi_mod_180[fi_mod_180 > np.pi] -= 2 * np.pi
            fi_mapped = fi_mod_180

        # Mapped Flat Coordinate System
        r_mapped = r_polar
        z_mapped = z_polar

        # 提取有效粒子的 LocalID 和 GlobalID，并将它们转换为整数
        local_id = self.LocalBunch[valid_particles_mask, local_id_idx].astype(int)
        global_id = self.LocalBunch[valid_particles_mask, global_id_idx].astype(int)

        # 将 x, y, z, turns组合为 (n, 3) 的 ndarray
        coordinates_flat = np.column_stack((r_mapped, fi_mapped, z_mapped))

        # 返回笛卡尔坐标和整数类型的 LocalID、GlobalID
        return coordinates_flat, local_id, global_id

    # @profile
    def GetLocalCoordinates_2p5(self):
        """
        返回本地 Bunch 中所有存活粒子的 x, y, z 笛卡尔坐标（n, 3 的 ndarray），
        以及 LocalID 和 GlobalID（分别为 n, 的整数类型的 ndarray）。
        """
        # 获取属性的索引
        r_idx = self.BunchAttribute.Attribute['r']
        fi_idx = self.BunchAttribute.Attribute['fi']
        z_idx = self.BunchAttribute.Attribute['z']
        survive_flag_idx = self.BunchAttribute.Attribute['Survive']
        local_id_idx = self.BunchAttribute.Attribute['Local_ID']
        global_id_idx = self.BunchAttribute.Attribute['Global_ID']

        # 筛选出本地存活的粒子survive_flag == 1, 排除已损失或未注入的粒子
        valid_particles_mask = self.LocalBunch[:, survive_flag_idx] == 1

        # 提取有效粒子的 r, fi, z 坐标
        r_polar = self.LocalBunch[valid_particles_mask, r_idx]
        fi_polar = self.LocalBunch[valid_particles_mask, fi_idx]
        z_polar = self.LocalBunch[valid_particles_mask, z_idx]


        # 返回笛卡尔坐标和整数类型的 LocalID、GlobalID
        return r_polar, fi_polar, z_polar

    # @profile
    def Get_SC_Grid_para_Local(self):
        """
        获取当前存活粒子的边界参数，并将极坐标转换为笛卡尔坐标。
        更新 self.xmin_Local, self.xmax_Local, self.ymin_Local,
        self.ymax_Local, self.zmin_Local, self.zmax_Local
        """
        survive_flag_idx = self.BunchAttribute.Attribute['Survive']

        # 筛选出存活粒子（survive_flag == 1）
        survived_mask = self.LocalBunch[:, survive_flag_idx] == 1
        if not np.any(survived_mask):
            # 如果没有存活粒子，将边界参数设为 None
            self.xmin_Local = None
            self.xmax_Local = None
            self.ymin_Local = None
            self.ymax_Local = None
            self.zmin_Local = None
            self.zmax_Local = None
            return

        # 提取局部存活粒子的 r, fi 和 z 坐标

        r_polar = self.LocalBunch[survived_mask, self.BunchAttribute.Attribute['r']]  # 粒子的径向坐标
        fi_polar = self.LocalBunch[survived_mask, self.BunchAttribute.Attribute['fi']]  # 粒子的方位角坐标
        z_cart = self.LocalBunch[survived_mask, self.BunchAttribute.Attribute['z']]  # 粒子的纵向坐标（z 方向）

        # r_polar, fi_polar, z_cart = filter_survived(self.LocalBunch, survived_mask,self.BunchAttribute.Attribute['r'],self.BunchAttribute.Attribute['fi'],self.BunchAttribute.Attribute['z'])

        # 将极坐标转换为笛卡尔坐标
        x_cart = r_polar * np.cos(fi_polar)
        y_cart = r_polar * np.sin(fi_polar)

        # 计算 x, y 和 z 坐标的最小值和最大值
        self.xmin_Local = np.nanmin(x_cart) if len(x_cart) > 1 else None
        self.xmax_Local = np.nanmax(x_cart) if len(x_cart) > 1 else None
        self.ymin_Local = np.nanmin(y_cart) if len(y_cart) > 1 else None
        self.ymax_Local = np.nanmax(y_cart) if len(y_cart) > 1 else None
        self.zmin_Local = np.nanmin(z_cart) if len(z_cart) > 1 else None
        self.zmax_Local = np.nanmax(z_cart) if len(z_cart) > 1 else None


    def Get_SC_Grid_para_Local_FlatCoordinate(self):
        """
        获取当前存活粒子的边界参数，并将极坐标转换为平直坐标。
        """
        survive_flag_idx = self.BunchAttribute.Attribute['Survive']

        # 筛选出存活粒子（survive_flag == 1）
        survived_mask = self.LocalBunch[:, survive_flag_idx] == 1
        if not np.any(survived_mask):
            # 如果没有存活粒子，将边界参数设为 None
            self.rmin_Local_flat = None
            self.rmax_Local_flat = None
            self.zmin_Local_flat = None
            self.zmax_Local_flat = None
            self.fmin_Local_flat = None
            self.fmax_Local_flat = None
            return

        # 提取局部存活粒子的 r, fi 和 z 坐标
        r_polar = self.LocalBunch[survived_mask, self.BunchAttribute.Attribute['r']]  # 粒子的径向坐标
        fi_polar = self.LocalBunch[survived_mask, self.BunchAttribute.Attribute['fi']]  # 粒子的方位角坐标
        z_polar = self.LocalBunch[survived_mask, self.BunchAttribute.Attribute['z']]  # 粒子的纵向坐标（z 方向）

        # 将柱坐标 (r, z, phi) 转换为 Mapped Flat Coordinate System 中的直角坐标 (x, z, y)，
        x_mapped = r_polar
        z_mapped = z_polar
        fi_mapped = fi_polar

        fi_mod_360 = np.mod(fi_mapped, 2*np.pi)
        fi_mod_180 =  fi_mod_360.copy()
        fi_mod_180[fi_mod_180 > np.pi] -= 2 * np.pi

        # 计算 x, y 和 z 坐标的最小值和最大值
        self.rmin_Local_flat = np.nanmin(x_mapped) if len(x_mapped) > 1 else None
        self.rmax_Local_flat = np.nanmax(x_mapped) if len(x_mapped) > 1 else None
        self.zmin_Local_flat = np.nanmin(z_mapped) if len(z_mapped) > 1 else None
        self.zmax_Local_flat = np.nanmax(z_mapped) if len(z_mapped) > 1 else None

        fmin_Local_flat_360 = np.nanmin(fi_mod_360) if len(fi_mod_360) > 1 else None
        fmax_Local_flat_360 = np.nanmax(fi_mod_360) if len(fi_mod_360) > 1 else None
        fmin_Local_flat_180 = np.nanmin(fi_mod_180) if len(fi_mod_180) > 1 else None
        fmax_Local_flat_180 = np.nanmax(fi_mod_180) if len(fi_mod_180) > 1 else None
        if fmin_Local_flat_360 is not None:
            if (fmax_Local_flat_180 - fmin_Local_flat_180) < (fmax_Local_flat_360 - fmin_Local_flat_360) :
                self.fmin_Local_flat = fmin_Local_flat_180
                self.fmax_Local_flat = fmax_Local_flat_180
            else:
                self.fmin_Local_flat = fmin_Local_flat_360
                self.fmax_Local_flat = fmax_Local_flat_360
        else:
            self.fmin_Local_flat = fmin_Local_flat_360
            self.fmax_Local_flat = fmax_Local_flat_360

        pass

        # # 如果 self.fmin_Local_flat > self.fmax_Local_flat，则将 fmax_Local_flat 加 2π
        # if self.fmin_Local_flat is not None:
        #     self.fmin_Local_flat = np.mod(self.fmin_Local_flat, np.pi * 2)
        #     self.fmax_Local_flat = np.mod(self.fmax_Local_flat, np.pi * 2)
        #     if self.fmin_Local_flat > self.fmax_Local_flat:
        #         self.fmax_Local_flat += 2 * np.pi

        # print('rmin_Local_flat=', self.rmin_Local_flat, 'rmax_Local_flat=', self.rmax_Local_flat)

    # @profile
    def Get_SC_Grid_para_global(self):
        """
        使用MPI获取全局范围内的最小和最大x, y, z坐标。
        """
        # 先调用 Get_SC_Grid_para_Local 更新计算局部参数
        self.Get_SC_Grid_para_Local()

        comm = MPI.COMM_WORLD

        # 检查是否为有效值，避免NaN或None传递
        xmin_Local = self.xmin_Local if self.xmin_Local is not None else np.inf
        xmax_Local = self.xmax_Local if self.xmax_Local is not None else -np.inf
        ymin_Local = self.ymin_Local if self.ymin_Local is not None else np.inf
        ymax_Local = self.ymax_Local if self.ymax_Local is not None else -np.inf
        zmin_Local = self.zmin_Local if self.zmin_Local is not None else np.inf
        zmax_Local = self.zmax_Local if self.zmax_Local is not None else -np.inf


        # 通过allreduce获取全局的最小值和最大值
        self.xmin_Global = comm.allreduce(xmin_Local, op=MPI.MIN)
        self.xmax_Global = comm.allreduce(xmax_Local, op=MPI.MAX)
        self.ymin_Global = comm.allreduce(ymin_Local, op=MPI.MIN)
        self.ymax_Global = comm.allreduce(ymax_Local, op=MPI.MAX)
        self.zmin_Global = comm.allreduce(zmin_Local, op=MPI.MIN)
        self.zmax_Global = comm.allreduce(zmax_Local, op=MPI.MAX)

        x_range_max_min = self.xmax_Global - self.xmin_Global
        y_range_max_min = self.ymax_Global - self.ymin_Global
        z_range_max_min = self.zmax_Global - self.zmin_Global
        self.xmin_Global -= x_range_max_min*0.5
        self.xmax_Global += x_range_max_min*0.5
        self.ymin_Global -= y_range_max_min*0.5
        self.ymax_Global += y_range_max_min*0.5
        self.zmin_Global -= z_range_max_min * 1.5
        self.zmax_Global += z_range_max_min * 1.5

        return self.xmin_Global, self.xmax_Global, self.ymin_Global, self.ymax_Global, self.zmin_Global, self.zmax_Global


    def Get_SC_Grid_para_global_FlatCoordinate(self):
        """
        使用MPI获取全局范围内的最小和最大x, y, z坐标。
        """
        # 先调用 Get_SC_Grid_para_Local_FlatCoordinate 更新计算局部参数
        self.Get_SC_Grid_para_Local_FlatCoordinate()

        comm = MPI.COMM_WORLD

        # 检查是否为有效值，避免NaN或None传递
        rmin_Local_flat = self.rmin_Local_flat if self.rmin_Local_flat is not None else np.inf
        rmax_Local_flat = self.rmax_Local_flat if self.rmax_Local_flat is not None else -np.inf
        zmin_Local_flat = self.zmin_Local_flat if self.zmin_Local_flat is not None else np.inf
        zmax_Local_flat = self.zmax_Local_flat if self.zmax_Local_flat is not None else -np.inf
        fmin_Local_flat = self.fmin_Local_flat if self.fmin_Local_flat is not None else np.inf
        fmax_Local_flat = self.fmax_Local_flat if self.fmax_Local_flat is not None else -np.inf

        # print("rmin_g=", rmin_Local_flat, "rmax_g=", rmax_Local_flat)

        # 通过allreduce获取全局的最小值和最大值
        self.rmin_Global_flat = comm.allreduce(rmin_Local_flat, op=MPI.MIN)
        self.rmax_Global_flat = comm.allreduce(rmax_Local_flat, op=MPI.MAX)
        self.zmin_Global_flat = comm.allreduce(zmin_Local_flat, op=MPI.MIN)
        self.zmax_Global_flat = comm.allreduce(zmax_Local_flat, op=MPI.MAX)
        self.fmin_Global_flat = comm.allreduce(fmin_Local_flat, op=MPI.MIN)
        self.fmax_Global_flat = comm.allreduce(fmax_Local_flat, op=MPI.MAX)

        # print("rmin_g2=", self.rmin_Global_flat, "rmax_g2=", self.rmax_Global_flat)
        z_range_max_min = self.zmax_Global_flat - self.zmin_Global_flat
        r_range_max_min = self.rmax_Global_flat - self.rmin_Global_flat
        self.rmin_Global_flat -= r_range_max_min*0.5
        self.rmax_Global_flat += r_range_max_min*0.5
        self.zmin_Global_flat -= z_range_max_min*1.5
        self.zmax_Global_flat += z_range_max_min*1.5

        self.rmean = (self.rmin_Global_flat + self.rmax_Global_flat) / 2.0

        self.frmin_Global_flat = self.fmin_Global_flat * self.rmean
        self.frmax_Global_flat = self.fmax_Global_flat * self.rmean

        return (self.rmin_Global_flat, self.rmax_Global_flat, self.fmin_Global_flat,
                self.fmax_Global_flat, self.zmin_Global_flat, self.zmax_Global_flat)

    # @profile
    def DistributeChargeLocal(self, n, m, l):
        """
        根据 global 边界值和网格数量生成 3D 网格，并将 local bunch 中的粒子分配到 3D 网格中。
        :param n: x 方向的网格点数
        :param m: y 方向的网格点数
        :param l: z 方向的网格点数
        :return: local 3D 电荷分布矩阵
        """
        # 生成 3D 网格


        x_grid = np.linspace(self.xmin_Global, self.xmax_Global, n)
        y_grid = np.linspace(self.ymin_Global, self.ymax_Global, m)
        z_grid = np.linspace(self.zmin_Global, self.zmax_Global, l)
        Xmatrix, Ymatrix, Zmatrix = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

        # 初始化电荷分布矩阵
        charge_distribution_local = np.zeros((n, m, l))

        # 获取所有本地粒子的笛卡尔坐标
        coordinates, _, _ = self.GetLocalCoordinates()

        # # 使用 find_two_points_non_uniform_vect_njit 函数来找到粒子在网格中的邻近点
        # idx_array_x, _ = find_two_points_non_uniform_vect_njit(x_grid, coordinates[:, 0])  # x 坐标
        # idx_array_y, _ = find_two_points_non_uniform_vect_njit(y_grid, coordinates[:, 1])  # y 坐标
        # idx_array_z, _ = find_two_points_non_uniform_vect_njit(z_grid, coordinates[:, 2])  # z 坐标

        # 使用 find_two_points_uniform_vectorized 函数来找到粒子在网格中的邻近点
        idx_array_x, _ = find_two_points_uniform_vectorized(x_grid, coordinates[:, 0])  # r 坐标
        idx_array_y, _ = find_two_points_uniform_vectorized(y_grid, coordinates[:, 1])  # f 坐标
        idx_array_z, _ = find_two_points_uniform_vectorized(z_grid, coordinates[:, 2])  # z 坐标

        # 调用 DistributeChargeNjit 来分配电荷到网格点
        charge_scale = self.marcosize * FFAG_GlobalParameters().q
        charge_distribution_local_real = DistributeChargeNjit(idx_array_x, idx_array_y, idx_array_z, x_grid, y_grid, z_grid,
                                                         charge_distribution_local, coordinates, charge_scale)

        # charge_distribution_local_real = charge_distribution_local_marco * self.marcosize * FFAG_GlobalParameters().q
        # charge_distribution_local_real = charge_distribution_local_marco * self.marcosize * FFAG_GlobalParameters().q

        # 保存本地的 3D 电荷分布矩阵
        self.charge_distribution_local = charge_distribution_local_real

        return charge_distribution_local_real, Xmatrix, Ymatrix, Zmatrix, x_grid, y_grid, z_grid


    def DistributeChargeLocal_FlatCoordinate(self, n, m, l):
        """
        根据 global 边界值和网格数量生成 3D 网格，并将 local bunch 中的粒子分配到 3D 网格中。
        :param n: x 方向的网格点数
        :param m: y 方向的网格点数
        :param l: z 方向的网格点数
        :return: local 3D 电荷分布矩阵
        """
        # 生成 3D 网格
        r_grid_flat = np.linspace(self.rmin_Global_flat, self.rmax_Global_flat, n)
        f_grid_flat = np.linspace(self.fmin_Global_flat, self.fmax_Global_flat, m)
        z_grid_flat = np.linspace(self.zmin_Global_flat, self.zmax_Global_flat, l)
        Rmatrix_flat, Fmatrix_flat, Zmatrix_flat = np.meshgrid(r_grid_flat, f_grid_flat, z_grid_flat, indexing='ij')

        # 初始化电荷分布矩阵
        charge_distribution_local_flat = np.zeros((n, m, l))

        # 获取所有本地粒子的笛卡尔坐标
        coordinates_flat, _, _ = self.GetLocalCoordinates_FlatCoordinate()

        # 使用 find_two_points_non_uniform_vect_njit 函数来找到粒子在网格中的邻近点
        idx_array_r, _ = find_two_points_non_uniform_vect_njit(r_grid_flat, coordinates_flat[:, 0])  # r 坐标
        idx_array_f, _ = find_two_points_non_uniform_vect_njit(f_grid_flat, coordinates_flat[:, 1])  # f 坐标
        idx_array_z, _ = find_two_points_non_uniform_vect_njit(z_grid_flat, coordinates_flat[:, 2])  # z 坐标


        # 调用 DistributeChargeNjit 来分配电荷到网格点
        charge_distribution_local_marco_flat = DistributeChargeNjit(idx_array_r, idx_array_f, idx_array_z,
                                                               r_grid_flat, f_grid_flat, z_grid_flat,
                                                               charge_distribution_local_flat, coordinates_flat)

        charge_distribution_local_real_flat = charge_distribution_local_marco_flat * self.marcosize * FFAG_GlobalParameters().q

        # 保存本地的 3D 电荷分布矩阵
        self.charge_distribution_local_flat = charge_distribution_local_real_flat

        return charge_distribution_local_real_flat, Rmatrix_flat, Fmatrix_flat, Zmatrix_flat, r_grid_flat, f_grid_flat, z_grid_flat

    # @profile
    def DistributeChargeGlobal(self, n, m, l):
        """
        汇总所有进程的局部电荷分布，得到全局的 3D 电荷分布矩阵。
        :param n: x 方向的网格点数
        :param m: y 方向的网格点数
        :param l: z 方向的网格点数
        :return: global 3D 电荷分布矩阵
        """
        # 检查边界值是否无效
        if (self.xmin_Global >= self.xmax_Global or
                self.ymin_Global >= self.ymax_Global or
                self.zmin_Global >= self.zmax_Global):

            self.xmin_Global = None
            self.xmax_Global = None
            self.ymin_Global = None
            self.ymax_Global = None
            self.zmin_Global = None
            self.zmax_Global = None

            return None, None, None, None, None, None, None

        # 初始化 MPI 通信
        comm = MPI.COMM_WORLD

        # 调用 DistributeChargeLocal 获取本地的 3D 电荷分布矩阵
        local_charge_distribution, Xmatrix, Ymatrix, Zmatrix, x_grid, y_grid, z_grid = (
            self.DistributeChargeLocal(n, m, l))

        # 初始化全局 3D 电荷分布矩阵
        global_charge_distribution = np.zeros((n, m, l))

        # 使用 MPI.allreduce 汇总所有进程的电荷分布矩阵
        comm.Allreduce(local_charge_distribution, global_charge_distribution, op=MPI.SUM)

        # 保存全局的 3D 电荷分布矩阵
        self.charge_distribution_global = global_charge_distribution
        self.Xmatrix = Xmatrix
        self.Ymatrix = Ymatrix
        self.Zmatrix = Zmatrix
        self.Xgrid = x_grid
        self.Ygrid = y_grid
        self.Zgrid = z_grid

        return global_charge_distribution, Xmatrix, Ymatrix, Zmatrix, x_grid, y_grid, z_grid


    def DistributeChargeGlobal_FlatCoordinate(self, n, m, l):
        """
        汇总所有进程的局部电荷分布，得到全局的 3D 电荷分布矩阵。
        :param n: x 方向的网格点数
        :param m: y 方向的网格点数
        :param l: z 方向的网格点数
        :return: global 3D 电荷分布矩阵
        """
        # 检查边界值是否无效
        if (self.rmin_Global_flat >= self.rmax_Global_flat or
                self.zmin_Global_flat >= self.zmax_Global_flat or
                self.fmin_Global_flat >= self.fmax_Global_flat):

            self.rmin_Global_flat = None
            self.rmax_Global_flat = None
            self.zmin_Global_flat = None
            self.zmax_Global_flat = None
            self.fmin_Global_flat = None
            self.fmax_Global_flat = None

            return None, None, None, None, None, None, None

        # 初始化 MPI 通信
        comm = MPI.COMM_WORLD

        # 调用 DistributeChargeLocal 获取本地的 3D 电荷分布矩阵
        (local_charge_distribution_flat, Rmatrix_flat, Fmatrix_flat, Zmatrix_flat,
         r_grid_flat, f_grid_flat, z_grid_flat) = (
            self.DistributeChargeLocal_FlatCoordinate(n, m, l))

        # 初始化全局 3D 电荷分布矩阵
        global_charge_distribution_flat = np.zeros((n, m, l))

        # 使用 MPI.allreduce 汇总所有进程的电荷分布矩阵
        comm.Allreduce(local_charge_distribution_flat, global_charge_distribution_flat, op=MPI.SUM)

        # 保存全局的 3D 电荷分布矩阵
        self.charge_distribution_global_flat = global_charge_distribution_flat
        self.Rmatrix_flat = Rmatrix_flat
        self.Fmatrix_flat = Fmatrix_flat
        self.Zmatrix_flat = Zmatrix_flat
        self.Rgrid_flat = r_grid_flat
        self.Fgrid_flat = f_grid_flat
        self.Zgrid_flat = z_grid_flat

        return global_charge_distribution_flat, Rmatrix_flat, Fmatrix_flat, Zmatrix_flat, r_grid_flat, f_grid_flat, z_grid_flat


    def Update_SC_Efield_Local(self, Er_Local, Efi_Local, Ez_Local, Local_ID):
        """
        更新LocalBunch中的空间电荷效应self induced电场分量，包括径向电场Er，方位角电场Efi和纵向电场Ez。
        :param Er_Local: 空间电荷效应径向电场
        :param Efi_Local: 空间电荷效应方位角电场
        :param Ez_Local: 空间电荷效应纵向电场
        """
        # 获取电场对应的列索引
        er_idx = self.BunchAttribute.Attribute['Esc_r']
        ez_idx = self.BunchAttribute.Attribute['Esc_z']
        efi_idx = self.BunchAttribute.Attribute['Esc_fi']

        # 检查输入数组的长度是否匹配LocalBunch的粒子数
        if len(Er_Local) != len(Local_ID) or len(Efi_Local) != len(Local_ID) or len(Ez_Local) != len(
                Local_ID):
            raise ValueError("Input field arrays must have the same length as LocalBunch.")

        # 更新电场到LocalBunch的对应列
        self.LocalBunch[Local_ID, er_idx] = Er_Local
        self.LocalBunch[Local_ID, ez_idx] = Ez_Local
        self.LocalBunch[Local_ID, efi_idx] = Efi_Local


    # @profile
    def Build_2_5D_SliceGrid(self, fft_fwd_1, fft_inv_1):
        """
        基于 (x, y, z) 进行 φ slice 划分，并在 (x, z) 平面上构建 2D 电荷网格。
        使用 DistributeCharge2D_Njit 进行加权电荷分配。支持多线程并行合并
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        segments = self.f_grid_size
        nx = self.x_grid_size
        nz = self.z_grid_size

        SEO_fi_axis = self.SEO_fi_axis
        SEO_Ek_axis = self.SEO_Ek_axis
        SEO_r_matrix = self.SEO_r_matrix
        SEO_perimeter = self.SEO_perimeter

        # 1. 获取当前线程存活粒子的极坐标, coords_mean_gamma是预生成的全为1的数组
        (coords_r, coords_fi, coords_z, coords_gamma, coords_mean_gamma, coords_fi_mod, coords_survive_idx, survive_num, sum_gamma_local) = GetLocalCoordinates_2p5(self.LocalBunch,
                                                                                          self.Step_SurviveFlag,
                                                                                          self.Malloc_SC_r,
                                                                                          self.Malloc_SC_fi,
                                                                                          self.Malloc_SC_z,
                                                                                          self.Malloc_SC_gamma,self.Malloc_SC_mean_gamma_interp,
                                                                                          self.Malloc_SC_fiMod, self.Malloc_SC_surviveIDX, self.speed_c)

        # 2. 当前线程的 φ 覆盖范围
        arc_start_local, arc_end_local, _ = divide_arc2(coords_fi_mod)
        # print(f"arc_start_local: {arc_start_local}, arc_end_local: {arc_end_local}, rank: {rank}")
        # 3. 所有线程收集 arcs
        local_arc = np.array([arc_start_local, arc_end_local])
        all_arcs = np.zeros((size, 2))
        comm.Allgather(local_arc, all_arcs)
        arcs_list = [tuple(row) for row in all_arcs]
        # 4. 全局 φ 范围合并
        fi_s_global, fi_e_global = merge_multiple_arcs(arcs_list)
        if fi_e_global <= fi_s_global:
            fi_e_global += 2 * np.pi
        # 5. φ 方向均匀划分
        slice_fi_step = (fi_e_global - fi_s_global) / (segments-1)
        fi_grid = np.linspace(fi_s_global, fi_e_global, segments)
        # print(f"fi_s_global: {fi_s_global}, fi_e_global: {fi_e_global}, arclist={arcs_list}")

        # MPI 汇总
        sum_gamma_global = comm.allreduce(sum_gamma_local, op=MPI.SUM)
        count_global = comm.allreduce(survive_num, op=MPI.SUM)

        if count_global > 0:
            mean_gamma_global = sum_gamma_global / count_global - 1.0
        else:
            mean_gamma_global = 0.0

        # 9. 根据平均gamma对应的SEO,
        coords_r0, _ = Bilinear_interp_2D_vect_uniform(SEO_Ek_axis[0], SEO_Ek_axis[1] - SEO_Ek_axis[0],
                                                       len(SEO_Ek_axis),
                                                       SEO_fi_axis[0], SEO_fi_axis[1] - SEO_fi_axis[0],
                                                       len(SEO_fi_axis),
                                                       SEO_r_matrix,
                                                       # coords_mean_gamma * mean_gamma_global * self.E0_rest_MeV,
                                                       (coords_gamma-1) * self.E0_rest_MeV,
                                                       coords_fi_mod,
                                                       self.Malloc_SC_r_SEO, self.Malloc_SC_interp_flag)

        # print("coords_r0=", [coords_r0[0], coords_r0[1000], coords_r0[2000]], "Ek=", mean_gamma_global * self.E0_rest_MeV )

        # 将r转换为x, 然后计算xz的统计量，再计算xz平均值和sigma, 再将fi转换为s
        (coords_x, r_slices, z_slices, count_slices, slice_indices,
         sum_x_local, sum_z_local, sum_x2_local, sum_z2_local) = (
            ComputeSliceBoundsAndCoords_merged(coords_r, coords_z, coords_fi_mod, coords_r0,
                                               self.Malloc_SC_x,
                                               slice_fi_step, fi_s_global, segments,
                                               self.Malloc_SC_slice_r2d, self.Malloc_SC_slice_z2d,
                                               self.Malloc_SC_slice_indices))

        # 6. 粒子分配到 φ slice（考虑跨越 2π）
        # 7. 每 slice 分别计算局部 rmin/rmax/zmin/zmax

        # MPI gather 缓存，改为 numpy 数组
        sum_x_global = np.zeros(1, dtype=np.float64)
        sum_z_global = np.zeros(1, dtype=np.float64)
        sum_x2_global = np.zeros(1, dtype=np.float64)
        sum_z2_global = np.zeros(1, dtype=np.float64)
        count_slices_global = np.zeros_like(count_slices)

        # Allreduce
        comm.Allreduce(np.array([sum_x_local,]), sum_x_global, op=MPI.SUM)
        comm.Allreduce(np.array([sum_z_local,]), sum_z_global, op=MPI.SUM)
        comm.Allreduce(np.array([sum_x2_local,]), sum_x2_global, op=MPI.SUM)
        comm.Allreduce(np.array([sum_z2_local,]), sum_z2_global, op=MPI.SUM)
        comm.Allreduce(count_slices, count_slices_global, op=MPI.SUM)

        if count_global > 0:
            global_mean_x = np.sum(sum_x_global) / count_global
            global_mean_z = np.sum(sum_z_global) / count_global

            global_var_x = np.sum(sum_x2_global) / count_global - global_mean_x ** 2
            global_var_z = np.sum(sum_z2_global) / count_global - global_mean_z ** 2

            global_sigma_x = np.sqrt(max(global_var_x, 0.0))
            global_sigma_z = np.sqrt(max(global_var_z, 0.0))
        else:
            global_mean_x = global_mean_z = global_sigma_x = global_sigma_z = 0.0

        # 获取束长
        SEO_perimeter_MeanGamma = np.interp(mean_gamma_global * self.E0_rest_MeV, SEO_Ek_axis, SEO_perimeter)
        BunchLength = SEO_perimeter_MeanGamma  * (fi_e_global - fi_s_global)/ np.pi / 2.0
        s_grid = fi_grid/np.pi/2.0*SEO_perimeter_MeanGamma
        # coords_x, phi_times_R, BunchLength = coordinate_convert_SC(coords_r, mean_r_global, fi_s_global, fi_e_global, segments, slice_indices)
        scale = self.sigma_ratio
        # scale=20
        # print(f"BunchLength: {BunchLength}")

        # 每个slice、local粒子的边界
        global_xmin = global_mean_x - scale * global_sigma_x
        global_xmax = global_mean_x + scale * global_sigma_x
        global_zmin = global_mean_z - scale * global_sigma_z
        global_zmax = global_mean_z + scale * global_sigma_z
        # # 所有slices、local粒子的边界
        # local_xmin = np.min(x_min_slices)
        # local_xmax = np.max(x_max_slices)
        # local_zmin = np.min(z_min_slices)
        # local_zmax = np.max(z_max_slices)
        # # 所有slices、global粒子的边界
        # global_xmin = comm.allreduce(local_xmin, op=MPI.MIN)
        # global_xmax = comm.allreduce(local_xmax, op=MPI.MAX)
        # global_zmin = comm.allreduce(local_zmin, op=MPI.MIN)
        # global_zmax = comm.allreduce(local_zmax, op=MPI.MAX)

        # 9. 生成一个x-z的mesh网格，然后分配当前线程的电荷到网格点上，形成二维xz电荷分布和一维纵向分布rho_f
        q_part = self.marcosize * self.q_charge
        rho_xz = DistributeCharge2D_Njit(coords_x, coords_z, q_part,
                                                        BunchLength, global_xmin, global_xmax,
                                                        global_zmin, global_zmax, nx, nz)

        rho_f = count_slices_global * q_part

        # 10. 各进程进行通信，汇总加和各子束团的slice_charge_maps, 得到全局的rho_xz, rho_f,
        # 汇总各 slice 电荷密度网格
        comm.Allreduce(MPI.IN_PLACE, rho_xz, op=MPI.SUM)
        comm.Allreduce(MPI.IN_PLACE, rho_f, op=MPI.SUM)

        # ========================
        # # 11. 计算每个 φ‑slice 的二维电势分布 φ[nz, nx]
        # #    开放边界：零填充 2× 网格 → 频域卷积 → 裁剪
        # 用 FFTW 代替 scipy.fft
        # ========================

        dx = (global_xmax - global_xmin) / nx
        dz = (global_zmax - global_zmin) / nz
        if dx == 0.0 or dz == 0.0 or BunchLength <= 0.0:
            # slice_phi_maps2.append(np.zeros((nz, nx), dtype=np.float32))
            phi_pad = np.zeros((nz, nx), dtype=np.float32)
        else:
            # 1) 计算频域卷积核 (numpy float32, nz*nx)
            fft_kernel = make_fft_kernel_open_numba(nx, nz, dx, dz)
            # 2) FFT plan
            buf = fft_fwd_1.input_array  # complex64, shape=(nz,nx)
            buf.real[:] = rho_xz.astype(np.float32)  # 实部填ρ, 虚部0
            buf.imag[:] = 0.0
            # 3) 前向 FFTW
            fft_fwd_1()  # b_buf = FFT(a_buf)
            # 4) 频域乘核
            out = fft_fwd_1.output_array
            out *= fft_kernel
            # 5) 反向 FFTW
            fft_inv_1()  # a_buf = IFFT(b_buf)
            # 6) 裁剪并存回列表
            phi_pad = fft_inv_1.output_array.real

        Ef_map = total_field_1d_numba(s_grid, rho_f)

        # ================================================
        # 12/13: 从 slice_phi_maps2 计算当前线程所有粒子的 Ex, Ez
        # ================================================
        # 初始化输出：与 coords_r 一一对应
        # print(f"gamma-1={mean_gamma_global}")
        Ex_map, Ez_map = compute_field_from_potential_numba(phi_pad, dx, dz, BunchLength, self.Malloc_SC_Exmap, self.Malloc_SC_Ezmap, mean_gamma_global)

        interpolate_fields_to_particles_numba(
            Ex_map, Ez_map, Ef_map,
            coords_x, coords_z, slice_indices,
            coords_survive_idx,
            global_xmin, global_zmin, dx, dz,BunchLength,
            self.Malloc_SC_Ex_p, self.Malloc_SC_Ez_p, self.Malloc_SC_Ef_p,
            self.LocalBunch
        )

        return rho_xz, phi_pad, Ex_map, Ez_map, global_xmin, global_xmax, global_zmin, global_zmax, BunchLength, Ef_map, s_grid, rho_f, coords_r, coords_fi, coords_x, coords_r0, coords_z, fi_grid

    # @profile
    def Build_2_5D_SliceGrid_sbs(self, fft_fwd_1, fft_inv_1):
        """
        基于 (x, y, z) 进行 φ slice 划分，并在 (x, z) 平面上构建 2D 电荷网格, slice by slice版本。
        使用 DistributeCharge2D_Njit 进行加权电荷分配。支持多线程并行合并
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        ns = self.f_grid_size
        nx = self.x_grid_size
        nz = self.z_grid_size

        SEO_fi_axis = self.SEO_fi_axis
        SEO_Ek_axis = self.SEO_Ek_axis
        SEO_r_matrix = self.SEO_r_matrix
        SEO_pr_matrix = self.SEO_pr_matrix  # pr/pfi
        SEO_perimeter = self.SEO_perimeter

        # 1. 获取当前线程存活粒子的极坐标, coords_mean_gamma是预生成的全为1的数组
        if self.BunchType == "Boris":
            (coords_r, coords_fi, coords_z, coords_gamma, coords_mean_gamma,
             coords_fi_mod, coords_survive_idx, survive_num, sum_gamma_local) = GetLocalCoordinates_2p5(self.LocalBunch,
                                                                                              self.Step_SurviveFlag,
                                                                                              self.Malloc_SC_r,
                                                                                              self.Malloc_SC_fi,
                                                                                              self.Malloc_SC_z,
                                                                                              self.Malloc_SC_gamma,self.Malloc_SC_mean_gamma_interp,
                                                                                              self.Malloc_SC_fiMod, self.Malloc_SC_surviveIDX, self.speed_c)
        elif self.BunchType == "RK4":
            (coords_r, coords_fi, coords_z, coords_gamma, coords_mean_gamma,
             coords_fi_mod, coords_survive_idx, survive_num, sum_gamma_local) = GetLocalCoordinates_2p5_rk(
                self.LocalBunch,
                self.Step_SurviveFlag,
                self.Malloc_SC_r,
                self.Malloc_SC_fi,
                self.Malloc_SC_z,
                self.Malloc_SC_gamma,
                self.Malloc_SC_mean_gamma_interp,
                self.Malloc_SC_fiMod,
                self.Malloc_SC_surviveIDX,
                self.speed_c)
        else:
            raise("BunchType must be Boris or RK4")

        # 2. 当前线程的 φ 覆盖范围
        arc_start_local, arc_end_local, _ = divide_arc2(coords_fi_mod)
        # 3. 所有线程收集 arcs
        local_arc = np.array([arc_start_local, arc_end_local])
        all_arcs = np.zeros((size, 2))
        comm.Allgather(local_arc, all_arcs)
        arcs_list = [tuple(row) for row in all_arcs]
        # 4. 全局 φ 范围合并
        fi_s_global, fi_e_global = merge_multiple_arcs(arcs_list)
        if fi_e_global <= fi_s_global:
            fi_e_global += 2 * np.pi
        # 5. φ 方向均匀划分
        slice_fi_step = (fi_e_global - fi_s_global) / (ns-1)
        fi_grid = np.linspace(fi_s_global, fi_e_global, ns)

        # MPI 汇总
        sum_gamma_global = comm.allreduce(sum_gamma_local, op=MPI.SUM)
        count_global = comm.allreduce(survive_num, op=MPI.SUM)

        if count_global > 0:
            mean_gamma_global = sum_gamma_global / count_global - 1.0
        else:
            mean_gamma_global = 0.0

        # 9. 根据平均gamma对应的SEO,
        coords_r0, _ = Bilinear_interp_2D_vect_uniform(SEO_Ek_axis[0], SEO_Ek_axis[1] - SEO_Ek_axis[0],
                                                       len(SEO_Ek_axis),
                                                       SEO_fi_axis[0], SEO_fi_axis[1] - SEO_fi_axis[0],
                                                       len(SEO_fi_axis),
                                                       SEO_r_matrix,
                                                       # coords_mean_gamma * mean_gamma_global * self.E0_rest_MeV,
                                                       (coords_gamma-1) * self.E0_rest_MeV,
                                                       coords_fi_mod,
                                                       self.Malloc_SC_r_SEO, self.Malloc_SC_interp_flag)

        coords_pr0, _ = Bilinear_interp_2D_vect_uniform(
            SEO_Ek_axis[0], SEO_Ek_axis[1] - SEO_Ek_axis[0], len(SEO_Ek_axis),
            SEO_fi_axis[0], SEO_fi_axis[1] - SEO_fi_axis[0], len(SEO_fi_axis),
            SEO_pr_matrix,
            (coords_gamma - 1.0) * self.E0_rest_MeV,
            coords_fi_mod,
            self.Malloc_SC_pr_SEO,
            self.Malloc_SC_interp_flag)

        # 将r转换为x, 然后计算xz的统计量，再计算xz平均值和sigma, 再将fi转换为s
        (coords_x, r_slices, z_slices, x_slices, count_slices, slice_indices,
         sum_x_local, sum_z_local, sum_x2_local, sum_z2_local) = (
            ComputeSliceBoundsAndCoordsSBS_merged(coords_r, coords_z, coords_fi_mod, coords_r0,
                                               self.Malloc_SC_x,
                                               slice_fi_step, fi_s_global, ns,
                                               self.Malloc_SC_slice_r2d, self.Malloc_SC_slice_z2d,
                                                  self.Malloc_SC_slice_x2d,
                                               self.Malloc_SC_slice_indices))

        # 6. 粒子分配到 φ slice（考虑跨越 2π）
        # 7. 每 slice 分别计算局部 rmin/rmax/zmin/zmax
        # MPI gather 缓存，改为 numpy 数组
        sum_x_global = np.zeros(1, dtype=np.float64)
        sum_z_global = np.zeros(1, dtype=np.float64)
        sum_x2_global = np.zeros(1, dtype=np.float64)
        sum_z2_global = np.zeros(1, dtype=np.float64)
        count_slices_global = np.zeros_like(count_slices)

        # Allreduce
        comm.Allreduce(np.array([sum_x_local,]), sum_x_global, op=MPI.SUM)
        comm.Allreduce(np.array([sum_z_local,]), sum_z_global, op=MPI.SUM)
        comm.Allreduce(np.array([sum_x2_local,]), sum_x2_global, op=MPI.SUM)
        comm.Allreduce(np.array([sum_z2_local,]), sum_z2_global, op=MPI.SUM)
        comm.Allreduce(count_slices, count_slices_global, op=MPI.SUM)

        if count_global > 1:
            global_mean_x = np.sum(sum_x_global) / count_global
            global_mean_z = np.sum(sum_z_global) / count_global

            global_var_x = np.sum(sum_x2_global) / count_global - global_mean_x ** 2
            global_var_z = np.sum(sum_z2_global) / count_global - global_mean_z ** 2

            global_sigma_x = np.sqrt(max(global_var_x, 0.0))
            global_sigma_z = np.sqrt(max(global_var_z, 0.0))
        else:
            global_mean_x = global_mean_z = global_sigma_x = global_sigma_z = 0.0

        # 获取束长
        SEO_perimeter_MeanGamma = np.interp(mean_gamma_global * self.E0_rest_MeV, SEO_Ek_axis, SEO_perimeter)
        BunchLength = SEO_perimeter_MeanGamma  * (fi_e_global - fi_s_global)/ np.pi / 2.0
        s_grid = fi_grid/np.pi/2.0*SEO_perimeter_MeanGamma
        scale = self.sigma_ratio

        # 每个slice、local粒子的边界
        global_xmin = global_mean_x - scale * global_sigma_x
        global_xmax = global_mean_x + scale * global_sigma_x
        global_zmin = global_mean_z - scale * global_sigma_z
        global_zmax = global_mean_z + scale * global_sigma_z

        # 9. 生成一个x-z的mesh网格，然后分配当前线程的电荷到网格点上，形成二维xz电荷分布和一维纵向分布rho_f
        q_part = self.marcosize * self.q_charge

        rho_xz_slices_ = DistributeChargeSBS_Njit(z_slices, x_slices, count_slices, q_part,
                                                        BunchLength, global_xmin, global_xmax,
                                                        global_zmin, global_zmax, nx, nz, ns)

        rho_f = count_slices_global * q_part

        # ===== 11. per-slice 2D 卷积（owner-compute，最后 Allreduce 立方）=====
        dx = (global_xmax - global_xmin) / nx
        dz = (global_zmax - global_zmin) / nz
        if dx <= 0.0 or dz <= 0.0:
            # 无场：直接清零并返回/继续
            self.Malloc_SC_Ex_p[:] = 0.0
            self.Malloc_SC_Ez_p[:] = 0.0

            Ef_map = np.zeros((ns,))
            Ex_cube = np.zeros((ns, nz, nx), dtype=np.float32)  # 本进程持有的立方 非owner的 slice 行保持0
            Ez_cube = np.zeros((ns, nz, nx), dtype=np.float32)
            phi_cube = np.zeros((ns, nz, nx), dtype=np.float32)
        else:
            # 1) 频域核 / 纵向场 / 3D 立方, np.float32
            fft_kernel = make_fft_kernel_open_numba(nx, nz, dx, dz)
            # rho_f 建议用全局值（如前面未Allreduce，这里补一次）
            rho_f_global = rho_f.copy()
            # comm.Allreduce(MPI.IN_PLACE, rho_f_global, op=MPI.SUM)
            comm.Allreduce(MPI.IN_PLACE, rho_xz_slices_, op=MPI.SUM)

            Ef_map = total_field_1d_numba(s_grid, rho_f_global)
            Ex_cube = np.zeros((ns, nz, nx), dtype=np.float32)  # 本进程持有的立方 非owner的 slice 行保持0
            Ez_cube = np.zeros((ns, nz, nx), dtype=np.float32)
            phi_cube = np.zeros((ns, nz, nx), dtype=np.float32)

            for k in range(ns):
                buf = fft_fwd_1.input_array  # complex64, (nz, nx)
                buf.real[:] = rho_xz_slices_[k]
                buf.imag[:] = 0.0
                fft_fwd_1()
                out = fft_fwd_1.output_array
                out *= fft_kernel
                fft_inv_1()
                phi_k = fft_inv_1.output_array.real  # float32
                compute_field_from_potential_numba_SBS(
                    phi_k, k, dx, dz, BunchLength,
                    Ex_cube, Ez_cube, mean_gamma_global)

                phi_cube[k] = phi_k

            # 5) 统一把场插值回本地粒子, 仅处理本地有粒子的片
            interpolate_fields_to_particles_SBS_numba(
                Ex_cube, Ez_cube, Ef_map,
                coords_x, coords_z, coords_pr0, slice_indices, coords_survive_idx,
                global_xmin, global_zmin, dx, dz, BunchLength,
                self.Malloc_SC_Ex_p, self.Malloc_SC_Ez_p, self.Malloc_SC_Ef_p, self.LocalBunch)

        return rho_xz_slices_, phi_cube, Ex_cube, Ez_cube, global_xmin, global_xmax, global_zmin, global_zmax, BunchLength, Ef_map, s_grid, rho_f, coords_r, coords_fi, coords_x, coords_r0, coords_z, fi_grid


    def Build_2_5D_SliceGrid_sbs_benchmark(self, fft_fwd_1, fft_inv_1, USE_SEO=True, SEO_USE_R=10.0):
        """
        #兼容无SEO情况, 主要用于benchmark, 跟踪未使用#
        基于 (x, y, z) 进行 φ slice 划分，并在 (x, z) 平面上构建 2D 电荷网格, slice by slice版本。
        使用 DistributeCharge2D_Njit 进行加权电荷分配。支持多线程并行合并
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        ns = self.f_grid_size
        nx = self.x_grid_size
        nz = self.z_grid_size

        SEO_fi_axis = self.SEO_fi_axis
        SEO_Ek_axis = self.SEO_Ek_axis
        SEO_r_matrix = self.SEO_r_matrix
        SEO_perimeter = self.SEO_perimeter

        # 1. 获取当前线程存活粒子的极坐标, coords_mean_gamma是预生成的全为1的数组
        (coords_r, coords_fi, coords_z, coords_gamma, coords_mean_gamma, coords_fi_mod, coords_survive_idx, survive_num, sum_gamma_local) = GetLocalCoordinates_2p5(self.LocalBunch,
                                                                                          self.Step_SurviveFlag,
                                                                                          self.Malloc_SC_r,
                                                                                          self.Malloc_SC_fi,
                                                                                          self.Malloc_SC_z,
                                                                                          self.Malloc_SC_gamma,self.Malloc_SC_mean_gamma_interp,
                                                                                          self.Malloc_SC_fiMod, self.Malloc_SC_surviveIDX, self.speed_c)

        # 2. 当前线程的 φ 覆盖范围
        arc_start_local, arc_end_local, _ = divide_arc2(coords_fi_mod)
        # 3. 所有线程收集 arcs
        local_arc = np.array([arc_start_local, arc_end_local])
        all_arcs = np.zeros((size, 2))
        comm.Allgather(local_arc, all_arcs)
        arcs_list = [tuple(row) for row in all_arcs]
        # 4. 全局 φ 范围合并
        fi_s_global, fi_e_global = merge_multiple_arcs(arcs_list)
        if fi_e_global <= fi_s_global:
            fi_e_global += 2 * np.pi
        # 5. φ 方向均匀划分
        slice_fi_step = (fi_e_global - fi_s_global) / (ns-1)
        fi_grid = np.linspace(fi_s_global, fi_e_global, ns)

        # MPI 汇总
        sum_gamma_global = comm.allreduce(sum_gamma_local, op=MPI.SUM)
        count_global = comm.allreduce(survive_num, op=MPI.SUM)

        if count_global > 0:
            mean_gamma_global = sum_gamma_global / count_global - 1.0
        else:
            mean_gamma_global = 0.0

        # 9. 根据平均gamma对应的SEO,
        if USE_SEO:
            coords_r0, _ = Bilinear_interp_2D_vect_uniform(SEO_Ek_axis[0], SEO_Ek_axis[1] - SEO_Ek_axis[0],
                                                           len(SEO_Ek_axis),
                                                           SEO_fi_axis[0], SEO_fi_axis[1] - SEO_fi_axis[0],
                                                           len(SEO_fi_axis),
                                                           SEO_r_matrix,
                                                           # coords_mean_gamma * mean_gamma_global * self.E0_rest_MeV,
                                                           (coords_gamma-1) * self.E0_rest_MeV,
                                                           coords_fi_mod,
                                                           self.Malloc_SC_r_SEO, self.Malloc_SC_interp_flag)
        else:
            coords_r0 = np.ones_like(coords_fi_mod)*SEO_USE_R
        # 将r转换为x, 然后计算xz的统计量，再计算xz平均值和sigma,
        pass
        (coords_x, r_slices, z_slices, x_slices, count_slices, slice_indices,
         sum_x_local, sum_z_local, sum_x2_local, sum_z2_local) = (
            ComputeSliceBoundsAndCoordsSBS_merged(coords_r, coords_z, coords_fi_mod, coords_r0,
                                               self.Malloc_SC_x,
                                               slice_fi_step, fi_s_global, ns,
                                               self.Malloc_SC_slice_r2d, self.Malloc_SC_slice_z2d,
                                                  self.Malloc_SC_slice_x2d,
                                               self.Malloc_SC_slice_indices))

        # 6. 粒子分配到 φ slice（考虑跨越 2π）
        # 7. 每 slice 分别计算局部 rmin/rmax/zmin/zmax
        # MPI gather 缓存，改为 numpy 数组
        sum_x_global = np.zeros(1, dtype=np.float64)
        sum_z_global = np.zeros(1, dtype=np.float64)
        sum_x2_global = np.zeros(1, dtype=np.float64)
        sum_z2_global = np.zeros(1, dtype=np.float64)
        count_slices_global = np.zeros_like(count_slices)

        # Allreduce
        comm.Allreduce(np.array([sum_x_local,]), sum_x_global, op=MPI.SUM)
        comm.Allreduce(np.array([sum_z_local,]), sum_z_global, op=MPI.SUM)
        comm.Allreduce(np.array([sum_x2_local,]), sum_x2_global, op=MPI.SUM)
        comm.Allreduce(np.array([sum_z2_local,]), sum_z2_global, op=MPI.SUM)
        comm.Allreduce(count_slices, count_slices_global, op=MPI.SUM)

        if count_global > 0:
            global_mean_x = np.sum(sum_x_global) / count_global
            global_mean_z = np.sum(sum_z_global) / count_global

            global_var_x = np.sum(sum_x2_global) / count_global - global_mean_x ** 2
            global_var_z = np.sum(sum_z2_global) / count_global - global_mean_z ** 2

            global_sigma_x = np.sqrt(max(global_var_x, 0.0))
            global_sigma_z = np.sqrt(max(global_var_z, 0.0))
        else:
            global_mean_x = global_mean_z = global_sigma_x = global_sigma_z = 0.0

        # 获取束长
        if USE_SEO:
            SEO_perimeter_MeanGamma = np.interp(mean_gamma_global * self.E0_rest_MeV, SEO_Ek_axis, SEO_perimeter)
        else:
            SEO_perimeter_MeanGamma = np.pi*2*SEO_USE_R
        BunchLength = SEO_perimeter_MeanGamma  * (fi_e_global - fi_s_global)/ np.pi / 2.0
        s_grid = fi_grid/np.pi/2.0*SEO_perimeter_MeanGamma
        scale = self.sigma_ratio

        # 每个slice、local粒子的边界
        global_xmin = global_mean_x - scale * global_sigma_x
        global_xmax = global_mean_x + scale * global_sigma_x
        global_zmin = global_mean_z - scale * global_sigma_z
        global_zmax = global_mean_z + scale * global_sigma_z

        # 9. 生成一个x-z的mesh网格，然后分配当前线程的电荷到网格点上，形成二维xz电荷分布和一维纵向分布rho_f
        q_part = self.marcosize * self.q_charge

        rho_xz_slices_ = DistributeChargeSBS_Njit(z_slices, x_slices, count_slices, q_part,
                                                        BunchLength, global_xmin, global_xmax,
                                                        global_zmin, global_zmax, nx, nz, ns)

        rho_f = count_slices_global * q_part

        # ===== 11. per-slice 2D 卷积（owner-compute，最后 Allreduce 立方）=====
        dx = (global_xmax - global_xmin) / nx
        dz = (global_zmax - global_zmin) / nz
        if dx <= 0.0 or dz <= 0.0:
            # 无场：直接清零并返回/继续
            self.Malloc_SC_Ex_p[:] = 0.0
            self.Malloc_SC_Ez_p[:] = 0.0

            Ef_map = np.zeros((ns,))
            Ex_cube = np.zeros((ns, nz, nx), dtype=np.float32)  # 本进程持有的立方 非owner的 slice 行保持0
            Ez_cube = np.zeros((ns, nz, nx), dtype=np.float32)
            phi_cube = np.zeros((ns, nz, nx), dtype=np.float32)
        else:
            # 1) 频域核 / 纵向场 / 3D 立方, np.float32
            fft_kernel = make_fft_kernel_open_numba(nx, nz, dx, dz)
            # rho_f 建议用全局值（如前面未Allreduce，这里补一次）
            rho_f_global = rho_f.copy()
            comm.Allreduce(MPI.IN_PLACE, rho_f_global, op=MPI.SUM)
            comm.Allreduce(MPI.IN_PLACE, rho_xz_slices_, op=MPI.SUM)

            Ef_map = total_field_1d_numba(s_grid, rho_f_global)
            Ex_cube = np.zeros((ns, nz, nx), dtype=np.float32)  # 本进程持有的立方 非owner的 slice 行保持0
            Ez_cube = np.zeros((ns, nz, nx), dtype=np.float32)
            phi_cube = np.zeros((ns, nz, nx), dtype=np.float32)

            for k in range(ns):
                buf = fft_fwd_1.input_array  # complex64, (nz, nx)
                buf.real[:] = rho_xz_slices_[k]
                buf.imag[:] = 0.0
                fft_fwd_1()
                out = fft_fwd_1.output_array
                out *= fft_kernel
                fft_inv_1()
                phi_k = fft_inv_1.output_array.real  # float32
                compute_field_from_potential_numba_SBS(
                    phi_k, k, dx, dz, BunchLength,
                    Ex_cube, Ez_cube, mean_gamma_global)

                phi_cube[k] = phi_k

            # 5) 统一把场插值回本地粒子, 仅处理本地有粒子的片
            interpolate_fields_to_particles_SBS_numba(
                Ex_cube, Ez_cube, Ef_map,
                coords_x, coords_z, slice_indices, coords_survive_idx,
                global_xmin, global_zmin, dx, dz, BunchLength,
                self.Malloc_SC_Ex_p, self.Malloc_SC_Ez_p, self.Malloc_SC_Ef_p, self.LocalBunch)

        return rho_xz_slices_, phi_cube, Ex_cube, Ez_cube, global_xmin, global_xmax, global_zmin, global_zmax, BunchLength, Ef_map, s_grid, rho_f, coords_r, coords_fi, coords_x, coords_r0, coords_z, fi_grid


    # @profile
    def Build_2_5D_SliceGrid_sbs_mpi(self, fft_fwd_1, fft_inv_1):
        """
        基于 (x, y, z) 进行 φ slice 划分，并在 (x, z) 平面上构建 2D 电荷网格, slice by slice版本。
        使用 DistributeCharge2D_Njit 进行加权电荷分配。支持多线程并行合并
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        ns = self.f_grid_size
        nx = self.x_grid_size
        nz = self.z_grid_size

        SEO_fi_axis = self.SEO_fi_axis
        SEO_Ek_axis = self.SEO_Ek_axis
        SEO_r_matrix = self.SEO_r_matrix
        SEO_perimeter = self.SEO_perimeter

        # 1. 获取当前线程存活粒子的极坐标, coords_mean_gamma是预生成的全为1的数组
        (coords_r, coords_fi, coords_z, coords_gamma, coords_mean_gamma, coords_fi_mod, coords_survive_idx, survive_num, sum_gamma_local) = GetLocalCoordinates_2p5(self.LocalBunch,
                                                                                          self.Step_SurviveFlag,
                                                                                          self.Malloc_SC_r,
                                                                                          self.Malloc_SC_fi,
                                                                                          self.Malloc_SC_z,
                                                                                          self.Malloc_SC_gamma,self.Malloc_SC_mean_gamma_interp,
                                                                                          self.Malloc_SC_fiMod, self.Malloc_SC_surviveIDX, self.speed_c)

        # 2. 当前线程的 φ 覆盖范围
        arc_start_local, arc_end_local, _ = divide_arc2(coords_fi_mod)
        # 3. 所有线程收集 arcs
        local_arc = np.array([arc_start_local, arc_end_local])
        all_arcs = np.zeros((size, 2))
        comm.Allgather(local_arc, all_arcs)
        arcs_list = [tuple(row) for row in all_arcs]
        # 4. 全局 φ 范围合并
        fi_s_global, fi_e_global = merge_multiple_arcs(arcs_list)
        if fi_e_global <= fi_s_global:
            fi_e_global += 2 * np.pi
        # 5. φ 方向均匀划分
        slice_fi_step = (fi_e_global - fi_s_global) / (ns-1)
        fi_grid = np.linspace(fi_s_global, fi_e_global, ns)

        # MPI 汇总
        sum_gamma_global = comm.allreduce(sum_gamma_local, op=MPI.SUM)
        count_global = comm.allreduce(survive_num, op=MPI.SUM)

        if count_global > 0:
            mean_gamma_global = sum_gamma_global / count_global - 1.0
        else:
            mean_gamma_global = 0.0

        # 9. 根据平均gamma对应的SEO,
        coords_r0, _ = Bilinear_interp_2D_vect_uniform(SEO_Ek_axis[0], SEO_Ek_axis[1] - SEO_Ek_axis[0],
                                                       len(SEO_Ek_axis),
                                                       SEO_fi_axis[0], SEO_fi_axis[1] - SEO_fi_axis[0],
                                                       len(SEO_fi_axis),
                                                       SEO_r_matrix,
                                                       # coords_mean_gamma * mean_gamma_global * self.E0_rest_MeV,
                                                       (coords_gamma-1) * self.E0_rest_MeV,
                                                       coords_fi_mod,
                                                       self.Malloc_SC_r_SEO, self.Malloc_SC_interp_flag)

        # 将r转换为x, 然后计算xz的统计量，再计算xz平均值和sigma, 再将fi转换为s
        (coords_x, r_slices, z_slices, x_slices, count_slices, slice_indices,
         sum_x_local, sum_z_local, sum_x2_local, sum_z2_local) = (
            ComputeSliceBoundsAndCoordsSBS_merged(coords_r, coords_z, coords_fi_mod, coords_r0,
                                               self.Malloc_SC_x,
                                               slice_fi_step, fi_s_global, ns,
                                               self.Malloc_SC_slice_r2d, self.Malloc_SC_slice_z2d,
                                                  self.Malloc_SC_slice_x2d,
                                               self.Malloc_SC_slice_indices))

        # 6. 粒子分配到 φ slice（考虑跨越 2π）
        # 7. 每 slice 分别计算局部 rmin/rmax/zmin/zmax
        # MPI gather 缓存，改为 numpy 数组
        sum_x_global = np.zeros(1, dtype=np.float64)
        sum_z_global = np.zeros(1, dtype=np.float64)
        sum_x2_global = np.zeros(1, dtype=np.float64)
        sum_z2_global = np.zeros(1, dtype=np.float64)
        count_slices_global = np.zeros_like(count_slices)

        # Allreduce
        comm.Allreduce(np.array([sum_x_local,]), sum_x_global, op=MPI.SUM)
        comm.Allreduce(np.array([sum_z_local,]), sum_z_global, op=MPI.SUM)
        comm.Allreduce(np.array([sum_x2_local,]), sum_x2_global, op=MPI.SUM)
        comm.Allreduce(np.array([sum_z2_local,]), sum_z2_global, op=MPI.SUM)
        comm.Allreduce(count_slices, count_slices_global, op=MPI.SUM)

        if count_global > 0:
            global_mean_x = np.sum(sum_x_global) / count_global
            global_mean_z = np.sum(sum_z_global) / count_global

            global_var_x = np.sum(sum_x2_global) / count_global - global_mean_x ** 2
            global_var_z = np.sum(sum_z2_global) / count_global - global_mean_z ** 2

            global_sigma_x = np.sqrt(max(global_var_x, 0.0))
            global_sigma_z = np.sqrt(max(global_var_z, 0.0))
        else:
            global_mean_x = global_mean_z = global_sigma_x = global_sigma_z = 0.0

        # 获取束长
        SEO_perimeter_MeanGamma = np.interp(mean_gamma_global * self.E0_rest_MeV, SEO_Ek_axis, SEO_perimeter)
        BunchLength = SEO_perimeter_MeanGamma  * (fi_e_global - fi_s_global)/ np.pi / 2.0
        s_grid = fi_grid/np.pi/2.0*SEO_perimeter_MeanGamma
        scale = self.sigma_ratio

        # 每个slice、local粒子的边界
        global_xmin = global_mean_x - scale * global_sigma_x
        global_xmax = global_mean_x + scale * global_sigma_x
        global_zmin = global_mean_z - scale * global_sigma_z
        global_zmax = global_mean_z + scale * global_sigma_z

        # 9. 生成一个x-z的mesh网格，然后分配当前线程的电荷到网格点上，形成二维xz电荷分布和一维纵向分布rho_f
        q_part = self.marcosize * self.q_charge

        rho_xz_slices_ = DistributeChargeSBS_Njit(z_slices, x_slices, count_slices, q_part,
                                                        BunchLength, global_xmin, global_xmax,
                                                        global_zmin, global_zmax, nx, nz, ns)

        rho_f = count_slices_global * q_part

        # ===== 11. per-slice 2D 卷积（owner-compute，最后 Allreduce 立方）=====
        dx = (global_xmax - global_xmin) / nx
        dz = (global_zmax - global_zmin) / nz
        if dx <= 0.0 or dz <= 0.0:
            # 无场：直接清零并返回/继续
            self.Malloc_SC_Ex_p[:] = 0.0
            self.Malloc_SC_Ez_p[:] = 0.0

            Ef_map = np.zeros((ns,))
            Ex_cube = np.zeros((ns, nz, nx), dtype=np.float32)  # 本进程持有的立方 非owner的 slice 行保持0
            Ez_cube = np.zeros((ns, nz, nx), dtype=np.float32)
            phi_cube = np.zeros((ns, nz, nx), dtype=np.float32)
        else:
            # 1) 频域核 / 纵向场 / 3D 立方, np.float32
            fft_kernel = make_fft_kernel_open_numba(nx, nz, dx, dz)
            # rho_f 建议用全局值（如前面未Allreduce，这里补一次）
            rho_f_global = rho_f.copy()
            comm.Allreduce(MPI.IN_PLACE, rho_f_global, op=MPI.SUM)
            comm.Allreduce(MPI.IN_PLACE, rho_xz_slices_, op=MPI.SUM)

            # Ex_cube = np.zeros((ns, nz, nx), dtype=np.float32)  # 本进程持有的立方 非owner的 slice 行保持0
            # Ez_cube = np.zeros((ns, nz, nx), dtype=np.float32)
            # phi_cube = np.zeros((ns, nz, nx), dtype=np.float32)

            Ef_map = total_field_1d_numba(s_grid, rho_f_global)

            # 计算rank负责的 k 段
            k0, k1, my_count = contiguous_block_for_rank(ns, size, rank)
            # print(f"rank={rank}, k0={k0}, k1={k1}, my_count={my_count}")
            # 本地存放rank的 φ
            phi_local = np.empty((my_count, nz, nx), dtype=np.float32)
            Ex_local = np.empty((my_count, nz, nx), dtype=np.float32)
            Ez_local = np.empty((my_count, nz, nx), dtype=np.float32)

            # 逐 k 计算（保留原来的单张 FFT 方式）
            for t, k in enumerate(range(k0, k1)):
                buf = fft_fwd_1.input_array  # complex64, (nz, nx)
                buf.real[:] = rho_xz_slices_[k]
                buf.imag[:] = 0.0
                fft_fwd_1()
                out = fft_fwd_1.output_array
                out *= fft_kernel
                fft_inv_1()
                phi_local[t] = fft_inv_1.output_array.real  # (nz, nx) float32

                compute_field_from_potential_numba(
                    phi_local[t], dx, dz, BunchLength,
                    Ex_local[t], Ez_local[t], mean_gamma_global)

            # 把所有进程的 φ 段按顺序拼成全局 φ 立方
            phi_cube = allgatherv_phi_3d_inplace(comm, phi_local, ns, nz, nx)
            Ex_cube = allgatherv_phi_3d_inplace(comm, Ex_local, ns, nz, nx)
            Ez_cube = allgatherv_phi_3d_inplace(comm, Ez_local, ns, nz, nx)

            # 5) 统一把场插值回本地粒子, 仅处理本地有粒子的片
            interpolate_fields_to_particles_SBS_numba(
                Ex_cube, Ez_cube, Ef_map,
                coords_x, coords_z, slice_indices, coords_survive_idx,
                global_xmin, global_zmin, dx, dz, BunchLength,
                self.Malloc_SC_Ex_p, self.Malloc_SC_Ez_p, self.Malloc_SC_Ef_p, self.LocalBunch)

        return rho_xz_slices_, phi_cube, Ex_cube, Ez_cube, global_xmin, global_xmax, global_zmin, global_zmax, BunchLength, Ef_map, s_grid, rho_f, coords_r, coords_fi, coords_x, coords_r0, coords_z, fi_grid


    def Build_2_5D_SliceGrid_sbs_LinearFast(self, fft_fwd_1, fft_inv_1):
        """
        最简版：保留 slice 划分 + 纵向电荷统计 + 线性(KV-like)空间电荷力
        去掉 PIC：不做 2D 网格沉积、不做 FFT Poisson、不做网格插值
        只直接给粒子 Ex/Ez（线性力）+ Ef（1D 纵向场）
        """

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        ns = self.f_grid_size
        nx = self.x_grid_size
        nz = self.z_grid_size

        SEO_fi_axis = self.SEO_fi_axis
        SEO_Ek_axis = self.SEO_Ek_axis
        SEO_r_matrix = self.SEO_r_matrix
        SEO_pr_matrix = self.SEO_pr_matrix #pr/pfi
        SEO_perimeter = self.SEO_perimeter

        # 1) 本地存活粒子坐标（极坐标）
        if self.BunchType == "Boris":
            (coords_r, coords_fi, coords_z, coords_gamma, coords_mean_gamma,
             coords_fi_mod, coords_survive_idx, survive_num, sum_gamma_local) = GetLocalCoordinates_2p5(self.LocalBunch,
                                                                                                        self.Step_SurviveFlag,
                                                                                                        self.Malloc_SC_r,
                                                                                                        self.Malloc_SC_fi,
                                                                                                        self.Malloc_SC_z,
                                                                                                        self.Malloc_SC_gamma,
                                                                                                        self.Malloc_SC_mean_gamma_interp,
                                                                                                        self.Malloc_SC_fiMod,
                                                                                                        self.Malloc_SC_surviveIDX,
                                                                                                        self.speed_c)
        elif self.BunchType == "RK4":
            (coords_r, coords_fi, coords_z, coords_gamma, coords_mean_gamma,
             coords_fi_mod, coords_survive_idx, survive_num, sum_gamma_local) = GetLocalCoordinates_2p5_rk(
                self.LocalBunch,
                self.Step_SurviveFlag,
                self.Malloc_SC_r,
                self.Malloc_SC_fi,
                self.Malloc_SC_z,
                self.Malloc_SC_gamma,
                self.Malloc_SC_mean_gamma_interp,
                self.Malloc_SC_fiMod,
                self.Malloc_SC_surviveIDX,
                self.speed_c)
        else:
            raise ("BunchType must be Boris or RK4")

        # 2) 本地 φ 覆盖范围 -> 全局 φ 范围
        arc_start_local, arc_end_local, _ = divide_arc2(coords_fi_mod)
        local_arc = np.array([arc_start_local, arc_end_local])
        all_arcs = np.zeros((size, 2))
        comm.Allgather(local_arc, all_arcs)
        arcs_list = [tuple(row) for row in all_arcs]
        fi_s_global, fi_e_global = merge_multiple_arcs(arcs_list)
        if fi_e_global <= fi_s_global:
            fi_e_global += 2.0 * np.pi

        # 3) φ 网格
        slice_fi_step = (fi_e_global - fi_s_global) / (ns - 1)
        fi_grid = np.linspace(fi_s_global, fi_e_global, ns)

        # 4) 全局平均 gamma
        sum_gamma_global = comm.allreduce(sum_gamma_local, op=MPI.SUM)
        count_global = comm.allreduce(survive_num, op=MPI.SUM)
        if count_global > 0:
            mean_gamma_global = sum_gamma_global / count_global - 1.0
        else:
            mean_gamma_global = 0.0

        # 5) SEO 插值闭轨 r0(phi;Ek) -> 用粒子自己的 (gamma-1)Ek
        coords_r0, _ = Bilinear_interp_2D_vect_uniform(
            SEO_Ek_axis[0], SEO_Ek_axis[1] - SEO_Ek_axis[0], len(SEO_Ek_axis),
            SEO_fi_axis[0], SEO_fi_axis[1] - SEO_fi_axis[0], len(SEO_fi_axis),
            SEO_r_matrix,
            (coords_gamma-1) * self.E0_rest_MeV,
            coords_fi_mod,
            self.Malloc_SC_r_SEO,
            self.Malloc_SC_interp_flag
        )
        coords_pr0, _ = Bilinear_interp_2D_vect_uniform(
            SEO_Ek_axis[0], SEO_Ek_axis[1] - SEO_Ek_axis[0], len(SEO_Ek_axis),
            SEO_fi_axis[0], SEO_fi_axis[1] - SEO_fi_axis[0], len(SEO_fi_axis),
            SEO_pr_matrix,
            (coords_gamma - 1.0) * self.E0_rest_MeV,
            coords_fi_mod,
            self.Malloc_SC_pr_SEO,
            self.Malloc_SC_interp_flag
        )

        # 6) 计算 x、slice_index、以及 slice 统计（count_slices 等）
        (coords_x, r_slices, z_slices, x_slices, count_slices, slice_indices,
         sum_x_local, sum_z_local, sum_x2_local, sum_z2_local) = (
            ComputeSliceBoundsAndCoordsSBS_merged(
                coords_r, coords_z, coords_fi_mod, coords_r0,
                self.Malloc_SC_x,
                slice_fi_step, fi_s_global, ns,
                self.Malloc_SC_slice_r2d, self.Malloc_SC_slice_z2d, self.Malloc_SC_slice_x2d,
                self.Malloc_SC_slice_indices
            )
        )
        
        # 7) 全局均值/方差（用二阶矩）
        sum_x_global  = np.zeros(1, dtype=np.float64)
        sum_z_global  = np.zeros(1, dtype=np.float64)
        sum_x2_global = np.zeros(1, dtype=np.float64)
        sum_z2_global = np.zeros(1, dtype=np.float64)
        count_slices_global = np.zeros_like(count_slices)

        comm.Allreduce(np.array([sum_x_local, ]),  sum_x_global,  op=MPI.SUM)
        comm.Allreduce(np.array([sum_z_local, ]),  sum_z_global,  op=MPI.SUM)
        comm.Allreduce(np.array([sum_x2_local, ]), sum_x2_global, op=MPI.SUM)
        comm.Allreduce(np.array([sum_z2_local, ]), sum_z2_global, op=MPI.SUM)
        comm.Allreduce(count_slices, count_slices_global, op=MPI.SUM)

        if count_global > 1:
            global_mean_x = sum_x_global / count_global
            global_mean_z = sum_z_global / count_global

            global_var_x = sum_x2_global / count_global - global_mean_x ** 2
            global_var_z = sum_z2_global / count_global - global_mean_z ** 2

            global_sigma_x = np.sqrt(max(global_var_x, 0.0))
            global_sigma_z = np.sqrt(max(global_var_z, 0.0))
        else:
            global_mean_x = global_mean_z = 0.0
            global_sigma_x = global_sigma_z = 4.5/100

        # 8) 束长 + s_grid
        SEO_perimeter_MeanGamma = np.interp(mean_gamma_global * self.E0_rest_MeV, SEO_Ek_axis, SEO_perimeter)
        BunchLength = SEO_perimeter_MeanGamma * (fi_e_global - fi_s_global) / np.pi / 2.0
        s_grid = fi_grid / np.pi / 2.0 * SEO_perimeter_MeanGamma

        scale = self.sigma_ratio
        global_xmin = global_mean_x - scale * global_sigma_x
        global_xmax = global_mean_x + scale * global_sigma_x
        global_zmin = global_mean_z - scale * global_sigma_z
        global_zmax = global_mean_z + scale * global_sigma_z

        # 9) 只做 1D 纵向电荷：rho_f = 每个slice电荷
        q_part = self.marcosize * self.q_charge
        count_slices_global = np.ones((ns,))*count_global/ns
        rho_f = count_slices_global * q_part  # (ns,)
        rho_f_global = rho_f.copy()
        # comm.Allreduce(MPI.IN_PLACE, rho_f_global, op=MPI.SUM)

        # 10) 纵向场（可保留）
        Ef_map = total_field_1d_numba(s_grid, rho_f_global)

        # 11) 线性(KV-like)横向场：Ex ~ lambda_k * x, Ez ~ lambda_k * z
        #     这里用 a=scale*sigma_x, b=scale*sigma_z
        # a = scale * global_sigma_x
        # b = scale * global_sigma_z
        a = scale * 3.0/100
        b = scale * 3.0/100
        ds = BunchLength / ns
        lambda_k = rho_f_global / ds  # 线密度 (ns,)

        Ksc = 2.0e12  # 设一个常数匹配 tune shift

        self.Malloc_SC_Ex_p[:] = 0.0
        self.Malloc_SC_Ez_p[:] = 0.0
        self.Malloc_SC_Ef_p[:] = 0.0

        for j in range(survive_num):
            i = coords_survive_idx[j]          # 回填到原粒子索引
            k = int(slice_indices[j])          # slice id（与 survive 对齐）
            x = coords_x[j]
            z = coords_z[j]

            self.Malloc_SC_Ex_p[i] = (Ksc * lambda_k[k] * x / (a * (a + b)))*(1/np.sqrt(1+coords_pr0[i]*coords_pr0[i]))
            self.Malloc_SC_Ez_p[i] = Ksc * lambda_k[k] * z / (b * (a + b))
            self.Malloc_SC_Ef_p[i] = (Ksc * lambda_k[k] * x / (a * (a + b)))*(-coords_pr0[i]/np.sqrt(1+coords_pr0[i]*coords_pr0[i]))
            # print(self.Malloc_SC_Ex_p[i])
            self.LocalBunch[i, 10] = (Ksc * lambda_k[k] * x / (a * (a + b)))*(1/np.sqrt(1+coords_pr0[i]*coords_pr0[i]))
            self.LocalBunch[i, 11] = Ksc * lambda_k[k] * z / (b * (a + b))
            self.LocalBunch[i, 12] = (Ksc * lambda_k[k] * x / (a * (a + b)))*(-coords_pr0[i]/np.sqrt(1+coords_pr0[i]*coords_pr0[i]))


        # 不再返回 PIC 立方
        rho_xz_slices_ = None
        phi_cube = None
        Ex_cube = None
        Ez_cube = None

        return (rho_xz_slices_, phi_cube, Ex_cube, Ez_cube,
                global_xmin, global_xmax, global_zmin, global_zmax,
                BunchLength, Ef_map, s_grid, rho_f,
                coords_r, coords_fi, coords_x, coords_r0, coords_z, fi_grid)


if __name__ == "__main__":
    # 初始化 MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 定义一个模拟的粒子分布
    np.random.seed(int(MPI.Wtime() * 1000) + rank)  # 使用当前时间和rank作为随机种子
    ParticleNum = 20
    # 创建包含 (r, vr, z, vz, fi, Ek, inj_t, ini_flag, RF_phase, Esc_r, Esc_z, Esc_fi, Bunch_ID, Local_ID, Global_ID) 的粒子
    ParticlesDistribution = np.column_stack((
        np.random.random(ParticleNum) * 10,  # r
        np.random.random(ParticleNum),  # vr
        np.random.random(ParticleNum) * 10,  # z
        np.random.random(ParticleNum),  # vz
        np.random.random(ParticleNum) * 2 * np.pi,  # fi
        np.random.random(ParticleNum) * 100,  # Ek
        np.random.random(ParticleNum) * 5,  # inj_t
        np.ones(ParticleNum),  # ini_flag
        np.random.random(ParticleNum) * 2 * np.pi,  # RF_phase
        np.zeros(ParticleNum),  # Esc_r (initially zero)
        np.zeros(ParticleNum),  # Esc_z (initially zero)
        np.zeros(ParticleNum),  # Esc_fi (initially zero)
        np.zeros(ParticleNum),  # Bunch_ID
        np.zeros(ParticleNum),  # Local_ID (will be set by the class)
        np.zeros(ParticleNum)  # Global_ID (will be set by the class)
    ))

    # 创建一个 FFAG_Bunch 实例
    bunch = FFAG_Bunch(ParticlesDistribution)

    # 在所有线程中计算全局的坐标边界
    xmin_global, xmax_global, ymin_global, ymax_global, zmin_global, zmax_global = bunch.Get_SC_Grid_para_global()

    # 每个线程打印其局部的和全局的坐标边界，保留两位小数
    print(f"Rank {rank}:")
    print(
        f"  Local xmin: {bunch.xmin_Local:.2f}, xmax: {bunch.xmax_Local:.2f}, ymin: {bunch.ymin_Local:.2f}, ymax: {bunch.ymax_Local:.2f}, zmin: {bunch.zmin_Local:.2f}, zmax: {bunch.zmax_Local:.2f}")
    print(
        f"  Global xmin: {xmin_global:.2f}, xmax: {xmax_global:.2f}, ymin: {ymin_global:.2f}, ymax: {ymax_global:.2f}, zmin: {zmin_global:.2f}, zmax: {zmax_global:.2f}\n")
