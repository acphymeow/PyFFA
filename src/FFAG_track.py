"""
The Python-FFAG code is a RK solver for the beam dynamics simulations of the FFAG accelerator.
Contact Author: zhoukai@ihep.ac.cn
Version and Date:
V0.0 @ 2023.04.28
V0.1 @ 2023.11.20, Generating field maps, 6D tracking, search SEO, calculate twiss parameters
V0.2 @ 2024.11.01, 3d space charge modal, high-order tayler expansion modual
"""
import os
import math
import copy
import datetime
import time
import numpy as np
from mpi4py import MPI
from numpy.core.numeric import ones_like
from scipy.interpolate import interp1d
from FFAG_Utils import FFAG_FileOperation
from FFAG_MathTools import (FFAG_interpolation, FFAG_Algorithm, fast_mod_njit,
                            boris_push_cartesian_njit, fast_mod_parallel,
                            cylindrical_to_cartesian, convert_cartesian_to_cylindrical,
                            boris_push_cylindrical_speed_njit, UpdataPreStepNjit,
                            RK4_equations_of_motion_njit, RK4_Post_Step_njit)
from FFAG_ParasAndConversion import FFAG_ConversionTools
from FFAG_SC import Bunch_SC_Calculator, Bunch_SC_Calculator_FlatCoordinate
import pyfftw


class FFAG_RungeKutta:
    def __init__(self):
        self.running = True

    def rk4_step(self, func, t, r, h, GlobalParameters):
        # shape of r = number of particles * number of coordinates
        k1 = h * func(t, r, GlobalParameters)
        k2 = h * func(t + 0.5 * h, r + 0.5 * k1, GlobalParameters)
        k3 = h * func(t + 0.5 * h, r + 0.5 * k2, GlobalParameters)
        k4 = h * func(t + h, r + k3, GlobalParameters)
        # shape of k1,...,k4 = n*7
        return (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def rk4_step_2(self, func, t, r, h, GlobalParameters):
        k1 = h * func(t, r, GlobalParameters)[0]
        k2 = h * func(t + 0.5 * h, r + 0.5 * k1, GlobalParameters)[0]
        k3 = h * func(t + 0.5 * h, r + 0.5 * k2, GlobalParameters)[0]
        k4 = h * func(t + h, r + k3, GlobalParameters)[0]
        _, Bz_ThisStep, Br_ThisStep, Bfi_ThisStep = func(t, r, GlobalParameters)
        return (k1 + 2 * k2 + 2 * k3 + k4) / 6, Bz_ThisStep, Br_ThisStep, Bfi_ThisStep

    # @profile
    def rk4_step_malloc(self, func, t, r, h, GlobalParameters,
                        Bz0, Br, Bfi, E_z, E_r, E_fi, fi_vect, malloc_RFGap_index, dxdt,
                        SurvivedNum_Local, LostNum_Local,):

        k1_temp, Bz_ThisStep, Br_ThisStep, Bfi_ThisStep = func(t, r, GlobalParameters, Bz0, Br, Bfi, E_z, E_r, E_fi, fi_vect, malloc_RFGap_index, dxdt)
        k1 = h * k1_temp
        k2 = h * func(t + 0.5 * h, r + 0.5 * k1, GlobalParameters, Bz0, Br, Bfi, E_z, E_r, E_fi, fi_vect, malloc_RFGap_index, dxdt)[0]
        k3 = h * func(t + 0.5 * h, r + 0.5 * k2, GlobalParameters, Bz0, Br, Bfi, E_z, E_r, E_fi, fi_vect, malloc_RFGap_index, dxdt)[0]
        k4 = h * func(t + h, r + k3, GlobalParameters, Bz0, Br, Bfi, E_z, E_r, E_fi, fi_vect, malloc_RFGap_index, dxdt)[0]

        return (k1 + 2 * k2 + 2 * k3 + k4) / 6, Bz_ThisStep, Br_ThisStep, Bfi_ThisStep


    # @profile
    def rk4_solve_dt_bunch3_withSC(self, func, t_start, BunchObj,
                            GlobalParameters, LocalParameters):

        comm = MPI.COMM_WORLD

        # parameters for tracking
        stop_condition = LocalParameters['stop_condition']
        step_t = LocalParameters['time_step']
        enable_SC = LocalParameters['enable_SC']
        SC_type = LocalParameters['SC_type']
        SC_step = int(LocalParameters['SC_step'])
        StepDumpInfo = LocalParameters['step_dumps']

        max_stepsN = stop_condition['max_stepsN']
        max_turn = stop_condition['max_turn']
        max_time = stop_condition['max_time']
        max_fi = max_turn * np.pi * 2

        t_PreStep, steps_i = t_start, 0
        ExitLoopFlag = False

        # FFTW plan
        NZ, NX, NF = BunchObj.z_grid_size, BunchObj.x_grid_size, BunchObj.f_grid_size
        pyfftw.interfaces.cache.enable()
        a_buf = pyfftw.empty_aligned((NZ, NX), dtype='complex64')
        b_buf = pyfftw.empty_aligned((NZ, NX), dtype='complex64')
        fft_fwd = pyfftw.FFTW(a_buf, b_buf, axes=(0, 1),
                              direction='FFTW_FORWARD',
                              threads=1, flags=('FFTW_MEASURE',))
        fft_inv = pyfftw.FFTW(b_buf, a_buf, axes=(0, 1),
                              direction='FFTW_BACKWARD',
                              threads=1, flags=('FFTW_MEASURE',))

        # start tracking
        while steps_i < max_stepsN:

            BunchObj.InjectParticles(t_PreStep, step_t)
            BunchObj.UpdatePreSteps()
            if ExitLoopFlag:
                # if there are no particles in the threads,
                # execute an empty statement to ensure synchronization among threads
                pass
            else:
                # the pre step
                if enable_SC:
                    if np.mod(steps_i, SC_step) == 0:
                        if SC_type == 0:
                            # Step 3: 计算空间电荷电场
                            # BunchObj.UpdateGlobalBunch()
                            BunchObj.Get_SC_Grid_para_global()
                            BunchObj.DistributeChargeGlobal(56, 128, 28)
                            # Step 4: 计算空间电荷电场, 将插值后的电场值存储到 Bunch_obj中
                            _, _, _, _, _, _, _, _ = (
                                Bunch_SC_Calculator(BunchObj))
                        elif SC_type == 1:
                            # 计算2.5D空间电荷电场
                            BunchObj.Build_2_5D_SliceGrid_sbs(fft_fwd, fft_inv)
                            alive_now = (BunchObj.LocalBunch[:, 8] != 0)
                            # ===== 初始化 =====
                            if steps_i == 0:
                                # 存活标志
                                BunchObj.Malloc_SC_survive_flag_Pre[:] = alive_now
                                BunchObj.Malloc_SC_survive_flag_PrePre[:] = False
                                # 写入当前端点场
                                BunchObj.Malloc_SC_Ex_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 10]
                                BunchObj.Malloc_SC_Ez_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 11]
                                BunchObj.Malloc_SC_Ef_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 12]
                                # 初始化时：Pre_Pre = Pre
                                BunchObj.Malloc_SC_Ex_Pre_Pre[:] = BunchObj.Malloc_SC_Ex_Pre
                                BunchObj.Malloc_SC_Ez_Pre_Pre[:] = BunchObj.Malloc_SC_Ez_Pre
                                BunchObj.Malloc_SC_Ef_Pre_Pre[:] = BunchObj.Malloc_SC_Ef_Pre
                            # ===== 每个宏步端点 =====
                            elif steps_i % SC_step == 0:
                                # 滚动存活标志
                                BunchObj.Malloc_SC_survive_flag_PrePre[:] = BunchObj.Malloc_SC_survive_flag_Pre
                                BunchObj.Malloc_SC_survive_flag_Pre[:] = alive_now
                                # 滚动端点场
                                BunchObj.Malloc_SC_Ex_Pre_Pre[:] = BunchObj.Malloc_SC_Ex_Pre
                                BunchObj.Malloc_SC_Ez_Pre_Pre[:] = BunchObj.Malloc_SC_Ez_Pre
                                BunchObj.Malloc_SC_Ef_Pre_Pre[:] = BunchObj.Malloc_SC_Ef_Pre
                                # 新端点场（只对当前存活的粒子）
                                BunchObj.Malloc_SC_Ex_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 10]
                                BunchObj.Malloc_SC_Ez_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 11]
                                BunchObj.Malloc_SC_Ef_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 12]
                        elif SC_type == 2:
                            BunchObj.Build_2_5D_SliceGrid_sbs_LinearFast(fft_fwd, fft_inv)
                            alive_now = (BunchObj.LocalBunch[:, 8] != 0)
                            # ===== 初始化 =====
                            if steps_i == 0:
                                # 存活标志
                                BunchObj.Malloc_SC_survive_flag_Pre[:] = alive_now
                                BunchObj.Malloc_SC_survive_flag_PrePre[:] = False
                                # 写入当前端点场
                                BunchObj.Malloc_SC_Ex_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 10]
                                BunchObj.Malloc_SC_Ez_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 11]
                                BunchObj.Malloc_SC_Ef_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 12]
                                # 初始化时：Pre_Pre = Pre
                                BunchObj.Malloc_SC_Ex_Pre_Pre[:] = BunchObj.Malloc_SC_Ex_Pre
                                BunchObj.Malloc_SC_Ez_Pre_Pre[:] = BunchObj.Malloc_SC_Ez_Pre
                                BunchObj.Malloc_SC_Ef_Pre_Pre[:] = BunchObj.Malloc_SC_Ef_Pre
                            # ===== 每个宏步端点 =====
                            elif steps_i % SC_step == 0:
                                # 滚动存活标志
                                BunchObj.Malloc_SC_survive_flag_PrePre[:] = BunchObj.Malloc_SC_survive_flag_Pre
                                BunchObj.Malloc_SC_survive_flag_Pre[:] = alive_now
                                # 滚动端点场
                                BunchObj.Malloc_SC_Ex_Pre_Pre[:] = BunchObj.Malloc_SC_Ex_Pre
                                BunchObj.Malloc_SC_Ez_Pre_Pre[:] = BunchObj.Malloc_SC_Ez_Pre
                                BunchObj.Malloc_SC_Ef_Pre_Pre[:] = BunchObj.Malloc_SC_Ef_Pre
                                # 新端点场（只对当前存活的粒子）
                                BunchObj.Malloc_SC_Ex_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 10]
                                BunchObj.Malloc_SC_Ez_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 11]
                                BunchObj.Malloc_SC_Ef_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 12]
                        else:
                            raise TypeError('SC_type must be 0, 1 or 2')
                
                
                # ===== 宏步内外推 =====
                k = steps_i % SC_step
                s = k / SC_step if steps_i >= SC_step else 0.0

                pre_mask = BunchObj.Malloc_SC_survive_flag_Pre
                prepre_mask = BunchObj.Malloc_SC_survive_flag_PrePre

                both_mask = pre_mask & prepre_mask  # 两个端点都有 → 外推
                only_pre = pre_mask & (~prepre_mask)  # 只有 Pre → 用 Pre

                # 外推
                BunchObj.Malloc_SC_Ex_p[both_mask] = BunchObj.Malloc_SC_Ex_Pre[both_mask] + \
                                                     s * (BunchObj.Malloc_SC_Ex_Pre[both_mask] -
                                                          BunchObj.Malloc_SC_Ex_Pre_Pre[both_mask])
                BunchObj.Malloc_SC_Ez_p[both_mask] = BunchObj.Malloc_SC_Ez_Pre[both_mask] + \
                                                     s * (BunchObj.Malloc_SC_Ez_Pre[both_mask] -
                                                          BunchObj.Malloc_SC_Ez_Pre_Pre[both_mask])
                BunchObj.Malloc_SC_Ef_p[both_mask] = BunchObj.Malloc_SC_Ef_Pre[both_mask] + \
                                                     s * (BunchObj.Malloc_SC_Ef_Pre[both_mask] -
                                                          BunchObj.Malloc_SC_Ef_Pre_Pre[both_mask])

                # 新注入（只有 Pre）
                BunchObj.Malloc_SC_Ex_p[only_pre] = BunchObj.Malloc_SC_Ex_Pre[only_pre]
                BunchObj.Malloc_SC_Ez_p[only_pre] = BunchObj.Malloc_SC_Ez_Pre[only_pre]
                BunchObj.Malloc_SC_Ef_p[only_pre] = BunchObj.Malloc_SC_Ef_Pre[only_pre]

                # 写回 LocalBunch（只对 Pre==True 的粒子）
                BunchObj.LocalBunch[pre_mask, 10] = BunchObj.Malloc_SC_Ex_p[pre_mask]
                BunchObj.LocalBunch[pre_mask, 11] = BunchObj.Malloc_SC_Ez_p[pre_mask]
                BunchObj.LocalBunch[pre_mask, 12] = BunchObj.Malloc_SC_Ef_p[pre_mask]
                
                # the integration over the step
                dr_step, Bz_ThisStep, Br_ThisStep, Bfi_ThisStep = self.rk4_step_malloc(func, t_PreStep,
                                                                                       BunchObj.LocalBunch,
                                                                                       step_t, GlobalParameters,
                                                                                       BunchObj.Malloc_Bz_interp, BunchObj.Malloc_Br_interp, BunchObj.Malloc_Bf_interp,
                                                                                       BunchObj.Malloc_Ez_interp, BunchObj.Malloc_Er_interp, BunchObj.Malloc_Ef_interp,
                                                                                       BunchObj.Malloc_mod_fi, BunchObj.Malloc_RF_shift, BunchObj.Malloc_RK_dxdt,
                                                                                       BunchObj.SurvivedNum_Local, BunchObj.LostNum_Local)

                # the post step
                RK4_Post_Step_njit(BunchObj.LocalBunch, dr_step)
                t_PostStep = t_PreStep + step_t

                # 更新 Post-steps (积分之后)
                BunchObj.UpdatePostSteps()

                #  save coordinates with the given time interval
                StepDumpInfo.check_and_dump(t_PreStep, t_PostStep, BunchObj)

                # update the pre step
                t_PreStep = t_PostStep
                if np.mod(steps_i, 1000) == 0:
                    BunchObj.UpdateParticleNumGlobal()
                    if comm.Get_rank() == 0:
                        # 筛选非 NaN 的数据
                        filtered_fi = BunchObj.LocalBunch[BunchObj.Step_SurviveFlag, 4]

                        if len(filtered_fi) > 0:  # 检查是否有有效数据
                            print(
                                f"current step={steps_i:.0f}, "
                                f"r ~= {BunchObj.LocalBunch[BunchObj.Step_SurviveFlag, 0][0]:.2f} m, "
                                f"turn ~= {filtered_fi[0] / np.pi / 2:.2f}, "
                                f"remain particles: {BunchObj.SurvivedNum}. "
                                f"tracking time: {t_PostStep * 1e6:.2f} us,",
                                f"clock: {time.strftime('%m-%d %H:%M:%S', time.localtime())}",
                                flush=True
                            )
                        else:  # 如果没有有效数据
                            print(
                                f"current step={steps_i:.0f}, "
                                f"r = nan, "
                                f"turn = nan, "
                                f"remain particles: {BunchObj.SurvivedNum}. "
                                f"tracking time: {t_PostStep*1e6:.2f} us. ",
                                f"clock: {time.strftime('%m-%d %H:%M:%S', time.localtime())}",
                                flush=True)

                # # # variable time steps
                # # check the step length every 1/2 turn, adapt the step length if needed
                # if np.mod(steps_i, int(steps_oneturn / 2)) == 0:
                #     dfidt_mean = BunchObj.GetMeanDfiDt()
                #     step_fi_ThisStep = dfidt_mean * step_t
                #     if np.abs(step_fi_ThisStep) > np.abs(max_step_fi) or np.abs(step_fi_ThisStep) < np.abs(min_step_fi):
                #         step_t = step_fi / dfidt_mean * TrackDirection

                # update loop index
                steps_i += 1

            comm.barrier()

            # Exit the loop
            if np.mod(steps_i, 1000) == 0:
                # stop conditions
                StopFlag_fi_local = copy.deepcopy(BunchObj.LocalBunch[:, 4])
                StopFlag_time_local = t_PreStep
                # 使用allgather将各线程汇总到一个进程中
                StopFlag_fi_gather = comm.allgather(StopFlag_fi_local)
                StopFlag_time_gather = comm.allgather(StopFlag_time_local)
                # 根进程拥有拼接后的数据
                StopFlag_fi_global = np.concatenate(StopFlag_fi_gather, axis=0)
                StopFlag_time_global = np.mean(StopFlag_time_gather)
                StopFlag_fi_survived_particles = StopFlag_fi_global[~np.isnan(StopFlag_fi_global)]

                if np.count_nonzero(np.isnan(StopFlag_fi_global)) > 0:
                    pass

                if np.abs(np.mean(StopFlag_fi_survived_particles)) > max_fi:
                    break
                if StopFlag_time_global > max_time:
                    break

        return 0

    # @profile
    def rk4_solve_dt_bunch3_withSC_boris(self, func, t_start, BunchObj,
                            GlobalParameters, LocalParameters):

        comm = MPI.COMM_WORLD

        # parameters for tracking
        stop_condition = LocalParameters['stop_condition']
        step_t = LocalParameters['time_step']
        enable_SC = LocalParameters['enable_SC']
        SC_type = LocalParameters['SC_type']
        SC_step = int(LocalParameters['SC_step'])
        StepDumpInfo = LocalParameters['step_dumps']

        max_stepsN = stop_condition['max_stepsN']
        max_turn = stop_condition['max_turn']
        max_time = stop_condition['max_time']
        max_fi = max_turn * np.pi * 2

        t_PreStep, steps_i = t_start, 0
        ExitLoopFlag = False

        # FFTW plan
        NZ, NX, NF = BunchObj.z_grid_size, BunchObj.x_grid_size, BunchObj.f_grid_size
        pyfftw.interfaces.cache.enable()
        a_buf = pyfftw.empty_aligned((NZ, NX), dtype='complex64')
        b_buf = pyfftw.empty_aligned((NZ, NX), dtype='complex64')
        fft_fwd = pyfftw.FFTW(a_buf, b_buf, axes=(0, 1),
                              direction='FFTW_FORWARD',
                              threads=1, flags=('FFTW_MEASURE',))
        fft_inv = pyfftw.FFTW(b_buf, a_buf, axes=(0, 1),
                              direction='FFTW_BACKWARD',
                              threads=1, flags=('FFTW_MEASURE',))

        # start tracking
        while steps_i < max_stepsN:

            BunchObj.InjectParticles(t_PreStep, step_t)
            BunchObj.UpdatePreSteps()
            if ExitLoopFlag:
                # if there are no particles in the threads,
                # execute an empty statement to ensure synchronization among threads
                pass
            else:
                # the pre step
                if enable_SC:
                    if np.mod(steps_i, SC_step) == 0:
                        if SC_type ==0:
                            # Step 3: 计算空间电荷电场
                            # BunchObj.UpdateGlobalBunch()
                            BunchObj.Get_SC_Grid_para_global()
                            BunchObj.DistributeChargeGlobal(256, 256, 128)
                            # Step 4: 计算空间电荷电场, 将插值后的电场值存储到 Bunch_obj中
                            _, _, _, _, _, _, _, _ = (
                                Bunch_SC_Calculator(BunchObj))
                        elif SC_type ==1:
                            pass
                            # 计算2.5D空间电荷电场
                            BunchObj.Build_2_5D_SliceGrid_sbs(fft_fwd, fft_inv)

                            alive_now = (BunchObj.LocalBunch[:, 8] != 0)
                            # ===== 初始化 =====
                            if steps_i == 0:
                                # 存活标志
                                BunchObj.Malloc_SC_survive_flag_Pre[:] = alive_now
                                BunchObj.Malloc_SC_survive_flag_PrePre[:] = False
                                # 写入当前端点场
                                BunchObj.Malloc_SC_Ex_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 10]
                                BunchObj.Malloc_SC_Ez_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 11]
                                BunchObj.Malloc_SC_Ef_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 12]
                                # 初始化时：Pre_Pre = Pre
                                BunchObj.Malloc_SC_Ex_Pre_Pre[:] = BunchObj.Malloc_SC_Ex_Pre
                                BunchObj.Malloc_SC_Ez_Pre_Pre[:] = BunchObj.Malloc_SC_Ez_Pre
                                BunchObj.Malloc_SC_Ef_Pre_Pre[:] = BunchObj.Malloc_SC_Ef_Pre
                            # ===== 每个宏步端点 =====
                            elif steps_i % SC_step == 0:
                                # 滚动存活标志
                                BunchObj.Malloc_SC_survive_flag_PrePre[:] = BunchObj.Malloc_SC_survive_flag_Pre
                                BunchObj.Malloc_SC_survive_flag_Pre[:] = alive_now
                                # 滚动端点场
                                BunchObj.Malloc_SC_Ex_Pre_Pre[:] = BunchObj.Malloc_SC_Ex_Pre
                                BunchObj.Malloc_SC_Ez_Pre_Pre[:] = BunchObj.Malloc_SC_Ez_Pre
                                BunchObj.Malloc_SC_Ef_Pre_Pre[:] = BunchObj.Malloc_SC_Ef_Pre
                                # 新端点场（只对当前存活的粒子）
                                BunchObj.Malloc_SC_Ex_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 10]
                                BunchObj.Malloc_SC_Ez_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 11]
                                BunchObj.Malloc_SC_Ef_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 12]

                        elif SC_type == 2:
                            pass
                            # 计算2.5D空间电荷电场
                            BunchObj.Build_2_5D_SliceGrid_sbs_LinearFast(fft_fwd, fft_inv)

                            alive_now = (BunchObj.LocalBunch[:, 8] != 0)
                            # ===== 初始化 =====
                            if steps_i == 0:
                                # 存活标志
                                BunchObj.Malloc_SC_survive_flag_Pre[:] = alive_now
                                BunchObj.Malloc_SC_survive_flag_PrePre[:] = False
                                # 写入当前端点场
                                BunchObj.Malloc_SC_Ex_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 10]
                                BunchObj.Malloc_SC_Ez_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 11]
                                BunchObj.Malloc_SC_Ef_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 12]
                                # 初始化时：Pre_Pre = Pre
                                BunchObj.Malloc_SC_Ex_Pre_Pre[:] = BunchObj.Malloc_SC_Ex_Pre
                                BunchObj.Malloc_SC_Ez_Pre_Pre[:] = BunchObj.Malloc_SC_Ez_Pre
                                BunchObj.Malloc_SC_Ef_Pre_Pre[:] = BunchObj.Malloc_SC_Ef_Pre
                            # ===== 每个宏步端点 =====
                            elif steps_i % SC_step == 0:
                                # 滚动存活标志
                                BunchObj.Malloc_SC_survive_flag_PrePre[:] = BunchObj.Malloc_SC_survive_flag_Pre
                                BunchObj.Malloc_SC_survive_flag_Pre[:] = alive_now
                                # 滚动端点场
                                BunchObj.Malloc_SC_Ex_Pre_Pre[:] = BunchObj.Malloc_SC_Ex_Pre
                                BunchObj.Malloc_SC_Ez_Pre_Pre[:] = BunchObj.Malloc_SC_Ez_Pre
                                BunchObj.Malloc_SC_Ef_Pre_Pre[:] = BunchObj.Malloc_SC_Ef_Pre
                                # 新端点场（只对当前存活的粒子）
                                BunchObj.Malloc_SC_Ex_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 10]
                                BunchObj.Malloc_SC_Ez_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 11]
                                BunchObj.Malloc_SC_Ef_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 12]

                            # BunchObj.Malloc_Ez_cart = BunchObj.LocalBunch[:, 2] * 2.0e7
                            # BunchObj.LocalBunch[:, 11] = BunchObj.Malloc_Ez_cart
                            # alive_now = (BunchObj.LocalBunch[:, 8] != 0)
                            # # ===== 初始化 =====
                            # if steps_i == 0:
                            #     # 存活标志
                            #     BunchObj.Malloc_SC_survive_flag_Pre[:] = alive_now
                            #     BunchObj.Malloc_SC_survive_flag_PrePre[:] = False
                            #     # 写入当前端点场
                            #     BunchObj.Malloc_SC_Ex_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 10]
                            #     BunchObj.Malloc_SC_Ez_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 11]
                            #     BunchObj.Malloc_SC_Ef_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 12]
                            #     # 初始化时：Pre_Pre = Pre
                            #     BunchObj.Malloc_SC_Ex_Pre_Pre[:] = BunchObj.Malloc_SC_Ex_Pre
                            #     BunchObj.Malloc_SC_Ez_Pre_Pre[:] = BunchObj.Malloc_SC_Ez_Pre
                            #     BunchObj.Malloc_SC_Ef_Pre_Pre[:] = BunchObj.Malloc_SC_Ef_Pre
                            # # ===== 每个宏步端点 =====
                            # elif steps_i % SC_step == 0:
                            #     # 滚动存活标志
                            #     BunchObj.Malloc_SC_survive_flag_PrePre[:] = BunchObj.Malloc_SC_survive_flag_Pre
                            #     BunchObj.Malloc_SC_survive_flag_Pre[:] = alive_now
                            #     # 滚动端点场
                            #     BunchObj.Malloc_SC_Ex_Pre_Pre[:] = BunchObj.Malloc_SC_Ex_Pre
                            #     BunchObj.Malloc_SC_Ez_Pre_Pre[:] = BunchObj.Malloc_SC_Ez_Pre
                            #     BunchObj.Malloc_SC_Ef_Pre_Pre[:] = BunchObj.Malloc_SC_Ef_Pre
                            #     # 新端点场（只对当前存活的粒子）
                            #     BunchObj.Malloc_SC_Ex_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 10]
                            #     BunchObj.Malloc_SC_Ez_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 11]
                            #     BunchObj.Malloc_SC_Ef_Pre[alive_now] = BunchObj.LocalBunch[alive_now, 12]
                        else:
                            raise TypeError('SC_type must be 0, 1 or 2')

                # ===== 宏步内外推 =====
                k = steps_i % SC_step
                s = k / SC_step if steps_i >= SC_step else 0.0

                pre_mask = BunchObj.Malloc_SC_survive_flag_Pre
                prepre_mask = BunchObj.Malloc_SC_survive_flag_PrePre

                both_mask = pre_mask & prepre_mask  # 两个端点都有 → 外推
                only_pre = pre_mask & (~prepre_mask)  # 只有 Pre → 用 Pre

                # 外推
                BunchObj.Malloc_SC_Ex_p[both_mask] = BunchObj.Malloc_SC_Ex_Pre[both_mask] + \
                                                     s * (BunchObj.Malloc_SC_Ex_Pre[both_mask] -
                                                          BunchObj.Malloc_SC_Ex_Pre_Pre[both_mask])
                BunchObj.Malloc_SC_Ez_p[both_mask] = BunchObj.Malloc_SC_Ez_Pre[both_mask] + \
                                                     s * (BunchObj.Malloc_SC_Ez_Pre[both_mask] -
                                                          BunchObj.Malloc_SC_Ez_Pre_Pre[both_mask])
                BunchObj.Malloc_SC_Ef_p[both_mask] = BunchObj.Malloc_SC_Ef_Pre[both_mask] + \
                                                     s * (BunchObj.Malloc_SC_Ef_Pre[both_mask] -
                                                          BunchObj.Malloc_SC_Ef_Pre_Pre[both_mask])

                # 新注入（只有 Pre）
                BunchObj.Malloc_SC_Ex_p[only_pre] = BunchObj.Malloc_SC_Ex_Pre[only_pre]
                BunchObj.Malloc_SC_Ez_p[only_pre] = BunchObj.Malloc_SC_Ez_Pre[only_pre]
                BunchObj.Malloc_SC_Ef_p[only_pre] = BunchObj.Malloc_SC_Ef_Pre[only_pre]

                # 写回 LocalBunch（只对 Pre==True 的粒子）
                BunchObj.LocalBunch[pre_mask, 10] = BunchObj.Malloc_SC_Ex_p[pre_mask]
                BunchObj.LocalBunch[pre_mask, 11] = BunchObj.Malloc_SC_Ez_p[pre_mask]
                BunchObj.LocalBunch[pre_mask, 12] = BunchObj.Malloc_SC_Ef_p[pre_mask]

                # the integration over the step
                self.Boris_solve(t_PreStep, BunchObj.LocalBunch, step_t,
                                 BunchObj.Malloc_x_cart, BunchObj.Malloc_y_cart, BunchObj.Malloc_z_cart,
                                 BunchObj.Malloc_vx_cart, BunchObj.Malloc_vy_cart, BunchObj.Malloc_vz_cart,
                                 BunchObj.Malloc_Ex_cart, BunchObj.Malloc_Ey_cart, BunchObj.Malloc_Ez_cart,
                                 BunchObj.Malloc_Bx_cart, BunchObj.Malloc_By_cart, BunchObj.Malloc_Bz_cart,
                                 BunchObj.Malloc_RF_phase, BunchObj.Malloc_mod_fi,
                                 BunchObj.Malloc_Bz_interp, BunchObj.Malloc_Br_interp, BunchObj.Malloc_Bf_interp,
                                 BunchObj.Malloc_Ez_interp, BunchObj.Malloc_Er_interp, BunchObj.Malloc_Ef_interp,
                                 BunchObj.Step_SurviveFlag,
                                 BunchObj.Malloc_RF_shift,
                                 BunchObj.SurvivedNum_Local, BunchObj.LostNum_Local,
                                 GlobalParameters)

                t_PostStep = t_PreStep + step_t

                # 更新 Post-steps (积分之后)
                BunchObj.UpdatePostSteps()

                # #  save coordinates with the given time interval
                StepDumpInfo.check_and_dump(t_PreStep, t_PostStep, BunchObj, data_type='Boris')

                # if np.mod(steps_i, 1000) == 0:

                # # update the pre step
                # t_PreStep = t_PostStep
                # if np.mod(steps_i, 1000) == 0:
                #     if comm.Get_rank() == 0:
                #         print(
                #             f"current step={steps_i:.0f}, "
                #             f"fi = {np.min(r_PreStep[BunchObj.survive_mask, 4]):.2f} rad, "
                #             f"turn = {np.min(r_PreStep[BunchObj.survive_mask, 4]) / np.pi / 2:.2f}, "
                #             f"remain particles: {BunchObj.SurvivedNum}. "
                #             f"current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}",
                #             flush=True)

                # update the pre step
                t_PreStep = t_PostStep
                if np.mod(steps_i, 1000) == 0:
                    BunchObj.UpdateParticleNumGlobal()
                    if comm.Get_rank() == 0:
                        # 筛选非 NaN 的数据
                        filtered_fi = BunchObj.LocalBunch[BunchObj.Step_SurviveFlag, 4]

                        if len(filtered_fi) > 0:  # 检查是否有有效数据
                            print(
                                f"current step={steps_i:.0f}, "
                                f"r ~= {BunchObj.LocalBunch[BunchObj.Step_SurviveFlag, 0][0]:.2f} m, "
                                f"turn ~= {filtered_fi[0] / np.pi / 2:.2f}, "
                                f"remain particles: {BunchObj.SurvivedNum}. "
                                f"tracking time: {t_PostStep*1e6:.2f} us,",
                                f"clock: {time.strftime('%m-%d %H:%M:%S', time.localtime())}",
                                flush=True
                            )
                            # print(BunchObj._uninject_head, BunchObj.uninject_num)
                        else:  # 如果没有有效数据
                            print(
                                f"current step={steps_i:.0f}, "
                                f"r ~= nan, "
                                f"turn ~= nan, "
                                f"remain particles: {BunchObj.SurvivedNum}. "
                                f"tracking time: {t_PostStep*1e6:.2f} us. ",
                                f"clock time: {time.strftime('%m-%d %H:%M:%S', time.localtime())}",
                                flush=True)

                # # # variable time steps
                # # check the step length every 1/2 turn, adapt the step length if needed
                # if np.mod(steps_i, int(steps_oneturn / 2)) == 0:
                #     dfidt_mean = BunchObj.GetMeanDfiDt()
                #     step_fi_ThisStep = dfidt_mean * step_t
                #     if np.abs(step_fi_ThisStep) > np.abs(max_step_fi) or np.abs(step_fi_ThisStep) < np.abs(min_step_fi):
                #         step_t = step_fi / dfidt_mean * TrackDirection

                # update loop index
                steps_i += 1

            comm.barrier()

            # Exit the loop
            if np.mod(steps_i, 1000) == 0:
                # stop conditions
                StopFlag_fi_local = copy.deepcopy(BunchObj.LocalBunch[:, 4])
                StopFlag_time_local = t_PreStep
                # 使用allgather将各线程汇总到一个进程中
                StopFlag_fi_gather = comm.allgather(StopFlag_fi_local)
                StopFlag_time_gather = comm.allgather(StopFlag_time_local)
                # 根进程拥有拼接后的数据
                StopFlag_fi_global = np.concatenate(StopFlag_fi_gather, axis=0)
                StopFlag_time_global = np.mean(StopFlag_time_gather)
                StopFlag_fi_survived_particles = StopFlag_fi_global[~np.isnan(StopFlag_fi_global)]

                if np.count_nonzero(np.isnan(StopFlag_fi_global)) > 0:
                    pass

                if np.abs(np.mean(StopFlag_fi_survived_particles)) > max_fi:
                    break
                if StopFlag_time_global > max_time:
                    break

        return 0


    def rk4_solve_vect(self, func, t_start, t_end, r_start, h, GlobalParameters, LocalParameters=False):
        steps = int((t_end - t_start) / h)
        t_points = np.linspace(t_start, t_end, steps)  # independent variables
        steps_real = t_points[1] - t_points[0]
        NSteps = len(t_points)
        t_points_AllSteps, r_points_AllSteps, rt_points_AllSteps = [], [], []
        r_points = copy.deepcopy(r_start)

        t_PreStep, i = t_points[0], 0

        while i < NSteps:
            # the pre step
            r_PreStep = copy.deepcopy(r_points)
            tr_PreStep = np.column_stack((np.ones_like(r_PreStep[:, 0]) * t_PreStep, r_PreStep))

            t_points_AllSteps.append(t_PreStep)
            r_points_AllSteps.append(r_PreStep)
            rt_points_AllSteps.append(tr_PreStep)

            dr_step = self.rk4_step(func, t_PreStep, r_PreStep, steps_real, GlobalParameters)
            r_PostStep = r_PreStep + dr_step
            t_PostStep = t_PreStep + steps_real

            # update loop index
            i += 1

            # update the pre step
            r_points = copy.deepcopy(r_PostStep)
            t_PreStep = t_PostStep

            # beam lost condition
            rstop_max = GlobalParameters.Bmap.r_max + GlobalParameters.Bmap.r_step * 10
            rstop_min = GlobalParameters.Bmap.r_min - GlobalParameters.Bmap.r_step * 10
            flag_delete_condition = (r_points[:, 0] < rstop_min) | (r_points[:, 0] > rstop_max)
            r_points = np.delete(r_points, flag_delete_condition, axis=0)

        return rt_points_AllSteps[-1]

    # @profile
    def rk4_solve(self, func, t_start, t_end, r_start, h, GlobalParameters):
        steps = int((t_end - t_start) / h)
        t_points = np.linspace(t_start, t_end, steps)  # independent variables
        steps_real = t_points[1] - t_points[0]
        t_length, r_length = len(t_points), len(r_start)  # steps
        r_points = np.zeros((t_length, r_length))  # coordinates for all steps
        r_points[0, :] = r_start  # set the first step

        for i in range(len(t_points) - 1):
            r_PreStep = r_points[i, :].copy()  # the pre step
            t_PreStep = t_points[i]

            # the post step
            dr_step = self.rk4_step(func, t_PreStep, r_PreStep, steps_real, GlobalParameters)
            r_PostStep = r_PreStep + dr_step

            r_points[i + 1, :] = r_PostStep

        return t_points, r_points


    def rk4_solve_vect_3DMatrix(self, func, t_start, t_end, r_start, h, GlobalParameters):

        steps = int((t_end - t_start) / h)
        t_points = np.linspace(t_start, t_end, steps)  # independent variables

        steps_real = t_points[1] - t_points[0]
        NSteps = len(t_points)
        # t_points_AllSteps, r_points_AllSteps, rt_points_AllSteps = [], [], []
        r_points = copy.deepcopy(r_start)
        InitParticleNum = np.size(r_points, 0)
        # Set the initial matrix
        tr_points_AllSteps = np.zeros((0, InitParticleNum, np.size(r_points, 1) + 1))

        t_PreStep, i = t_points[0], 0

        # Set the initial Matrix
        Bz_trajectory = np.zeros((InitParticleNum, 0))
        Br_trajectory = np.zeros((InitParticleNum, 0))
        Bf_trajectory = np.zeros((InitParticleNum, 0))
        fi_trajectory = np.zeros((1, 0))

        while i < NSteps:
            # the pre step
            r_PreStep = copy.deepcopy(r_points)
            tr_points_ThisStep = np.column_stack((np.ones_like(r_PreStep[:, 0]) * t_PreStep, r_PreStep))
            tr_points_AllSteps = np.concatenate((tr_points_AllSteps, tr_points_ThisStep[np.newaxis, :, :]), axis=0)

            dr_step, Bz_ThisStep, Br_ThisStep, Bfi_ThisStep = self.rk4_step_2(func, t_PreStep, r_PreStep, steps_real, GlobalParameters)

            Bz_trajectory = np.hstack((Bz_trajectory, Bz_ThisStep.reshape(-1, 1)))
            Br_trajectory = np.hstack((Br_trajectory, Br_ThisStep.reshape(-1, 1)))
            Bf_trajectory = np.hstack((Bf_trajectory, Bfi_ThisStep.reshape(-1, 1)))
            fi_trajectory = np.append(fi_trajectory, [[t_PreStep]], axis=1)

            r_PostStep = r_PreStep + dr_step
            t_PostStep = t_PreStep + steps_real

            # update loop index
            i += 1

            # update the pre step
            r_points = copy.deepcopy(r_PostStep)
            t_PreStep = t_PostStep

            # beam lost condition
            rstop_max = GlobalParameters.Bmap.r_max + GlobalParameters.Bmap.r_step * 10
            rstop_min = GlobalParameters.Bmap.r_min - GlobalParameters.Bmap.r_step * 10
            flag_delete_condition = (r_points[:, 0] < rstop_min) | (r_points[:, 0] > rstop_max)
            r_points = np.delete(r_points, flag_delete_condition, axis=0)

        return tr_points_AllSteps[-1, :, :], tr_points_AllSteps, Bz_trajectory, Br_trajectory, Bf_trajectory, fi_trajectory


    # def Boris_solve(self,t, x, GlobalParameters):
    #     """
    #     标准 Boris, 对应笛卡尔坐标(3D)的 (x, v).
    #     x, v: np.array([Nx, 3]) 分别是位置、速度
    #     E, B: np.array([Nx, 3]) 本时间步内粒子所在位置的电场、磁场
    #     func, t_PreStep, r_PreStep, step_t, GlobalParameters
    #     """
    #
    #     (r, rdot, z, zdot, fi, Etotal_J, t_inj, flag_inj,
    #      RF_Phase, Esc_r, Esc_z, Esc_fi, BID_local, PID_local, PID_global) = \
    #         (x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5], x[:, 6],
    #          x[:, 7], x[:, 8], x[:, 9], x[:, 10], x[:, 11], x[:, 12], x[:, 13], x[:, 14])
    #
    #     nParticles = x.shape[0]
    #     q = GlobalParameters.q
    #     c = GlobalParameters.c
    #     E0_J = GlobalParameters.E0
    #     Bmap = GlobalParameters.Bmap
    #     Emap = GlobalParameters.Emap
    #     NSectors = Bmap.Nsectors
    #     max_order = Bmap.max_order
    #
    #     gamma = Etotal_J / E0_J
    #     beta = np.sqrt(1 - 1 / (gamma ** 2))
    #     v = beta * c
    #     fidot = np.sqrt(v ** 2 - rdot ** 2 - zdot ** 2) / r
    #     fi_vect = fi % (np.pi * 2 / NSectors)
    #
    #     # 初始化 Bz、Br 和 Bfi 分量
    #     Bz0 = np.zeros_like(r)
    #     Br = np.zeros_like(r)
    #     Bfi = np.zeros_like(r)
    #
    #     # 动态添加高阶项，根据 max_order 自动展开
    #     for n in range(0, max_order + 1):
    #         # 获取每阶的拉普拉斯项
    #         Bz_coeff, _ = Bmap.Interpolation2DMap(r, fi_vect, n, 0)
    #         Bz0 += (z ** (2 * n)) * Bz_coeff  # Bz 只包含偶次项
    #
    #         Br_coeff, _ = Bmap.Interpolation2DMap(r, fi_vect, n, 2)
    #         Bfi_coeff, _ = Bmap.Interpolation2DMap(r, fi_vect, n, 1)
    #         Br += (z ** (2 * n + 1)) * Br_coeff * (2 * n + 1)  # Br 只包含奇次项
    #         Bfi += (z ** (2 * n + 1)) * Bfi_coeff * (2 * n + 1)  # Bfi 只包含奇次项
    #
    #     # 电场分量设为 0
    #     E_z, E_r, E_fi = Emap.Interpolation2D_EMap(r, fi_vect)
    #     sin_RFphase = np.sin(RF_Phase)
    #
    #     # 考虑空间电荷的电场分量
    #     Ez_tot = E_z * sin_RFphase + Esc_z
    #     Er_tot = E_r * sin_RFphase + Esc_r
    #     Efi_tot = E_fi * sin_RFphase + Esc_fi

    def Boris_push_cartesian(self,
                             x_cart, y_cart, z_cart,
                             vx_cart, vy_cart, vz_cart,
                             Ex_cart, Ey_cart, Ez_cart,
                             Bx_cart, By_cart, Bz_cart,
                             Inj_flag, Survive_flag,
                             dt, q, m, c):
        """
        Boris 推进方法。

        参数:
        ------------
        x_cart, y_cart, z_cart : 1D numpy.ndarray
            粒子的笛卡尔坐标 (x, y, z).
        vx_cart, vy_cart, vz_cart : 1D numpy.ndarray
            粒子的速度分量 (vx, vy, vz).
        Ex_cart, Ey_cart, Ez_cart : 1D numpy.ndarray
            粒子在各自位置处的电场分量 (Ex, Ey, Ez).
        Bx_cart, By_cart, Bz_cart : 1D numpy.ndarray
            粒子在各自位置处的磁场分量 (Bx, By, Bz).
        dt : float
            时间步长
        q : float
            带电粒子的电荷量
        m : float
            带电粒子的 **静止质量**
        c : float
            光速

        返回:
        ------------
        x_new, y_new, z_new : 1D numpy.ndarray
            更新后的粒子坐标.
        vx_new, vy_new, vz_new : 1D numpy.ndarray
            更新后的粒子速度分量.
        """

        # 计算速度模长 v^2
        v2 = vx_cart ** 2 + vy_cart ** 2 + vz_cart ** 2

        # 计算洛伦兹因子 γ = 1 / sqrt(1 - v^2 / c^2)
        gamma = 1.0 / np.sqrt(1 - v2 / c ** 2)

        # 计算相对论动量 p = γ m v
        px = gamma * m * vx_cart
        py = gamma * m * vy_cart
        pz = gamma * m * vz_cart

        # **Step 1: 半步电场更新 (p^- = p^n + qE dt/2)**
        coeff_e = 0.5 * dt * q
        px_minus = px + coeff_e * Ex_cart
        py_minus = py + coeff_e * Ey_cart
        pz_minus = pz + coeff_e * Ez_cart

        # **更新 γ (因为 p 发生了变化)**
        p2_minus = px_minus ** 2 + py_minus ** 2 + pz_minus ** 2
        gamma_minus = np.sqrt(1 + p2_minus / (m ** 2 * c ** 2))

        # **Step 2: 磁场旋转 (Boris 旋转)**
        # 计算磁旋向量 t = (q B dt) / (2 γ m)
        coeff_b = 0.5 * dt * q / (gamma_minus * m)
        t_x = coeff_b * Bx_cart
        t_y = coeff_b * By_cart
        t_z = coeff_b * Bz_cart

        # 计算 v' = v^- + (v^- × t)
        cross1_x = py_minus * t_z - pz_minus * t_y
        cross1_y = pz_minus * t_x - px_minus * t_z
        cross1_z = px_minus * t_y - py_minus * t_x

        px_prime = px_minus + cross1_x
        py_prime = py_minus + cross1_y
        pz_prime = pz_minus + cross1_z

        # 计算修正向量 s = 2 t / (1 + |t|^2)
        t_mag2 = t_x ** 2 + t_y ** 2 + t_z ** 2  # 计算 |t|^2
        s_scalar = 2.0 / (1.0 + t_mag2)  # 计算 2 / (1 + |t|^2)
        s_x = s_scalar * t_x
        s_y = s_scalar * t_y
        s_z = s_scalar * t_z

        # 计算 p^+ = p^- + (p' × s)
        cross2_x = py_prime * s_z - pz_prime * s_y
        cross2_y = pz_prime * s_x - px_prime * s_z
        cross2_z = px_prime * s_y - py_prime * s_x

        px_plus = px_minus + cross2_x
        py_plus = py_minus + cross2_y
        pz_plus = pz_minus + cross2_z

        # **Step 3: 再半步电场更新 (p_new = p^+ + qE dt/2)**
        px_new = px_plus + coeff_e * Ex_cart
        py_new = py_plus + coeff_e * Ey_cart
        pz_new = pz_plus + coeff_e * Ez_cart

        # 更新 γ (因为 p 发生了变化)
        p2_new = px_new ** 2 + py_new ** 2 + pz_new ** 2
        gamma_new = np.sqrt(1 + p2_new / (m ** 2 * c ** 2))

        # **计算最终速度 v = p / (γ m)**
        vx_new = px_new / (gamma_new * m)
        vy_new = py_new / (gamma_new * m)
        vz_new = pz_new / (gamma_new * m)

        # **Step 4: 更新位置**
        x_new = x_cart + vx_new * dt
        y_new = y_cart + vy_new * dt
        z_new = z_cart + vz_new * dt

        return x_new, y_new, z_new, vx_new, vy_new, vz_new


    # @profile
    def Boris_push_cartesian_new(self,
                             x_cart, y_cart, z_cart,
                             vx_cart, vy_cart, vz_cart,
                             Ex_cart, Ey_cart, Ez_cart,
                             Bx_cart, By_cart, Bz_cart,
                             Inj_flag, Survive_flag, r,
                             dt, q, m, c, Aperture_enable, Aperture_m,
                                 SurvivedNum_Local, LostNum_Local):
        """
        Boris 推进方法。

        参数:
        ------------
        x_cart, y_cart, z_cart : 1D numpy.ndarray
            粒子的笛卡尔坐标 (x, y, z).
        vx_cart, vy_cart, vz_cart : 1D numpy.ndarray
            粒子的速度分量 (vx, vy, vz).
        Ex_cart, Ey_cart, Ez_cart : 1D numpy.ndarray
            粒子在各自位置处的电场分量 (Ex, Ey, Ez).
        Bx_cart, By_cart, Bz_cart : 1D numpy.ndarray
            粒子在各自位置处的磁场分量 (Bx, By, Bz).
        dt : float
            时间步长
        q : float
            带电粒子的电荷量
        m : float
            带电粒子的静止质量
        c : float
            光速

        返回:
        ------------
        x_new, y_new, z_new : 1D numpy.ndarray
            更新后的粒子坐标.
        vx_new, vy_new, vz_new : 1D numpy.ndarray
            更新后的粒子速度分量.
        """
        x_new = np.empty_like(x_cart)
        y_new = np.empty_like(x_cart)
        z_new = np.empty_like(x_cart)
        vx_new = np.empty_like(x_cart)
        vy_new = np.empty_like(x_cart)
        vz_new = np.empty_like(x_cart)

        boris_push_cartesian_njit(x_cart, y_cart, z_cart,
                                  vx_cart, vy_cart, vz_cart,
                                  Ex_cart, Ey_cart, Ez_cart,
                                  Bx_cart, By_cart, Bz_cart,
                                  Inj_flag, Survive_flag, r,
                                  dt, q, m, c,
                                  x_new, y_new, z_new,
                                  vx_new, vy_new, vz_new, Aperture_enable, Aperture_m,
                                  SurvivedNum_Local, LostNum_Local)

        return x_new, y_new, z_new, vx_new, vy_new, vz_new

    def Boris_push_cylindrical_new(self,
            r, z, fi, rdot, zdot, fidot,
            E_r, E_z, E_fi, Esc_r, Esc_z, Esc_fi,
            B_r, B_z, B_fi, RF_Phase,
            Inj_flag, Survive_flag,
            dt, q, m, c, mat_in,
            r_new, z_new, fi_new,
            rdot_new, zdot_new, fidot_new,
            apply_RF=False, apply_space_charge=False):

        # r_new = np.empty_like(r)
        # fi_new = np.empty_like(fi)
        # z_new = np.empty_like(z)
        # rdot_new = np.empty_like(rdot)
        # fidot_new = np.empty_like(fidot)
        # zdot_new = np.empty_like(zdot)

        boris_push_cylindrical_speed_njit(
            r, z, fi, rdot, zdot, fidot,
            E_r, E_z, E_fi, Esc_r, Esc_z, Esc_fi,
            B_r, B_z, B_fi, RF_Phase,
            Inj_flag, Survive_flag,
            dt, q, m, c, mat_in,
            r_new, z_new, fi_new,
            rdot_new, zdot_new, fidot_new,
            apply_RF=apply_RF, apply_space_charge=apply_space_charge)

        return r_new, fi_new, z_new, rdot_new, fidot_new, zdot_new


    # @profile
    def Boris_solve(self, t, mat_in, dt,
                    Malloc_x_cart,Malloc_y_cart,Malloc_z_cart,
                    Malloc_vx_cart,Malloc_vy_cart,Malloc_vz_cart,
                    Malloc_Ex_cart,Malloc_Ey_cart,Malloc_Ez_cart,
                    Malloc_Bx_cart,Malloc_By_cart,Malloc_Bz_cart,
                    Malloc_sinRF, fi_vect,
                    Bz0, Br, Bfi,
                    E_z, E_r, E_fi,
                    Malloc_Step_SurviveFlag,
                    Malloc_RF_shift,
                    SurvivedNum_Local, LostNum_Local,
                    GlobalParameters):
        """

        """
        q = GlobalParameters.q
        c = GlobalParameters.c
        Bmap = GlobalParameters.Bmap
        Emap = GlobalParameters.Emap
        NSectors = Bmap.Nsectors
        max_order = Bmap.max_order
        m0 = GlobalParameters.m0
        Aperture_enable = GlobalParameters.Aperture_enable
        Aperture_m = GlobalParameters.Aperture_m
        BHarmonics = GlobalParameters.BHarmonics

        # 1) 解出列

        r = mat_in[:, 0]
        rdot = mat_in[:, 1]
        z = mat_in[:, 2]
        zdot = mat_in[:, 3]
        fi = mat_in[:, 4]
        fidot = mat_in[:, 5]
        Inj_flag = mat_in[:, 7]
        Survive_flag = mat_in[:, 8]
        RF_Phase = mat_in[:, 9]
        Esc_r = mat_in[:, 10]
        Esc_z = mat_in[:, 11]
        Esc_fi = mat_in[:, 12]

        # fi_vect = fi % (np.pi * 2 / NSectors)
        fast_mod_parallel(fi, (np.pi * 2 / NSectors), fi_vect)

        # 初始化 Bz、Br 和 Bfi 分量
        # Bz0 = np.zeros_like(r)
        # Br = Bz0 * 0.0
        # Bfi = Bz0 * 0.0
        # 0阶采用理论模型
        # Bz_coef0, Br_coef0, Bfi_coef0 = Bmap.GetBfields(r, fi_vect, SearchMod=False)
        # Bz_coef0, Br_coef0, Bfi_coef0 = Bmap.GetBfields(r, fi_vect, z, Inj_flag, Survive_flag, SearchMod=False)
        # Bz0 += Bz_coef0
        # Br += Br_coef0 * z
        # Bfi += Bfi_coef0 * z
        Bmap.GetBfields(r, fi_vect, z, Inj_flag, Survive_flag, Bz0, Br, Bfi, SearchMod=False)
        if max_order >=1:
            Bmap.AddHigherOrderBmaps(r, fi_vect, z, Bz0, Br, Bfi, max_order)

        # 磁场误差谐波
        BHarmonics.AddBHarmonics_to_Bmap_njit(r, fi, z, Bz0, Br, Bfi)

        # 电场分量
        dRF_phase, v0_volt = Emap.Interpolation1D_freq_curve(t)
        Emap.Interpolation2D_EMap(r, fi, z, E_r, E_z, E_fi, Malloc_RF_shift, v0_volt=v0_volt)
        dRF_phase = dRF_phase * np.pi * 2 * dt


        # 6) 柱坐标转cartesian
        cylindrical_to_cartesian(r, rdot, z, zdot, fi, fidot,
                                 E_r, E_z, E_fi, Esc_r, Esc_z, Esc_fi,
                                 Br, Bz0, Bfi, RF_Phase,
                                 Malloc_x_cart, Malloc_y_cart, Malloc_z_cart,
                                 Malloc_vx_cart, Malloc_vy_cart, Malloc_vz_cart,
                                 Malloc_Ex_cart, Malloc_Ey_cart, Malloc_Ez_cart,
                                 Malloc_Bx_cart, Malloc_By_cart, Malloc_Bz_cart,
                                 Malloc_RF_shift, Emap.rf_shift,
                                 apply_RF=Emap.EnableFlag)

        # 5) 使用 Boris 推进更新 (x_cart, v_cart)
        x_new, y_new, z_new, vx_new, vy_new, vz_new = self.Boris_push_cartesian_new(
            Malloc_x_cart, Malloc_y_cart, Malloc_z_cart,
            Malloc_vx_cart, Malloc_vy_cart, Malloc_vz_cart,
            Malloc_Ex_cart, Malloc_Ey_cart, Malloc_Ez_cart,
            Malloc_Bx_cart, Malloc_By_cart, Malloc_Bz_cart,
            Inj_flag, Survive_flag, r,
            dt, q, m0, c, Aperture_enable, Aperture_m, SurvivedNum_Local, LostNum_Local)

        # 6) 将结果转回柱坐标:
        convert_cartesian_to_cylindrical(x_new, y_new, z_new,
                                         vx_new, vy_new, vz_new,
                                         Malloc_x_cart, Malloc_y_cart, fi,
                                         mat_in, dRF_phase)

        # if Survive_flag[0]:
        #     with open("Bzrf_test.txt", "a+") as fid:
        #         v2 = vx_new[0]*vx_new[0]+ vy_new[0]*vy_new[0]+ vz_new[0]*vz_new[0]
        #         gamma = 1.0 / np.sqrt(1 - v2 / (c * c))
        #         # beta = np.sqrt(v2)/c
        #         np.savetxt(fid, np.array([[r[0], fi[0], z[0], Bz0[0], Br[0], Bfi[0],gamma]]))


        return mat_in

    # # @profile
    # def Boris_solve_cylindrical(self, t, mat_in, dt,
    #                             Bz0, Br, Bfi,
    #                             E_z, E_r, E_fi,
    #                             Malloc_r_cart, Malloc_z_cart, Malloc_fi_cart,
    #                             Malloc_vr_cart, Malloc_vz_cart, Malloc_vfi_cart,
    #                             fi_vect,
    #                             GlobalParameters):
    #     """
    #
    #     """
    #     q = GlobalParameters.q
    #     c = GlobalParameters.c
    #     Bmap = GlobalParameters.Bmap
    #     Emap = GlobalParameters.Emap
    #     NSectors = Bmap.Nsectors
    #     max_order = Bmap.max_order
    #     m0 = GlobalParameters.m0
    #
    #     # 1) 解出列
    #     r = mat_in[:, 0]
    #     rdot = mat_in[:, 1]
    #     z = mat_in[:, 2]
    #     zdot = mat_in[:, 3]
    #     fi = mat_in[:, 4]
    #     fidot = mat_in[:, 5]
    #     Inj_flag = mat_in[:, 7]
    #     Survive_flag = mat_in[:, 8]
    #     RF_Phase = mat_in[:, 9]
    #     Esc_r = mat_in[:, 10]
    #     Esc_z = mat_in[:, 11]
    #     Esc_fi = mat_in[:, 12]
    #
    #     # fi_vect = fi % (np.pi * 2 / NSectors)
    #     fast_mod_parallel(fi, (np.pi * 2 / NSectors), fi_vect)
    #
    #     # Bz、Br 和 Bfi 分量
    #     Bmap.GetBfields(r, fi_vect, z, Inj_flag, Survive_flag, Bz0, Br, Bfi, SearchMod=False)
    #
    #     # 高阶采用数值模型, 动态添加高阶项, 根据 max_order 自动展开
    #     for n in range(0, max_order):
    #         # 获取每阶的拉普拉斯项
    #
    #         Bz_coeff, _ = Bmap.Interpolation2DMap(r, fi_vect, n, 0)
    #         Br_coeff, _ = Bmap.Interpolation2DMap(r, fi_vect, n, 2)
    #         Bfi_coeff, _ = Bmap.Interpolation2DMap(r, fi_vect, n, 1)
    #
    #         Bz0 += (z ** (2 * n)) * Bz_coeff  # Bz 只包含偶次项
    #         Br += (z ** (2 * n + 1)) * Br_coeff  # Br 只包含奇次项
    #         Bfi += (z ** (2 * n + 1)) * Bfi_coeff  # Bfi 只包含奇次项
    #
    #     # 电场分量
    #     Emap.Interpolation2D_EMap(r, fi_vect, E_z, E_r, E_fi)
    #     with open("11.txt", 'a') as fid:
    #         np.savetxt(fid, np.array([[r[0], fi[0], z[0], zdot[0], fidot[0], rdot[0]]]))
    #
    #     # 5) 使用 Boris 推进更新 (x_cart, v_cart)
    #     self.Boris_push_cylindrical_new(
    #         r, z, fi, rdot, zdot, fidot,
    #         E_r, E_z, E_fi, Esc_r, Esc_z, Esc_fi,
    #         Br, Bz0, Bfi, RF_Phase,
    #         Inj_flag, Survive_flag,
    #         dt, q, m0, c, mat_in,
    #         Malloc_r_cart, Malloc_z_cart, Malloc_fi_cart,
    #         Malloc_vr_cart, Malloc_vz_cart, Malloc_vfi_cart,
    #         apply_RF=False, apply_space_charge=False
    #     )
    #
    #     return mat_in


class FFAG_MPI:
    def __init__(self):
        pass

    def DivideVariables(self, xGlobal):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        TotalNum = len(xGlobal)
        MinNumPerCPU = math.floor(TotalNum / size)
        ModNum = TotalNum - MinNumPerCPU * size
        xLocalIndex = list(range(rank * MinNumPerCPU, (rank + 1) * MinNumPerCPU))
        if rank < ModNum:
            xLocalIndex.append(MinNumPerCPU * size + rank)
        xLocal = [xGlobal[i] for i in xLocalIndex]
        # print(f"Rank {MPI.COMM_WORLD.Get_rank()}: xLocal = {xLocal}")
        return np.array(xLocal, dtype=int)


class FFAG_SearchSEO:
    def __init__(self, GlobalParas):
        self.GlobalParas = GlobalParas

    def ObjectiveFuncForSEO(self, x, ObjFuncPara):
        Ek_value = ObjFuncPara.Ek_value
        t_end = ObjFuncPara.t_end
        h = ObjFuncPara.h
        r0, pr0 = x[0], x[1]
        t_points, r_points = self.TrackAParticle(r0, pr0, Ek_value, t_end, h)
        ee = ((r_points[0, 0] - r_points[-1, 0]) / 1e-6) ** 2 + ((r_points[0, 1] - r_points[-1, 1]) / 1e-6) ** 2
        return ee

    # @profile
    def TrackAParticle(self, r0, pr0, Ek_value, t_end, h):

        rk = FFAG_RungeKutta()
        t_start = 0
        P_start = FFAG_ConversionTools().Ek2P(Ek_value)
        ini_no_offset = np.array([r0, pr0, 0, 0, P_start, 0])  # r,pr,z,pz,P,index
        tr_points = rk.rk4_solve(FunctionForSEO, t_start, t_end, ini_no_offset, h,
                                          self.GlobalParas)
        return tr_points

    # @profile
    def TrackBunchSEO(self, r0, p0, Ek_value, t_end, h, verbose):
        t_start = 0
        rk = FFAG_RungeKutta()

        P_start = FFAG_ConversionTools().Ek2P(Ek_value)
        # r,pr,z,pz,P,index
        delta_r, delta_p = r0 * 0.0001, 0.001
        for k in range(25):
            Ini_start = np.array([[r0, p0, 0, 0, P_start, 0],
                                  [r0 + delta_r, p0, 0, 0, P_start, 1],
                                  [r0, p0 + delta_p, 0, 0, P_start, 2]])

            tr_points = rk.rk4_solve_vect(FunctionForSEOVect, t_start, t_end, Ini_start, h, self.GlobalParas)
            r0f, p0f = tr_points[0, 1], tr_points[0, 2]
            r1f, p1f = tr_points[1, 1], tr_points[1, 2]
            r2f, p2f = tr_points[2, 1], tr_points[2, 2]

            a11 = (r1f - r0f) / delta_r
            a12 = (r2f - r0f) / delta_p
            a21 = (p1f - p0f) / delta_r
            a22 = (p2f - p0f) / delta_p
            a11_prime, a22_prime = a11 - 1, a22 - 1
            DetermineA = a11_prime * a22_prime - a12 * a21

            if verbose:
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print('In accurate calculation, (r0f - r0)/r0 = ', (r0f - r0) / r0)
            if np.abs((r0f - r0) / r0) < 1e-8:
                break

            re = r0 + a22_prime / DetermineA * (r0 - r0f) - a12 / DetermineA * (p0 - p0f)
            pe = p0 + a11_prime / DetermineA * (p0 - p0f) - a21 / DetermineA * (r0 - r0f)

            r0 = re
            p0 = pe

        return r0, p0


    # def SearchSEOUsingInitialEkVect(self, Ek_value, h=0.001, delta_r0_ini=0.0, verbose=False):
    #     """
    #     search for the SEO for a given Ek value
    #     return: initial condition r0, pr0
    #     """
    #     self.GlobalParas.TempVariable = Ek_value
    #     # Find analytical initial value r0, pr0
    #     r0_analytical, _ = FFAG_Algorithm().BiSection(self.GlobalParas.Bmap.r_min, self.GlobalParas.Bmap.r_max,
    #                                                   func_get_R_B0, self.GlobalParas)
    #     r0 = r0_analytical + delta_r0_ini
    #     pr0 = 0.0
    #
    #     t_end = np.pi * 2 / self.GlobalParas.Bmap.Nsectors
    #     # print("Nsectors=", self.GlobalParas.Bmap.Nsectors)
    #     # Find approximate initial value r0, pr0   30
    #     # for i_nSector in np.arange(0, self.GlobalParas.Bmap.Nsectors):
    #     for i_nSector in [0, ]:
    #         t_end_i = t_end * (i_nSector + 1)
    #
    #         for k in np.arange(0, 100):
    #             t_points, r_points = self.TrackAParticle(r0, pr0, Ek_value, t_end_i, h)
    #
    #             r_of_fi, pr_of_fi = r_points[:, 0], r_points[:, 1]
    #             r0, pr0 = (r_of_fi[0] + r_of_fi[-1]) / 2, (pr_of_fi[0] + pr_of_fi[-1]) / 2
    #             if verbose:
    #                 if MPI.COMM_WORLD.Get_rank() == 0:
    #                     print('i=', Ek_value, ' k=', k, 'ee=',
    #                           np.sqrt(
    #                               ((r_of_fi[0] - r_of_fi[-1]) / 1e-6) ** 2 + ((pr_of_fi[0] - pr_of_fi[-1]) / 1e-6) ** 2),
    #                           'r_start=', r_of_fi[0], 'r_end=', r_of_fi[-1])
    #
    #             if np.sqrt(((r_of_fi[0] - r_of_fi[-1]) / 1e-6) ** 2 + ((pr_of_fi[0] - pr_of_fi[-1]) / 1e-6) ** 2) < 1:
    #                 break
    #
    #     r0, pr0 = self.TrackBunchSEO(r0, pr0, Ek_value, t_end, h, verbose)
    #
    #     return r0, pr0, r0_analytical

    # @profile
    def SearchSEOUsingInitialEkVect(self, Ek_value, h=0.001, delta_r0_ini=0.0, verbose=False):
        """
        search for the SEO for a given Ek value
        return: initial condition r0, pr0
        """

        self.GlobalParas.TempVariable = Ek_value
        # Find analytical initial value r0, pr0
        r0_analytical, _ = FFAG_Algorithm().BiSection(self.GlobalParas.Bmap.r_min, self.GlobalParas.Bmap.r_max,
                                                      func_get_R_B0, self.GlobalParas)
        r0 = r0_analytical + delta_r0_ini
        BMapHarmonics = self.GlobalParas.BHarmonics
        EnableBHarmonics = BMapHarmonics.enable_flag

        t_end_i = np.pi * 2 / self.GlobalParas.Bmap.Nsectors
        t_end_full = np.pi * 2

        # 定义调整步长的列表
        pr0_list = [0.00, 0.02, -0.02, 0.04, -0.04, 0.06, -0.06, 0.08, -0.08, 0.10, -0.10, 0.12, -0.12, 0.14, -0.14]

        # 标志：表示是否已经找到符合条件的 r0 和 pr0
        found = False

        # Find approximate initial value r0, pr0
        for pr0 in pr0_list:

            # 重置初始值 r0 和 pr0
            r0 = r0_analytical + delta_r0_ini

            for k in np.arange(0, 200):
                # 跟踪粒子轨迹
                t_points, r_points = self.TrackAParticle(r0, pr0, Ek_value, t_end_i, h)

                # 提取位置和动量
                r_of_fi, pr_of_fi = r_points[:, 0], r_points[:, 1]

                # 更新初始值 r0 和 pr0
                r0, pr0 = (r_of_fi[0] + r_of_fi[-1]) / 2, (pr_of_fi[0] + pr_of_fi[-1]) / 2

                # 打印调试信息
                if verbose:
                    if MPI.COMM_WORLD.Get_rank() == 0:
                        print('i=', Ek_value, ' k=', k, 'ee=',
                              np.sqrt(
                                  ((r_of_fi[0] - r_of_fi[-1]) / 1e-6) ** 2 + (
                                          (pr_of_fi[0] - pr_of_fi[-1]) / 1e-6) ** 2),
                              'r_start=', r_of_fi[0], 'r_end=', r_of_fi[-1])

                # 判断是否出现NaN情况
                if np.isnan(r_of_fi[-1]) or np.isnan(pr_of_fi[-1]):
                    # 如果是NaN，退出当前 100 次迭代
                    print(f"NaN detected at iteration {k}. Adjusting pr0 and retrying.")

                    # 恢复到最初的 r0 和 delta_r0_ini
                    r0 = r0_analytical + delta_r0_ini

                    # 跳过当前的 100 次循环，继续尝试下一个 pr0
                    break  # 跳出当前 100 次迭代，继续尝试下一个 pr0

                # 终止条件：当位置和动量的变化量小于某个阈值时跳出循环
                if np.abs(r_of_fi[0] - r_of_fi[-1]) < 1e-5 and np.abs(pr_of_fi[0] - pr_of_fi[-1]) < 1e-5:
                    found = True  # 设置标志为 True，表示满足条件
                    break  # 跳出当前 200 次迭代

            if found:
                # 如果满足条件，跳出 pr0_list 循环
                break  # 跳出 pr0_list 循环

        # 终止后的结果处理
        if found:
            if EnableBHarmonics:
                r0, pr0 = self.TrackBunchSEO(r0, pr0, Ek_value, t_end_full, h, verbose)
            else:
                r0, pr0 = self.TrackBunchSEO(r0, pr0, Ek_value, t_end_i, h, verbose)
        else:
            raise RuntimeError(f"No close orbit solution found for {Ek_value}MeV.")


        return r0, pr0, r0_analytical

    def GetQrQz(self, r0, pr0, Ek_value, h):
        t_start = 0
        t_end = np.pi * 2 / self.GlobalParas.Bmap.Nsectors
        z0, pz0 = 0.0, 0.0
        delta_rr, delta_pp = 1e-6, 1e-6
        delta_zz, delta_pz = 1e-7, 1e-7

        P_start = FFAG_ConversionTools().Ek2P(Ek_value)
        ini_no_offset = np.array([r0, pr0, z0, pz0, P_start, 0])
        ini_offset_R = np.array([r0 + delta_rr, pr0, z0, pz0, P_start, 0])
        ini_offset_Pr = np.array([r0, pr0 + delta_pp, z0, pz0, P_start, 0])
        ini_offset_Z = np.array([r0, pr0, z0 + delta_zz, pz0, P_start, 0])
        ini_offset_PZ = np.array([r0, pr0, z0, pz0 + delta_pz, P_start, 0])
        rk = FFAG_RungeKutta()
        # r1, pr1, z1, pz1
        t_points, r_points = rk.rk4_solve(FunctionForSEO, t_start, t_end, ini_no_offset, h,
                                          self.GlobalParas)
        r1, p1, z1, pz1 = r_points[-1, 0], r_points[-1, 1], 0.0, 0.0
        OrbitFreq, Perimeter = self.GetFreq(t_points, r_points[:, 0], r_points[:, 1], Ek_value)
        Mean_R = np.mean(r_points[:, 0])

        _, r_points = rk.rk4_solve(FunctionForSEO, t_start, t_end, ini_offset_R, h,
                                   self.GlobalParas)
        r2, p2 = r_points[-1, 0], r_points[-1, 1]

        _, r_points = rk.rk4_solve(FunctionForSEO, t_start, t_end, ini_offset_Pr, h,
                                   self.GlobalParas)
        r3, p3 = r_points[-1, 0], r_points[-1, 1]

        _, r_points = rk.rk4_solve(FunctionForSEO, t_start, t_end, ini_offset_Z, h,
                                   self.GlobalParas)
        z2, pz2 = r_points[-1, 2], r_points[-1, 3]

        _, r_points = rk.rk4_solve(FunctionForSEO, t_start, t_end, ini_offset_PZ, h,
                                   self.GlobalParas)
        z3, pz3 = r_points[-1, 2], r_points[-1, 3]

        a11 = (r2 - r1) / delta_rr
        a22 = (p3 - p1) / delta_pp
        b11 = (z2 - z1) / delta_zz
        b22 = (pz3 - pz1) / delta_pz
        # print(f"b11={b11}, b22={b22}")

        Qr_float2 = np.arccos((a11 + a22) / 2) / (2 * np.pi) * self.GlobalParas.Bmap.Nsectors
        Qz_float2 = np.arccos((b11 + b22) / 2) / (2 * np.pi) * self.GlobalParas.Bmap.Nsectors

        return Qr_float2, Qz_float2, OrbitFreq, Mean_R, Perimeter

    def GetQrQz_(self, r0, pr0, Ek_value, h):
        EnableBHarmonics = self.GlobalParas.BHarmonics.enable_flag

        t_start = 0
        t_end = np.pi * 2
        z0, pz0 = 0.0, 0.0
        delta_rr, delta_pp = 1e-6, 1e-6
        delta_zz, delta_pz = 1e-7, 1e-7

        P_start = FFAG_ConversionTools().Ek2P(Ek_value)
        ini_no_offset = np.array([r0, pr0, z0, pz0, P_start, 0])
        ini_offset_R = np.array([r0 + delta_rr, pr0, z0, pz0, P_start, 0])
        ini_offset_Pr = np.array([r0, pr0 + delta_pp, z0, pz0, P_start, 0])
        ini_offset_Z = np.array([r0, pr0, z0 + delta_zz, pz0, P_start, 0])
        ini_offset_PZ = np.array([r0, pr0, z0, pz0 + delta_pz, P_start, 0])
        rk = FFAG_RungeKutta()
        # r1, pr1, z1, pz1
        t_points, r_points = rk.rk4_solve(FunctionForSEO, t_start, t_end, ini_no_offset, h,
                                          self.GlobalParas)
        r1, p1, z1, pz1 = r_points[-1, 0], r_points[-1, 1], 0.0, 0.0
        OrbitFreq, Perimeter = self.GetFreq(t_points, r_points[:, 0], r_points[:, 1], Ek_value)
        Mean_R = np.mean(r_points[:, 0])

        _, r_points = rk.rk4_solve(FunctionForSEO, t_start, t_end, ini_offset_R, h,
                                   self.GlobalParas)
        r2, p2 = r_points[-1, 0], r_points[-1, 1]

        _, r_points = rk.rk4_solve(FunctionForSEO, t_start, t_end, ini_offset_Pr, h,
                                   self.GlobalParas)
        r3, p3 = r_points[-1, 0], r_points[-1, 1]

        _, r_points = rk.rk4_solve(FunctionForSEO, t_start, t_end, ini_offset_Z, h,
                                   self.GlobalParas)
        z2, pz2 = r_points[-1, 2], r_points[-1, 3]

        _, r_points = rk.rk4_solve(FunctionForSEO, t_start, t_end, ini_offset_PZ, h,
                                   self.GlobalParas)
        z3, pz3 = r_points[-1, 2], r_points[-1, 3]

        a11 = (r2 - r1) / delta_rr
        a22 = (p3 - p1) / delta_pp
        b11 = (z2 - z1) / delta_zz
        b22 = (pz3 - pz1) / delta_pz
        # print(f"b11={b11}, b22={b22}")

        Qr_float2 = np.arccos((a11 + a22) / 2) / (2 * np.pi)
        Qz_float2 = np.arccos((b11 + b22) / 2) / (2 * np.pi)

        # Qr_float2 = np.arccos((a11 + a22) / 2) / (2 * np.pi)
        # Qz_float2 = np.arccos((b11 + b22) / 2) / (2 * np.pi)
        return Qr_float2, Qz_float2

    def GetFreq(self, fi_SEO, r_SEO, pr_SEO, Ek_value):
        """
        Return the orbital frequency for the given Ek value
        """
        rp_SEO = pr_SEO / (np.sqrt(1 - pr_SEO ** 2)) * r_SEO
        DiffArcLength = np.sqrt(r_SEO ** 2 + rp_SEO ** 2)
        DiffArcLengthDfi = DiffArcLength * (fi_SEO[2] - fi_SEO[1])
        EnableBHarmonics = self.GlobalParas.BHarmonics.enable_flag

        Perimeter = sum(DiffArcLengthDfi[:-1]) * self.GlobalParas.Bmap.Nsectors
        v = FFAG_ConversionTools().Ek2v(Ek_value)
        Period = Perimeter / v
        frequency = 1 / Period
        return frequency, Perimeter

    # @profile
    def SearchSEOsControllerVect(self, EkRange, SavedataFlag=True):
        EkIndex = np.arange(0, len(EkRange))
        EkIndexLocal = FFAG_MPI().DivideVariables(EkIndex)
        EkNumLocal, EkNumTotal = len(EkIndexLocal), len(EkIndex)

        SEOdataLocal = np.zeros((EkNumLocal, 17))  # Ek_index, Ek_value, r0, pr0, r_end, pr_end, Qr, Qz, freq, MeanR
        SEO_fir_Local = []
        SEO_fipr_Local = []
        SEO_fi_Local = []
        h = 0.001

        index = 0
        delta_r0_ini = 0.0
        for Ek_index in EkIndexLocal:
            Ek_value = EkRange[Ek_index]
            r0, pr0, r0_analytical = self.SearchSEOUsingInitialEkVect(Ek_value, h,
                                                                      delta_r0_ini, verbose=False)  # the accurate initial value

            t_points, r_points = self.TrackAParticle(r0, pr0, Ek_value, np.pi * 2, 0.001)
            r_of_fi, pr_of_fi = r_points[:, 0], r_points[:, 1]

            Ek_fi_label = np.hstack((Ek_value, t_points))
            Ek_r_of_fi = np.hstack((Ek_value, r_of_fi))
            Ek_pr_of_fi = np.hstack((Ek_value, pr_of_fi))

            SEO_fi_Local.append(Ek_fi_label)
            SEO_fipr_Local.append(Ek_pr_of_fi)
            SEO_fir_Local.append(Ek_r_of_fi)

            # 替换线性插值器，改用 SciPy 的 interp1d
            rfi_interp = interp1d(t_points, r_of_fi, kind='linear', fill_value="extrapolate")
            prfi_interp = interp1d(t_points, pr_of_fi, kind='linear', fill_value="extrapolate")

            # 使用插值器计算特定角度下的 r 和 pr 值
            r90, pr90 = rfi_interp(np.deg2rad(90.0)), prfi_interp(np.deg2rad(90.0))
            r180, pr180 = rfi_interp(np.deg2rad(180.0)), prfi_interp(np.deg2rad(180.0))
            r270, pr270 = rfi_interp(np.deg2rad(270.0)), prfi_interp(np.deg2rad(270.0))

            Qr, Qz, OrbitFreq, MeanR, Perimeter = self.GetQrQz(r0, pr0, Ek_value, h)
            print(f"Closed orbit found for {Ek_value}MeV, Qr={Qr:.3f}, Qz={Qz:.3f}, r_mean = {MeanR:.3f}m, "
                  f"Orbital Frequency={OrbitFreq/1e6:.3f}MHz", flush=True)

            SEOdataLocal[index, :] = np.array(
                (Ek_index, Ek_value, Qr, Qz, OrbitFreq, MeanR,
                 r0, pr0, Perimeter, r_points[-1, 0], r_points[-1, 1],
                 r90, pr90, r180, pr180, r270, pr270,))

            index += 1

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        SEOdataGlobal = comm.allgather(SEOdataLocal)
        SEO_fir_Global = comm.allgather(SEO_fir_Local)
        SEO_fipr_Global = comm.allgather(SEO_fipr_Local)
        SEO_fi_Global = comm.allgather(SEO_fi_Local)

        SEO_foldname = None
        # 根进程拥有拼接后的数据
        if rank == 0:
            SEOdataGlobal_arr = np.concatenate(SEOdataGlobal, axis=0)

            SEO_fir_Global_arr = np.concatenate(SEO_fir_Global, axis=0)
            SEO_fipr_Global_arr = np.concatenate(SEO_fipr_Global, axis=0)
            SEO_fi_Global_arr = np.concatenate(SEO_fi_Global, axis=0)
            # 获取按第一列排序的索引
            sorted_indices = np.argsort(SEOdataGlobal_arr[:, 0])
            sorted_indices_rpr = np.argsort(SEO_fir_Global_arr[:, 0])
            # 使用索引对数组进行排序
            SEOdataGlobal_arr_sorted = SEOdataGlobal_arr[sorted_indices]

            SEO_fir_Global_arr_sorted = SEO_fir_Global_arr[sorted_indices_rpr]
            SEO_fipr_Global_arr_sorted = SEO_fipr_Global_arr[sorted_indices_rpr]
            SEO_fi_Global_arr_sorted = SEO_fi_Global_arr[sorted_indices_rpr]

            ExcelTitles = ["Ek_index", "Ek_value(MeV)", "Qr", "Qz", "OrbitFreq(Hz)", "MeanR(m)",
                           "r0(m)", "pr0", "Perimeter(m)", "r360(m)", "pr360",
                           "r90(m)", "pr90", "r180(m)", "pr180", "r270(m)", "pr270"]

            if SavedataFlag:

                BmapFoldname = self.GlobalParas.Bmap.BmapFoldName
                SEO_foldname = os.path.join(BmapFoldname, self.GlobalParas.SEO_SaveFold)
                # if self.GlobalParas.BHarmonics.enable_flag:
                #     SEO_foldname = os.path.join(BmapFoldname, "resultsSEO_withBError")
                # else:
                #     SEO_foldname = os.path.join(BmapFoldname, "resultsSEO")
                os.makedirs(SEO_foldname, exist_ok=True)

                print(f"SEO information will be writen in folder '{SEO_foldname}'.", flush=True)

                SEO_foldfilename = SEO_foldname + '/SEO_ini.txt'
                SEO_fipr_filename = SEO_foldname + '/SEO_pr.txt'
                SEO_fir_filename = SEO_foldname + '/SEO_r.txt'

                # 合并表头和数据矩阵为一个整体的列表
                SEOdataGlobal_arr_sorted_float = SEOdataGlobal_arr_sorted.astype(float)
                SEOdataGlobal_arr_sorted_float[:, 2:6] = np.round(SEOdataGlobal_arr_sorted_float[:, 2:6], 3)
                SEOdataGlobal_arr_sorted_float[:, 6:] = np.round(SEOdataGlobal_arr_sorted_float[:, 6:], 8)

                combined_data = [ExcelTitles] + SEOdataGlobal_arr_sorted_float.tolist()

                # 计算每列的最大宽度，包括表头的每列
                max_widths = [max(len(str(row[i])) for row in combined_data) for i in range(len(combined_data[0]))]

                # 格式化并左对齐每列，包括表头，并在相邻两列之间插入两列空格
                formatted_data = []
                for row in combined_data:
                    formatted_row = ["  ".join([str(value).ljust(width) for value, width in zip(row, max_widths)])]
                    formatted_data.extend(formatted_row)

                # 在首行添加Bmap相关信息
                TheFirstLine = "Bmap foldname=" + self.GlobalParas.Bmap.BmapFoldName + ", step h = %.6f rad" % (h,)
                formatted_data.insert(0, TheFirstLine)

                # 将格式化后的数据写入文本文件
                with open(SEO_foldfilename, "w") as file:
                    file.write("\n".join(formatted_data))

                SEO_fir_Global_matrix = np.row_stack((SEO_fi_Global_arr_sorted[0, :], SEO_fir_Global_arr_sorted))
                SEO_fipr_Global_matrix = np.row_stack((SEO_fi_Global_arr_sorted[0, :], SEO_fipr_Global_arr_sorted))

                with open(SEO_fir_filename, "w") as fid:
                    fid.write("unitEk=MeV, unitFi=deg, unitR=m\n")
                with open(SEO_fir_filename, "a") as fid_fir:
                    np.savetxt(fid_fir, SEO_fir_Global_matrix)

                with open(SEO_fipr_filename, "w") as fid:
                    fid.write("unitEk=MeV, unitFi=deg, unitPr=1\n")
                with open(SEO_fipr_filename, "a") as fid_fipr:
                    np.savetxt(fid_fipr, SEO_fipr_Global_matrix)

        SEO_foldname = comm.bcast(SEO_foldname, root=0)
        return SEO_foldname

    def task(self, Ek_value):
        h = 0.001
        delta_r0_ini = -0.05
        r0, pr0, r0_analytical = self.SearchSEOUsingInitialEkVect(Ek_value, h,
                                                                  delta_r0_ini, verbose=False)  # the accurate initial value
        t_points, r_points = self.TrackAParticle(r0, pr0, Ek_value, np.pi * 2 / self.GlobalParas.Bmap.Nsectors, h)
        r_of_fi, pr_of_fi = r_points[:, 0], r_points[:, 1]

        orbitalfreq, _ = self.GetFreq(t_points, r_of_fi, pr_of_fi, Ek_value)

        return orbitalfreq


class FFAG_DynamicAperture:
    def __init__(self):
        pass

    def SearchDAUsingInitialEkVect(self):
        pass

# @profile
def FunctionForSEO(t, x, GlobalParameters):
    q = GlobalParameters.q
    dxdt = np.zeros(6)
    Bmap = GlobalParameters.Bmap
    BHarmonics = GlobalParameters.BHarmonics
    NSectors=Bmap.Nsectors
    fi = t % (np.pi * 2)
    r, pr, z, pz, P, ParticleIndex = x[0], x[1], x[ 2], x[3], x[ 4], x[ 5]

    r_vect = np.array([r,])
    fi_vect = np.array([fi,])
    fi_vect_mod = np.mod(fi_vect, 2*np.pi/NSectors)
    z_vect = np.array([z, ])
    Inj_flag = np.array([1,], dtype=bool)
    Survive_flag = np.array([1,], dtype=bool)

    if not Bmap.flag3D:
        Bz0 = Bmap.GetBz(r_vect, fi_vect_mod, SearchMod = True)
        Bfi = Bz0*0
        Br = Bz0*0

    else:
        # Bz0 = Bmap.GetBz(r_vect, fi_vect_mod)
        # Bfi = Bmap.GetBfi(r_vect, fi_vect_mod) * z_vect
        # Br = Bmap.GetBr(r_vect, fi_vect_mod) * z_vect
        Bz0, Bfi, Br = np.array([0.0,]), np.array([0.0,]), np.array([0.0,])
        Bmap.GetBfields(r_vect, fi_vect_mod, z_vect, Inj_flag, Survive_flag, Bz0, Br, Bfi, SearchMod=False)

    # harmonics of field error
    BHarmonics.AddBHarmonics_to_Bmap_njit(r_vect, fi_vect, z_vect, Bz0, Br, Bfi)

    pfi = np.sqrt((1 - pr ** 2 - pz ** 2))

    dxdt[0] = r * pr / pfi
    dxdt[1] = pfi - q / P * (r * Bz0 - r * pz / pfi * Bfi)
    dxdt[2] = r * pz / pfi
    dxdt[3] = q / P * (r * Br - r * pr / pfi * Bfi)
    dxdt[4] = 0
    dxdt[5] = 0

    return dxdt


# @profile
def FunctionForSEOVect(t, x, GlobalParameters):
    q = GlobalParameters.q
    nParticles = np.size(x, 0)  # 获取粒子数量
    dxdt = np.zeros((nParticles, 6))
    Bmap = GlobalParameters.Bmap
    NSectors = Bmap.Nsectors
    max_order = Bmap.max_order
    fi = np.ones(nParticles) * (t % (np.pi * 2))  # 为每个粒子生成相同的时间角度
    BHarmonics = GlobalParameters.BHarmonics

    # 提取粒子状态
    r, pr, z, pz, P, ParticleIndex = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]

    # 调整角度以适应扇区数
    fi_vect_mod = np.mod(fi, 2 * np.pi / NSectors)

    # 初始化 Bz、Br 和 Bfi 分量
    Bz0 = np.zeros_like(r)
    Br = np.zeros_like(r)
    Bfi = np.zeros_like(r)
    Inj_flag, Survive_flag = np.ones_like(r, dtype=bool), np.ones_like(r, dtype=bool)


    if not Bmap.flag3D:
        Bz_coeff = Bmap.GetBz(r, fi_vect_mod, SearchMod = True)
        # 只有Bz
        Bz0 += Bz_coeff
    else:
        # Bz_coeff = Bmap.GetBz(r, fi_vect_mod, SearchMod = True)
        # Br_coeff = Bmap.GetBr(r, fi_vect_mod, SearchMod = True) * z
        # Bfi_coeff = Bmap.GetBfi(r, fi_vect_mod, SearchMod = True) * z
        #
        # Bz0 += Bz_coeff
        # Br += Br_coeff
        # Bfi += Bfi_coeff

        Bmap.GetBfields(r, fi_vect_mod, z, Inj_flag, Survive_flag, Bz0, Br, Bfi, SearchMod=False)

    # 磁场误差谐波
    BHarmonics.AddBHarmonics_to_Bmap_njit(r, fi, z, Bz0, Br, Bfi)

    # 动量分量 pfi
    pfi = np.sqrt(1 - pr ** 2 - pz ** 2)

    # 计算 dxdt 各分量
    dxdt[:, 0] = r * pr / pfi
    dxdt[:, 1] = pfi - q / P * (r * Bz0 - r * pz / pfi * Bfi)
    dxdt[:, 2] = r * pz / pfi
    dxdt[:, 3] = q / P * (r * Br - r * pr / pfi * Bfi)
    dxdt[:, 4] = 0
    dxdt[:, 5] = 0

    return dxdt


def FunctionForSEOVectBz(t, x, GlobalParameters):
    q = GlobalParameters.q
    nParticles = np.size(x, 0)  # 获取粒子数量
    dxdt = np.zeros((nParticles, 6))
    Bmap = GlobalParameters.Bmap
    NSectors = Bmap.Nsectors
    max_order = Bmap.max_order
    fi = np.ones(nParticles) * (t % (np.pi * 2))  # 为每个粒子生成相同的时间角度

    # 提取粒子状态
    r, pr, z, pz, P, ParticleIndex = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]

    # 调整角度以适应扇区数
    fi_vect_mod = np.mod(fi, 2 * np.pi / NSectors)

    # 初始化 Bz、Br 和 Bfi 分量
    Bz0 = np.zeros_like(r)
    Br = np.zeros_like(r)
    Bfi = np.zeros_like(r)

    if not Bmap.flag3D:
        Bz_coeff = Bmap.GetBz(r, fi_vect_mod, SearchMod = False)
        # 只有Bz
        Bz0 += Bz_coeff
    else:
        Bz_coeff = Bmap.GetBz(r, fi_vect_mod, SearchMod = False)
        Br_coeff = Bmap.GetBr(r, fi_vect_mod, SearchMod = False) * z
        Bfi_coeff = Bmap.GetBfi(r, fi_vect_mod, SearchMod = False) * z

        Bz0 += Bz_coeff
        Br += Br_coeff
        Bfi += Bfi_coeff

    # 动量分量 pfi
    pfi = np.sqrt(1 - pr ** 2 - pz ** 2)

    # 计算 dxdt 各分量
    dxdt[:, 0] = r * pr / pfi
    dxdt[:, 1] = pfi - q / P * (r * Bz0 - r * pz / pfi * Bfi)
    dxdt[:, 2] = r * pz / pfi
    dxdt[:, 3] = q / P * (r * Br - r * pr / pfi * Bfi)
    dxdt[:, 4] = 0
    dxdt[:, 5] = 0

    return dxdt, Bz0, Br, Bfi

# @profile
def FunctionForAccelerationBunch_dt(t, mat_in,
                                    GlobalParameters,
                                    Bz0, Br, Bfi,
                                    E_z, E_r, E_fi,
                                    fi_vect, malloc_RFGap_index, dxdt):
    # shape of x: number of particles * number of coordinates in formula
    # sequence of the columns: (r, vr), (z, vz), (fi, dfidt), (t_inj, inj_flag),
    # (rf_phase, Esc_r, Esc_z, Esc_fi), (Bunch_ID, Local_ID, Global_ID)

    r = mat_in[:, 0]
    rdot = mat_in[:, 1]
    z = mat_in[:, 2]
    zdot = mat_in[:, 3]
    fi = mat_in[:, 4]
    Etotal_J = mat_in[:, 5]
    Inj_flag = mat_in[:, 7]
    Survive_flag = mat_in[:, 8]
    RF_Phase = mat_in[:, 9]
    Esc_r = mat_in[:, 10]
    Esc_z = mat_in[:, 11]
    Esc_fi = mat_in[:, 12]

    q = GlobalParameters.q
    c = GlobalParameters.c
    E0_J = GlobalParameters.E0
    Bmap = GlobalParameters.Bmap
    Emap = GlobalParameters.Emap
    NSectors = Bmap.Nsectors
    max_order = Bmap.max_order
    BHarmonics = GlobalParameters.BHarmonics

    # fi_vect = fi % (np.pi * 2 / NSectors)
    fast_mod_parallel(fi, (np.pi * 2 / NSectors), fi_vect)

    Bmap.GetBfields(r, fi_vect, z, Inj_flag, Survive_flag, Bz0, Br, Bfi, SearchMod=False)
    if max_order >= 1:
        Bmap.AddHigherOrderBmaps(r, fi_vect, z, Bz0, Br, Bfi, max_order)

    # 磁场误差谐波
    BHarmonics.AddBHarmonics_to_Bmap_njit(r, fi, z, Bz0, Br, Bfi)

    # 电场分量
    dRF_phase, v0_volt = Emap.Interpolation1D_freq_curve(t)
    dRF_phase = dRF_phase * np.pi * 2
    # print(t * 1e9, dRF_phase*365*1e-9, np.rad2deg(RF_Phase[0]))
    Emap.Interpolation2D_EMap(r, fi, z, E_r, E_z, E_fi, malloc_RFGap_index, v0_volt=v0_volt)

    RK4_equations_of_motion_njit(
        r, z, fi,
        rdot, zdot, Etotal_J,
        E_r, E_z, E_fi, Esc_r, Esc_z, Esc_fi,
        Br, Bz0, Bfi, RF_Phase,
        Inj_flag, Survive_flag,
        q, c, E0_J,
        dxdt, dRF_phase,
        Emap.rf_shift, malloc_RFGap_index,
        apply_RF=Emap.EnableFlag)

    return dxdt, Bz0, Br, Bfi


def func_get_R_B0(r, paras):
    q = paras.q
    f1 = paras.Bmap.f1
    Ek_value = paras.TempVariable
    P_start = FFAG_ConversionTools().Ek2P(Ek_value)
    y = r - P_start / q / f1(r)
    return y


if __name__ == '__main__':
    pass
