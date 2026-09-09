import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from FFAG_MathTools import FFAG_interpolation
from FFAG_ParasAndConversion import FFAG_ConversionTools, FFAG_GlobalParameters
from FFAG_track import FFAG_RungeKutta, FunctionForSEOUnified
from FFAG_Field import FFAG_Bfield_analytical, FFAG_Bfield_3D_opera


def LoadBMapAuto(bmap_path, max_order=0, flag3D=True):
    analytical_config = os.path.join(bmap_path, "config_Bmap.json")
    if os.path.exists(analytical_config):
        return FFAG_Bfield_analytical(analytical_config, max_order, flag3D=flag3D)

    return FFAG_Bfield_3D_opera(field_dir=bmap_path)



class FFAG_BetaFuncCalc:
    def __init__(self):
        pass

    def LoadSEOParams(self, SEOFileName, Ek_c):
        path = os.path.dirname(os.path.dirname(SEOFileName))
        SEOdata = np.loadtxt(SEOFileName, skiprows=2)
        Ek0, r0, pr0 = SEOdata[:, 1], SEOdata[:, 6], SEOdata[:, 7]
        Func_Ek_r = FFAG_interpolation().linear_interpolation_vect(Ek0, r0)
        Func_Ek_pr = FFAG_interpolation().linear_interpolation_vect(Ek0, pr0)
        r_c, pr_c = Func_Ek_r(Ek_c), Func_Ek_pr(Ek_c)
        return np.column_stack([Ek_c, r_c, pr_c]), path, SEOdata

    def find_coef(self, r1_s, pr1_s, r1_e, pr1_e, r2_s, pr2_s, r2_e, pr2_e):
        # It takes four equations to determine the four elements.
        coef_matrix_00 = np.array([[r1_s, pr1_s, 0, 0],
                                   [0, 0, r1_s, pr1_s],
                                   [r2_s, pr2_s, 0, 0],
                                   [0, 0, r2_s, pr2_s]])
        coef_matrix_11, coef_matrix_12 = copy.deepcopy(coef_matrix_00), copy.deepcopy(coef_matrix_00)
        coef_matrix_21, coef_matrix_22 = copy.deepcopy(coef_matrix_00), copy.deepcopy(coef_matrix_00)
        coef_matrix_11[:, 0] = np.array([r1_e, pr1_e, r2_e, pr2_e])
        coef_matrix_12[:, 1] = np.array([r1_e, pr1_e, r2_e, pr2_e])
        coef_matrix_21[:, 2] = np.array([r1_e, pr1_e, r2_e, pr2_e])
        coef_matrix_22[:, 3] = np.array([r1_e, pr1_e, r2_e, pr2_e])
        delta00 = np.linalg.det(coef_matrix_00)
        a11 = np.linalg.det(coef_matrix_11) / delta00
        a12 = np.linalg.det(coef_matrix_12) / delta00
        a21 = np.linalg.det(coef_matrix_21) / delta00
        a22 = np.linalg.det(coef_matrix_22) / delta00
        return a11, a12, a21, a22

    def CalcBetaFunc(self, SEOPATHName, Ek_arr):
        h = 0.001
        delta_rr, delta_pr = 1e-7, 1e-7
        delta_zz, delta_pz = 1e-7, 1e-7
        delta_matrix = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                 [delta_rr, 0.0, 0.0, 0.0, 0.0, 0.0],
                                 [0.0, delta_pr, 0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, delta_zz, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, delta_pz, 0.0, 0.0]])

        # 0. 加载SEO
        SEOiniFileName = os.path.join(SEOPATHName, "SEO_ini.txt")

        # 1. 给定一组能量值Ek
        # Ek_arr = np.array([15, 25, 35])
        # Ek_arr = np.array(Ek_list)
        Ek_arr_len = np.size(Ek_arr, 0)
        SEO_ini, BmapPath, SEOdata = self.LoadSEOParams(SEOiniFileName, Ek_arr)
        Ek0, r0, pr0 = SEO_ini[:, 0], SEO_ini[:, 1], SEO_ini[:, 2]
        z0, pz0, ID0 = np.zeros_like(r0), np.zeros_like(pr0), np.zeros_like(r0)
        P0 = FFAG_ConversionTools().Ek2P(Ek0)

        # BMap = FFAG_BField_new(os.path.dirname(BmapPath), 0)
        BMap = LoadBMapAuto(BmapPath, 0, flag3D=True)

        # 2. 对于每个能量值，在SEO附近设置5个初始粒子，共5*Ek_arr_len个粒子
        ini_matrix_orig = np.column_stack((r0, pr0, z0, pz0, P0, ID0))
        ini_matrix_rept = np.repeat(ini_matrix_orig, 5, axis=0)
        del_matrix_tile = np.tile(delta_matrix, (Ek_arr_len, 1))
        ini_matrix = ini_matrix_rept + del_matrix_tile
        NumTestParticles = np.size(ini_matrix, 0)
        ini_matrix[:, -1] = np.arange(0, NumTestParticles)
        pass

        # 3. 跟踪2圈(改为了跟踪2个cell)
        print(f"Calculating Twiss Function, please wait... ...")
        t_start = 0
        t_end = np.pi * 2 * 2 + t_start
        # t_end = np.pi * 2 * 2 / BMap.Nsectors + t_start
        Ini_start = ini_matrix
        GlobalParas = FFAG_GlobalParameters()
        GlobalParas.AddBMap(BMap)

        _, tr_points_AllSteps, Bz_trajectory, Br_trajectory, Bf_trajectory, fi_trajectory = FFAG_RungeKutta().rk4_solve_vect_3DMatrix(
            FunctionForSEOUnified, t_start, t_end, Ini_start, h, GlobalParas)

        # 5. 得到r, pr, z, pz相对于fi的插值函数，共5*n组,n为能量值
        a_axis0 = tr_points_AllSteps[:, 0, 0]  # steps
        a_axis1 = tr_points_AllSteps[0, :, -1]  # IDs
        func_interp = FFAG_interpolation().My2p5DInterp(tr_points_AllSteps, a_axis0, a_axis1)

        # 6. 给定任意fi0作为起始点，得到其起始点坐标，插值得到其1圈后的终点坐标
        beta_n = 1000
        # beta_fi0 = np.linspace(0, 2 * np.pi / BMap.Nsectors, beta_n)
        beta_fi0 = np.linspace(0, 2 * np.pi, beta_n)
        beta_r_arr, beta_z_arr = np.zeros((Ek_arr_len, beta_n)), np.zeros((Ek_arr_len, beta_n))
        alpha_r_arr, alpha_z_arr = np.zeros((Ek_arr_len, beta_n)), np.zeros((Ek_arr_len, beta_n))

        for index_n in range(beta_n):
            # 遍历方位角
            t_s_oneTurn = beta_fi0[index_n]
            # t_e_oneTurn = t_s_oneTurn + np.pi * 2 / BMap.Nsectors
            t_e_oneTurn = t_s_oneTurn + np.pi * 2

            fi_s = np.ones(5 * Ek_arr_len) * t_s_oneTurn
            fi_e = np.ones(5 * Ek_arr_len) * t_e_oneTurn
            ID_s = a_axis1
            ID_e = a_axis1

            # 0~4对应第一个Ek, 5~9对应第二个Ek
            r1_s, _ = func_interp(fi_s, ID_s)  # axis0(fi_steps), axis1(IDs)
            r1_e, _ = func_interp(fi_e, ID_e)  # axis0(fi_steps), axis1(IDs)

            # 遍历能量值Ek
            for index_Ek in range(Ek_arr_len):
                # print(f"indexek={index_Ek}")
                # 起点和终点的x,px,z,pz坐标 1:5-0 6:10-5--->(index_Ek*5+1):(index_Ek+1)*5 - index_Ek*5
                # xM_s, xM_e = r1_s[1:5, 1:5]-r1_s[0, 1:5], r1_e[1:5, 1:5]-r1_e[0, 1:5]
                xM_s, xM_e = (r1_s[(index_Ek * 5 + 1):(index_Ek + 1) * 5, 1:5] - r1_s[index_Ek * 5, 1:5],
                              r1_e[(index_Ek * 5 + 1):(index_Ek + 1) * 5, 1:5] - r1_e[index_Ek * 5, 1:5])

                # xM_s, xM_e共4行，每行代表一个试探粒子，在初始位置和结束位置的dx,dpx,dy,dpy
                xs_1, pxs_1, xe_1, pxe_1 = xM_s[0, 0], xM_s[0, 1], xM_e[0, 0], xM_e[0, 1]
                xs_2, pxs_2, xe_2, pxe_2 = xM_s[1, 0], xM_s[1, 1], xM_e[1, 0], xM_e[1, 1]
                zs_1, pzs_1, ze_1, pze_1 = xM_s[2, 2], xM_s[2, 3], xM_e[2, 2], xM_e[2, 3]
                zs_2, pzs_2, ze_2, pze_2 = xM_s[3, 2], xM_s[3, 3], xM_e[3, 2], xM_e[3, 3]

                # (xs_1, pxs_1) ---> (xe_1, pxe_1)
                # (xs_2, pxs_2) ---> (xe_2, pxe_2)
                # 对应的线性方程组为：
                # xe_1 = a11*xs_1 + a12*pxs_1
                # pxe_1 = a21*xs_1 + a22*pxs_1
                # xe_2 = a11*xs_2 + a12*pxs_2
                # pxe_2 = a21*xs_2 + a22*pxs_2
                # 由于矩阵有4个元素, 确定矩阵需要4个方程：
                # xe_1 = a11*xs_1 + a12*pxs_1 + a21*0 + a22*0
                # pxe_1 = a11*0 + a12*0 + a21*xs_1 + a22*pxs_1
                # xe_2 = a11*xs_2 + a12*pxs_2 + a21*0 + a22*0
                # pxe_2 = a11*0 + a12*0 + a21*xs_2 + a22*pxs_2
                # 扩展为矩阵形式：
                # [xs_1, pxs_1, 0, 0] [a11]    [xe_1]
                # [0, 0, xs_1, pxs_1] [a12] =  [pxe_1]
                # [xs_2, pxs_2, 0, 0] [a21]    [xe_2]
                # [0, 0, xs_2, pxs_2] [a22]    [pxe_2]
                a11, a12, a21, a22 = self.find_coef(xs_1, pxs_1, xe_1, pxe_1,
                                                    xs_2, pxs_2, xe_2, pxe_2)
                b11, b12, b21, b22 = self.find_coef(zs_1, pzs_1, ze_1, pze_1,
                                                    zs_2, pzs_2, ze_2, pze_2)

                # 7. 计算fi0处的一圈传输矩阵
                cos_ur, cos_uz = (a11 + a22) / 2, (b11 + b22) / 2
                # ur, uz = np.arccos(cos_ur), np.arccos(cos_uz)
                sin_ur, sin_uz = np.sqrt(1 - cos_ur ** 2), np.sqrt(1 - cos_uz ** 2)
                if a12 * sin_ur < 0:
                    sin_ur = sin_ur * (-1)
                if b12 * sin_uz < 0:
                    sin_uz = sin_uz * (-1)

                # 8. 计算beta_r(beta_x), beta_z(beta_y)
                beta_r_temp, beta_z_temp = a12 / sin_ur, b12 / sin_uz
                alpha_r_temp, alpha_z_temp = (a11 - a22) / sin_ur / 2, (b11 - b22) / sin_uz / 2

                # 9. 存储到矩阵beta_r_arr
                beta_r_arr[index_Ek, index_n] = beta_r_temp
                beta_z_arr[index_Ek, index_n] = beta_z_temp
                alpha_r_arr[index_Ek, index_n] = alpha_r_temp
                alpha_z_arr[index_Ek, index_n] = alpha_z_temp

        beta_fi0_deg = np.rad2deg(beta_fi0)
        beta_r_arr_save = np.row_stack((beta_fi0_deg, beta_r_arr))
        beta_z_arr_save = np.row_stack((beta_fi0_deg, beta_z_arr))
        alpha_r_arr_save = np.row_stack((beta_fi0_deg, alpha_r_arr))
        alpha_z_arr_save = np.row_stack((beta_fi0_deg, alpha_z_arr))
        Bz_trajectory_save = np.row_stack((np.reshape(fi_trajectory,(1,-1)), Bz_trajectory[::5,:]))
        Bf_trajectory_save = np.row_stack((np.reshape(fi_trajectory,(1,-1)), Bf_trajectory[3::5,:]))
        Br_trajectory_save = np.row_stack((np.reshape(fi_trajectory,(1,-1)), Br_trajectory[3::5,:]))

        a = np.insert(Ek_arr, 0, 0)
        beta_r_arr_save = np.column_stack((a, beta_r_arr_save))
        beta_z_arr_save = np.column_stack((a, beta_z_arr_save))
        alpha_r_arr_save = np.column_stack((a, alpha_r_arr_save))
        alpha_z_arr_save = np.column_stack((a, alpha_z_arr_save))
        Bz_trajectory_save = np.column_stack((a, Bz_trajectory_save))
        Bf_trajectory_save = np.column_stack((a, Bf_trajectory_save))
        Br_trajectory_save = np.column_stack((a, Br_trajectory_save))

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            BetaFuncFileName = os.path.join(SEOPATHName, "BetaFuncR.txt")
            with open(BetaFuncFileName, "w") as fid:
                fid.write("unitEk=MeV, unitFi=deg, unitBeta=m\n")
            with open(BetaFuncFileName, "a") as fid:
                np.savetxt(fid, beta_r_arr_save)

            BetaFuncFileName = os.path.join(SEOPATHName, "BetaFuncZ.txt")
            with open(BetaFuncFileName, "w") as fid:
                fid.write("unitEk=MeV, unitFi=deg, unitBeta=m\n")
            with open(BetaFuncFileName, "a") as fid:
                np.savetxt(fid, beta_z_arr_save)

            AlphaFuncFileName = os.path.join(SEOPATHName, "AlphaFuncR.txt")
            with open(AlphaFuncFileName, "w") as fid:
                fid.write("unitEk=MeV, unitFi=deg, unitAlpha=1\n")
            with open(AlphaFuncFileName, "a") as fid:
                np.savetxt(fid, alpha_r_arr_save)

            AlphaFuncFileName = os.path.join(SEOPATHName, "AlphaFuncZ.txt")
            with open(AlphaFuncFileName, "w") as fid:
                fid.write("unitEk=MeV, unitFi=deg, unitAlpha=m\n")
            with open(AlphaFuncFileName, "a") as fid:
                np.savetxt(fid, alpha_z_arr_save)

            BzFileName = os.path.join(SEOPATHName, "Bz.txt")
            with open(BzFileName, "w") as fid:
                fid.write("unitEk=MeV, unitFi=deg, unitB=T\n")
            with open(BzFileName, "a") as fid:
                np.savetxt(fid, Bz_trajectory_save)

            BrFileName = os.path.join(SEOPATHName, "Br.txt")
            with open(BrFileName, "w") as fid:
                fid.write("unitEk=MeV, unitFi=deg, unitB=T\n")
            with open(BrFileName, "a") as fid:
                np.savetxt(fid, Br_trajectory_save)

            BfFileName = os.path.join(SEOPATHName, "Bf.txt")
            with open(BfFileName, "w") as fid:
                fid.write("unitEk=MeV, unitFi=deg, unitB=T\n")
            with open(BfFileName, "a") as fid:
                np.savetxt(fid, Bf_trajectory_save)

            # NSectors = BMap.Nsectors
            NSectors = 1
            # plot beta func
            plt.figure()
            for index_Ek in range(Ek_arr_len):
                ThisLabel = f"Ek = {Ek_arr[index_Ek]} MeV"
                plt.plot(beta_fi0_deg, beta_r_arr[index_Ek, :], label=ThisLabel)
            plt.legend()
            plt.xlim([0.0, 360.0 / NSectors])
            plt.xlabel("fi (deg)", fontsize=16)
            plt.ylabel("beta R (m)", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()

            plt.figure()
            for index_Ek in range(Ek_arr_len):
                ThisLabel = f"Ek = {Ek_arr[index_Ek]} MeV"
                plt.plot(beta_fi0_deg, beta_z_arr[index_Ek, :], label=ThisLabel)
            plt.legend()
            plt.xlim([0.0, 360.0 / NSectors])
            plt.xlabel("fi (deg)", fontsize=16)
            plt.ylabel("beta Z(m)", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()

            plt.figure()
            for index_Ek in range(Ek_arr_len):
                ThisLabel = f"Ek = {Ek_arr[index_Ek]} MeV"
                plt.plot(beta_fi0_deg, alpha_r_arr[index_Ek, :], label=ThisLabel)
            plt.legend()
            plt.xlim([0.0, 360.0 / NSectors])
            plt.xlabel("fi (deg)", fontsize=16)
            plt.ylabel("alpha R", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()

            plt.figure()
            for index_Ek in range(Ek_arr_len):
                ThisLabel = f"Ek = {Ek_arr[index_Ek]} MeV"
                plt.plot(beta_fi0_deg, alpha_z_arr[index_Ek, :], label=ThisLabel)
            plt.legend()
            plt.xlim([0.0, 360.0 / NSectors])
            plt.xlabel("fi (deg)", fontsize=16)
            plt.ylabel("alpha Z", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()

            plt.show()

    def CalcBetaFuncX(self, SEOPATHName, Ek_arr):
        h = 0.001
        delta_rr, delta_pr = 1e-7, 1e-7
        delta_zz, delta_pz = 1e-7, 1e-7

        delta_matrix = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [delta_rr, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, delta_pr, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, delta_zz, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, delta_pz, 0.0, 0.0],
        ])

        # ============================================================
        # 0. 加载 SEO
        # ============================================================
        SEOiniFileName = os.path.join(SEOPATHName, "SEO_ini.txt")

        # ============================================================
        # 1. 给定一组能量值 Ek
        # ============================================================
        Ek_arr_len = np.size(Ek_arr, 0)

        SEO_ini, BmapPath, SEOdata = self.LoadSEOParams(SEOiniFileName, Ek_arr)

        Ek0, r0, pr0 = SEO_ini[:, 0], SEO_ini[:, 1], SEO_ini[:, 2]
        z0 = np.zeros_like(r0)
        pz0 = np.zeros_like(pr0)
        ID0 = np.zeros_like(r0)

        P0 = FFAG_ConversionTools().Ek2P(Ek0)

        # BMap = FFAG_BField_new(os.path.dirname(BmapPath), 0)
        BMap = LoadBMapAuto(BmapPath, 0, flag3D=True)

        # ============================================================
        # 2. 对于每个能量值，在 SEO 附近设置 5 个初始粒子
        # ============================================================
        ini_matrix_orig = np.column_stack((r0, pr0, z0, pz0, P0, ID0))
        ini_matrix_rept = np.repeat(ini_matrix_orig, 5, axis=0)
        del_matrix_tile = np.tile(delta_matrix, (Ek_arr_len, 1))

        ini_matrix = ini_matrix_rept + del_matrix_tile

        NumTestParticles = np.size(ini_matrix, 0)
        ini_matrix[:, -1] = np.arange(0, NumTestParticles)

        # ============================================================
        # 3. 跟踪 2 圈
        # ============================================================
        print(f"Calculating Twiss Function, please wait... ...")

        t_start = 0.0
        t_end = np.pi * 2.0 * 2.0 + t_start
        # t_end = np.pi * 2.0 * 2.0 / BMap.Nsectors + t_start

        Ini_start = ini_matrix

        GlobalParas = FFAG_GlobalParameters()
        GlobalParas.AddBMap(BMap)

        (
            _,
            tr_points_AllSteps,
            Bz_trajectory,
            Br_trajectory,
            Bf_trajectory,
            fi_trajectory,
        ) = FFAG_RungeKutta().rk4_solve_vect_3DMatrix(
            FunctionForSEOUnified,
            t_start,
            t_end,
            Ini_start,
            h,
            GlobalParas,
        )

        # ============================================================
        # 5. 得到 r, pr, z, pz 相对于 fi 的插值函数
        # ============================================================
        a_axis0 = tr_points_AllSteps[:, 0, 0]  # fi steps
        a_axis1 = tr_points_AllSteps[0, :, -1]  # IDs

        func_interp = FFAG_interpolation().My2p5DInterp(
            tr_points_AllSteps,
            a_axis0,
            a_axis1,
        )

        # ============================================================
        # 新增：局部闭轨坐标转换函数
        # ------------------------------------------------------------
        # row 的列约定：
        #   row[0] = fi
        #   row[1] = r
        #   row[2] = pr
        #   row[3] = z
        #   row[4] = pz
        #
        # 这里默认：
        #   pr = v_r / v
        #   pz = v_z / v
        #   pfi = sqrt(1 - pr^2 - pz^2)
        #
        # 如果你的 pr 不是 v_r/v，而是 r*p_r/p_phi，
        # 需要改 get_direction_from_state()。
        # ============================================================

        def get_direction_from_state(row):
            """
            由柱坐标状态 row 构造归一化速度/动量方向矢量 v_dir。

            默认：
                pr = v_r / v
                pz = v_z / v
                pfi = v_fi / v = sqrt(1 - pr^2 - pz^2)

            返回：
                v_dir = pr e_r + pfi e_fi + pz e_z
            """
            fi = row[0]
            pr = row[2]
            pz = row[4]

            c = np.cos(fi)
            s = np.sin(fi)

            tmp = 1.0 - pr * pr - pz * pz
            if tmp < 0.0:
                # 数值误差保护
                tmp = 0.0

            pfi = np.sqrt(tmp)

            vx = pr * c - pfi * s
            vy = pr * s + pfi * c
            vz = pz

            return np.array([vx, vy, vz], dtype=np.float64)

        def get_local_basis_from_closed_orbit(row0):
            """
            根据闭轨粒子 row0 构造局部坐标基：
                t0 : 闭轨切向
                n0 : 水平法向，n0 = b0 x t0
                b0 : 竖直方向
            """
            v0 = get_direction_from_state(row0)

            norm_v0 = np.linalg.norm(v0)
            if norm_v0 < 1.0e-30:
                raise RuntimeError("Closed-orbit direction vector has zero norm.")

            t0 = v0 / norm_v0

            b0 = np.array([0.0, 0.0, 1.0], dtype=np.float64)

            n0 = np.cross(b0, t0)
            norm_n0 = np.linalg.norm(n0)

            if norm_n0 < 1.0e-30:
                raise RuntimeError("Local normal vector has zero norm.")

            n0 = n0 / norm_n0

            return t0, n0, b0

        def state_to_local(row, row0):
            """
            将粒子状态 row 转换为相对于闭轨 row0 的局部坐标：

                x  = (R - R0) · n0
                xp = (v · n0) / (v · t0)

                z  = (R - R0) · b0
                zp = (v · b0) / (v · t0)

            这里没有做最近闭轨点搜索，而是使用同一个 fi 插值点处的闭轨粒子 row0。
            这已经比直接 r-r0 更准确。
            """
            fi = row[0]
            r = row[1]
            z = row[3]

            fi0_local = row0[0]
            r0_local = row0[1]
            z0_local = row0[3]

            R = np.array([
                r * np.cos(fi),
                r * np.sin(fi),
                z,
            ], dtype=np.float64)

            R0 = np.array([
                r0_local * np.cos(fi0_local),
                r0_local * np.sin(fi0_local),
                z0_local,
            ], dtype=np.float64)

            t0, n0, b0 = get_local_basis_from_closed_orbit(row0)

            dR = R - R0

            x_local = np.dot(dR, n0)
            z_local = np.dot(dR, b0)

            v = get_direction_from_state(row)

            v_s = np.dot(v, t0)

            if np.abs(v_s) < 1.0e-14:
                raise RuntimeError("v dot t0 is too small when computing local slope.")

            xp_local = np.dot(v, n0) / v_s
            zp_local = np.dot(v, b0) / v_s

            return np.array([
                x_local,
                xp_local,
                z_local,
                zp_local,
            ], dtype=np.float64)

        def build_local_matrix(rows_all, index_Ek):
            """
            对某个能量 index_Ek，把 5 个粒子状态转换成局部坐标矩阵。

            rows_all:
                shape = (5*Ek_arr_len, ncol)

            对应某个能量的 5 个粒子：
                local index 0 : 闭轨粒子
                local index 1 : 水平位置扰动粒子
                local index 2 : 水平动量扰动粒子
                local index 3 : 竖直位置扰动粒子
                local index 4 : 竖直动量扰动粒子

            输出：
                xM_local shape = (4, 4)

            每行分别是一个试探粒子的：
                [x, xp, z, zp]
            """
            i0 = index_Ek * 5

            row0 = rows_all[i0, :]
            trial_rows = rows_all[i0 + 1:i0 + 5, :]

            xM_local = np.zeros((4, 4), dtype=np.float64)

            for j in range(4):
                xM_local[j, :] = state_to_local(trial_rows[j, :], row0)

            return xM_local

        # ============================================================
        # 6. 给定任意 fi0 作为起始点，得到一圈传输矩阵
        # ============================================================
        beta_n = 1000

        # beta_fi0 = np.linspace(0, 2.0 * np.pi / BMap.Nsectors, beta_n)
        beta_fi0 = np.linspace(0.0, 2.0 * np.pi, beta_n)

        beta_r_arr = np.zeros((Ek_arr_len, beta_n))
        beta_z_arr = np.zeros((Ek_arr_len, beta_n))

        alpha_r_arr = np.zeros((Ek_arr_len, beta_n))
        alpha_z_arr = np.zeros((Ek_arr_len, beta_n))

        for index_n in range(beta_n):

            t_s_oneTurn = beta_fi0[index_n]

            # t_e_oneTurn = t_s_oneTurn + 2.0 * np.pi / BMap.Nsectors
            t_e_oneTurn = t_s_oneTurn + 2.0 * np.pi

            fi_s = np.ones(5 * Ek_arr_len) * t_s_oneTurn
            fi_e = np.ones(5 * Ek_arr_len) * t_e_oneTurn

            ID_s = a_axis1
            ID_e = a_axis1

            # 0~4 对应第一个 Ek，5~9 对应第二个 Ek，依此类推
            r1_s, _ = func_interp(fi_s, ID_s)
            r1_e, _ = func_interp(fi_e, ID_e)

            for index_Ek in range(Ek_arr_len):

                # ====================================================
                # 原始版本：
                #   xM_s = trial_state - closed_state
                #   使用的是 [r-r0, pr-pr0, z-z0, pz-pz0]
                #
                # 修改后：
                #   使用局部闭轨坐标 [x, xp, z, zp]
                #   x  = (R-R0)·n
                #   xp = (v·n)/(v·t0)
                # ====================================================
                xM_s = build_local_matrix(r1_s, index_Ek)
                xM_e = build_local_matrix(r1_e, index_Ek)

                # xM_s, xM_e 共 4 行：
                #   第 0 行：水平位置扰动粒子
                #   第 1 行：水平动量扰动粒子
                #   第 2 行：竖直位置扰动粒子
                #   第 3 行：竖直动量扰动粒子
                #
                # 每行变量为：
                #   [x, xp, z, zp]

                xs_1, pxs_1 = xM_s[0, 0], xM_s[0, 1]
                xe_1, pxe_1 = xM_e[0, 0], xM_e[0, 1]

                xs_2, pxs_2 = xM_s[1, 0], xM_s[1, 1]
                xe_2, pxe_2 = xM_e[1, 0], xM_e[1, 1]

                zs_1, pzs_1 = xM_s[2, 2], xM_s[2, 3]
                ze_1, pze_1 = xM_e[2, 2], xM_e[2, 3]

                zs_2, pzs_2 = xM_s[3, 2], xM_s[3, 3]
                ze_2, pze_2 = xM_e[3, 2], xM_e[3, 3]

                # ====================================================
                # 由两个水平试探粒子求水平 2x2 传输矩阵
                # ====================================================
                a11, a12, a21, a22 = self.find_coef(
                    xs_1, pxs_1, xe_1, pxe_1,
                    xs_2, pxs_2, xe_2, pxe_2,
                )

                # ====================================================
                # 由两个竖直试探粒子求竖直 2x2 传输矩阵
                # ====================================================
                b11, b12, b21, b22 = self.find_coef(
                    zs_1, pzs_1, ze_1, pze_1,
                    zs_2, pzs_2, ze_2, pze_2,
                )

                # ====================================================
                # 7. 计算 fi0 处的一圈传输矩阵相位
                # ====================================================
                cos_ur = (a11 + a22) / 2.0
                cos_uz = (b11 + b22) / 2.0

                # 数值保护，避免 arccos/sqrt 因为 1e-15 误差出 NaN
                cos_ur_clip = np.clip(cos_ur, -1.0, 1.0)
                cos_uz_clip = np.clip(cos_uz, -1.0, 1.0)

                sin_ur = np.sqrt(max(0.0, 1.0 - cos_ur_clip ** 2))
                sin_uz = np.sqrt(max(0.0, 1.0 - cos_uz_clip ** 2))

                if a12 * sin_ur < 0.0:
                    sin_ur = -sin_ur

                if b12 * sin_uz < 0.0:
                    sin_uz = -sin_uz

                # ====================================================
                # 8. 计算 beta_r(beta_x), beta_z(beta_y)
                # ====================================================
                if np.abs(sin_ur) < 1.0e-14:
                    beta_r_temp = np.nan
                    alpha_r_temp = np.nan
                else:
                    beta_r_temp = a12 / sin_ur
                    alpha_r_temp = (a11 - a22) / (2.0 * sin_ur)

                if np.abs(sin_uz) < 1.0e-14:
                    beta_z_temp = np.nan
                    alpha_z_temp = np.nan
                else:
                    beta_z_temp = b12 / sin_uz
                    alpha_z_temp = (b11 - b22) / (2.0 * sin_uz)

                # ====================================================
                # 9. 存储
                # ====================================================
                beta_r_arr[index_Ek, index_n] = beta_r_temp
                beta_z_arr[index_Ek, index_n] = beta_z_temp

                alpha_r_arr[index_Ek, index_n] = alpha_r_temp
                alpha_z_arr[index_Ek, index_n] = alpha_z_temp

        # ============================================================
        # 10. 保存 beta / alpha
        # ============================================================
        beta_fi0_deg = np.rad2deg(beta_fi0)

        beta_r_arr_save = np.row_stack((beta_fi0_deg, beta_r_arr))
        beta_z_arr_save = np.row_stack((beta_fi0_deg, beta_z_arr))

        alpha_r_arr_save = np.row_stack((beta_fi0_deg, alpha_r_arr))
        alpha_z_arr_save = np.row_stack((beta_fi0_deg, alpha_z_arr))

        Bz_trajectory_save = np.row_stack((
            np.reshape(fi_trajectory, (1, -1)),
            Bz_trajectory[::5, :],
        ))

        Bf_trajectory_save = np.row_stack((
            np.reshape(fi_trajectory, (1, -1)),
            Bf_trajectory[3::5, :],
        ))

        Br_trajectory_save = np.row_stack((
            np.reshape(fi_trajectory, (1, -1)),
            Br_trajectory[3::5, :],
        ))

        a = np.insert(Ek_arr, 0, 0)

        beta_r_arr_save = np.column_stack((a, beta_r_arr_save))
        beta_z_arr_save = np.column_stack((a, beta_z_arr_save))

        alpha_r_arr_save = np.column_stack((a, alpha_r_arr_save))
        alpha_z_arr_save = np.column_stack((a, alpha_z_arr_save))

        Bz_trajectory_save = np.column_stack((a, Bz_trajectory_save))
        Bf_trajectory_save = np.column_stack((a, Bf_trajectory_save))
        Br_trajectory_save = np.column_stack((a, Br_trajectory_save))

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:

            BetaFuncFileName = os.path.join(SEOPATHName, "BetaFuncR.txt")
            with open(BetaFuncFileName, "w") as fid:
                fid.write("unitEk=MeV, unitFi=deg, unitBeta=m\n")
            with open(BetaFuncFileName, "a") as fid:
                np.savetxt(fid, beta_r_arr_save)

            BetaFuncFileName = os.path.join(SEOPATHName, "BetaFuncZ.txt")
            with open(BetaFuncFileName, "w") as fid:
                fid.write("unitEk=MeV, unitFi=deg, unitBeta=m\n")
            with open(BetaFuncFileName, "a") as fid:
                np.savetxt(fid, beta_z_arr_save)

            AlphaFuncFileName = os.path.join(SEOPATHName, "AlphaFuncR.txt")
            with open(AlphaFuncFileName, "w") as fid:
                fid.write("unitEk=MeV, unitFi=deg, unitAlpha=1\n")
            with open(AlphaFuncFileName, "a") as fid:
                np.savetxt(fid, alpha_r_arr_save)

            AlphaFuncFileName = os.path.join(SEOPATHName, "AlphaFuncZ.txt")
            with open(AlphaFuncFileName, "w") as fid:
                fid.write("unitEk=MeV, unitFi=deg, unitAlpha=1\n")
            with open(AlphaFuncFileName, "a") as fid:
                np.savetxt(fid, alpha_z_arr_save)

            BzFileName = os.path.join(SEOPATHName, "Bz.txt")
            with open(BzFileName, "w") as fid:
                fid.write("unitEk=MeV, unitFi=deg, unitB=T\n")
            with open(BzFileName, "a") as fid:
                np.savetxt(fid, Bz_trajectory_save)

            BrFileName = os.path.join(SEOPATHName, "Br.txt")
            with open(BrFileName, "w") as fid:
                fid.write("unitEk=MeV, unitFi=deg, unitB=T\n")
            with open(BrFileName, "a") as fid:
                np.savetxt(fid, Br_trajectory_save)

            BfFileName = os.path.join(SEOPATHName, "Bf.txt")
            with open(BfFileName, "w") as fid:
                fid.write("unitEk=MeV, unitFi=deg, unitB=T\n")
            with open(BfFileName, "a") as fid:
                np.savetxt(fid, Bf_trajectory_save)

            # ========================================================
            # 11. 绘图
            # ========================================================
            # NSectors_plot = BMap.Nsectors
            NSectors_plot = 1

            plt.figure()
            for index_Ek in range(Ek_arr_len):
                ThisLabel = f"Ek = {Ek_arr[index_Ek]} MeV"
                plt.plot(beta_fi0_deg, beta_r_arr[index_Ek, :], label=ThisLabel)
            plt.legend()
            plt.xlim([0.0, 360.0 / NSectors_plot])
            plt.xlabel("fi (deg)", fontsize=16)
            plt.ylabel("beta R (m)", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()

            plt.figure()
            for index_Ek in range(Ek_arr_len):
                ThisLabel = f"Ek = {Ek_arr[index_Ek]} MeV"
                plt.plot(beta_fi0_deg, beta_z_arr[index_Ek, :], label=ThisLabel)
            plt.legend()
            plt.xlim([0.0, 360.0 / NSectors_plot])
            plt.xlabel("fi (deg)", fontsize=16)
            plt.ylabel("beta Z (m)", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()

            plt.figure()
            for index_Ek in range(Ek_arr_len):
                ThisLabel = f"Ek = {Ek_arr[index_Ek]} MeV"
                plt.plot(beta_fi0_deg, alpha_r_arr[index_Ek, :], label=ThisLabel)
            plt.legend()
            plt.xlim([0.0, 360.0 / NSectors_plot])
            plt.xlabel("fi (deg)", fontsize=16)
            plt.ylabel("alpha R", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()

            plt.figure()
            for index_Ek in range(Ek_arr_len):
                ThisLabel = f"Ek = {Ek_arr[index_Ek]} MeV"
                plt.plot(beta_fi0_deg, alpha_z_arr[index_Ek, :], label=ThisLabel)
            plt.legend()
            plt.xlim([0.0, 360.0 / NSectors_plot])
            plt.xlabel("fi (deg)", fontsize=16)
            plt.ylabel("alpha Z", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()

            plt.show()

    def CalcBetaFuncXPX(self, SEOPATHName, Ek_arr):
        h = 0.001

        # 原始扰动仍然加在程序状态变量上
        # 后面会自动转换成严格局部变量 [x, x', z, z']
        delta_rr, delta_pr = 1e-7, 1e-7
        delta_zz, delta_pz = 1e-7, 1e-7

        delta_matrix = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [delta_rr, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, delta_pr, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, delta_zz, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, delta_pz, 0.0, 0.0],
        ])

        # ============================================================
        # 0. 加载 SEO
        # ============================================================
        SEOiniFileName = os.path.join(SEOPATHName, "SEO_ini.txt")

        # ============================================================
        # 1. 给定一组能量值 Ek
        # ============================================================
        Ek_arr_len = np.size(Ek_arr, 0)

        SEO_ini, BmapPath, SEOdata = self.LoadSEOParams(SEOiniFileName, Ek_arr)

        Ek0, r0, pr0 = SEO_ini[:, 0], SEO_ini[:, 1], SEO_ini[:, 2]
        z0 = np.zeros_like(r0)
        pz0 = np.zeros_like(pr0)
        ID0 = np.zeros_like(r0)

        P0 = FFAG_ConversionTools().Ek2P(Ek0)

        BMap = LoadBMapAuto(BmapPath, 0, flag3D=True)

        # ============================================================
        # 2. 对于每个能量值，在 SEO 附近设置 5 个初始粒子
        # ============================================================
        ini_matrix_orig = np.column_stack((r0, pr0, z0, pz0, P0, ID0))
        ini_matrix_rept = np.repeat(ini_matrix_orig, 5, axis=0)
        del_matrix_tile = np.tile(delta_matrix, (Ek_arr_len, 1))

        ini_matrix = ini_matrix_rept + del_matrix_tile

        NumTestParticles = np.size(ini_matrix, 0)
        ini_matrix[:, -1] = np.arange(0, NumTestParticles)

        # ============================================================
        # 3. 跟踪 2 圈
        # ============================================================
        print("Calculating Twiss Function, please wait... ...")

        t_start = 0.0
        t_end = np.pi * 2.0 * 2.0 + t_start

        Ini_start = ini_matrix

        GlobalParas = FFAG_GlobalParameters()
        GlobalParas.AddBMap(BMap)

        (
            _,
            tr_points_AllSteps,
            Bz_trajectory,
            Br_trajectory,
            Bf_trajectory,
            fi_trajectory,
        ) = FFAG_RungeKutta().rk4_solve_vect_3DMatrix(
            FunctionForSEOUnified,
            t_start,
            t_end,
            Ini_start,
            h,
            GlobalParas,
        )

        # ============================================================
        # 4. 得到 r, pr, z, pz 相对于 fi 的插值函数
        # ============================================================
        a_axis0 = tr_points_AllSteps[:, 0, 0]  # fi steps
        a_axis1 = tr_points_AllSteps[0, :, -1]  # particle IDs

        func_interp = FFAG_interpolation().My2p5DInterp(
            tr_points_AllSteps,
            a_axis0,
            a_axis1,
        )

        # ============================================================
        # 5. 严格局部闭轨坐标转换
        # ------------------------------------------------------------
        # row 的列约定：
        #   row[0] = fi
        #   row[1] = r
        #   row[2] = pr = v_r / v
        #   row[3] = z
        #   row[4] = pz = v_z / v
        #
        # 对于闭轨：
        #   r0_prime = dr0 / dtheta
        #
        # 由闭轨几何和速度方向关系：
        #   pr0 / ptheta0 = r0_prime / r0
        #
        # 所以：
        #   r0_prime = r0 * pr0 / ptheta0
        #
        # 局部坐标：
        #   x  = (r - r0) * r0 / sqrt(r0^2 + r0_prime^2)
        #
        #   x' = (r0 * pr - r0_prime * ptheta)
        #        / (r0_prime * pr + r0 * ptheta)
        #
        #   z' = pz / p_s
        #
        # 其中：
        #   p_s = (r0_prime * pr + r0 * ptheta)
        #         / sqrt(r0^2 + r0_prime^2)
        # ============================================================

        def safe_ptheta(pr, pz):
            tmp = 1.0 - pr * pr - pz * pz
            if tmp < 0.0:
                tmp = 0.0
            return np.sqrt(tmp)

        def closed_orbit_rprime_from_row0(row0):
            """
            由闭轨粒子的 r0 和 pr0 反推出 r0' = dr0/dtheta。

            对闭轨：
                pr0 = v_r / v
                ptheta0 = v_theta / v

            几何关系：
                v_r / v_theta = r0' / r0

            所以：
                r0' = r0 * pr0 / ptheta0
            """
            r0_local = row0[1]
            pr0_local = row0[2]
            pz0_local = row0[4]

            ptheta0_local = safe_ptheta(pr0_local, pz0_local)

            if abs(ptheta0_local) < 1.0e-14:
                raise RuntimeError("Closed-orbit ptheta is too small.")

            r0_prime = r0_local * pr0_local / ptheta0_local

            return r0_prime

        def state_to_local_geometry(row, row0):
            """
            将粒子状态 row 转换为相对于闭轨 row0 的严格局部坐标：

                x  = (r - r0) * r0 / h0
                xp = (r0 * pr - r0_prime * ptheta)
                     / (r0_prime * pr + r0 * ptheta)

                z  = z - z0
                zp = pz / p_s

            其中：
                h0 = sqrt(r0^2 + r0_prime^2)

                p_s = (r0_prime * pr + r0 * ptheta) / h0

            注意：
                这里假设 row 和 row0 已经插值到同一个 fi 截面。
                你的 func_interp(fi_s, ID_s) 正是这么做的。
            """
            r = row[1]
            pr = row[2]
            z = row[3]
            pz = row[4]

            r0_local = row0[1]
            z0_local = row0[3]

            r0_prime = closed_orbit_rprime_from_row0(row0)

            h0 = np.sqrt(r0_local * r0_local + r0_prime * r0_prime)

            if h0 < 1.0e-14:
                raise RuntimeError("Closed-orbit metric h0 is too small.")

            ptheta = safe_ptheta(pr, pz)

            # 局部水平位置
            x_local = (r - r0_local) * r0_local / h0

            # 局部纵向速度分量 p_s = v_s / v
            ps_num = r0_prime * pr + r0_local * ptheta
            p_s = ps_num / h0

            if abs(ps_num) < 1.0e-14:
                raise RuntimeError("Longitudinal velocity component is too small.")

            # 局部水平斜率 x' = v_x / v_s
            xp_local = (r0_local * pr - r0_prime * ptheta) / ps_num

            # 竖直坐标与竖直斜率
            z_local = z - z0_local
            zp_local = pz / p_s

            return np.array([
                x_local,
                xp_local,
                z_local,
                zp_local,
            ], dtype=np.float64)

        def build_local_matrix(rows_all, index_Ek):
            """
            对某个能量 index_Ek，把 5 个粒子状态转换成局部坐标矩阵。

            rows_all:
                shape = (5*Ek_arr_len, ncol)

            对应某个能量的 5 个粒子：
                local index 0 : 闭轨粒子
                local index 1 : 水平位置扰动粒子
                local index 2 : 水平动量扰动粒子
                local index 3 : 竖直位置扰动粒子
                local index 4 : 竖直动量扰动粒子

            输出：
                xM_local shape = (4, 4)

            每行分别是一个试探粒子的：
                [x, xp, z, zp]
            """
            i0 = index_Ek * 5

            row0 = rows_all[i0, :]
            trial_rows = rows_all[i0 + 1:i0 + 5, :]

            xM_local = np.zeros((4, 4), dtype=np.float64)

            for j in range(4):
                xM_local[j, :] = state_to_local_geometry(trial_rows[j, :], row0)

            return xM_local

        # ============================================================
        # 6. 给定任意 fi0 作为起始点，得到一圈传输矩阵
        # ============================================================
        beta_n = 1000

        beta_fi0 = np.linspace(0.0, 2.0 * np.pi, beta_n)

        beta_r_arr = np.zeros((Ek_arr_len, beta_n))
        beta_z_arr = np.zeros((Ek_arr_len, beta_n))

        alpha_r_arr = np.zeros((Ek_arr_len, beta_n))
        alpha_z_arr = np.zeros((Ek_arr_len, beta_n))

        det_x_min = np.inf
        det_z_min = np.inf

        for index_n in range(beta_n):

            t_s_oneTurn = beta_fi0[index_n]
            t_e_oneTurn = t_s_oneTurn + 2.0 * np.pi

            fi_s = np.ones(5 * Ek_arr_len) * t_s_oneTurn
            fi_e = np.ones(5 * Ek_arr_len) * t_e_oneTurn

            ID_s = a_axis1
            ID_e = a_axis1

            # 0~4 对应第一个 Ek，5~9 对应第二个 Ek，依此类推
            r1_s, _ = func_interp(fi_s, ID_s)
            r1_e, _ = func_interp(fi_e, ID_e)

            for index_Ek in range(Ek_arr_len):

                # ====================================================
                # 使用严格闭轨几何局部变量：
                #
                #   [x, xp, z, zp]
                #
                # 其中：
                #   x  = (r-r0) r0 / sqrt(r0^2 + r0'^2)
                #   xp = vx / vs
                # ====================================================
                xM_s = build_local_matrix(r1_s, index_Ek)
                xM_e = build_local_matrix(r1_e, index_Ek)

                # 水平两个试探粒子
                xs_1, pxs_1 = xM_s[0, 0], xM_s[0, 1]
                xe_1, pxe_1 = xM_e[0, 0], xM_e[0, 1]

                xs_2, pxs_2 = xM_s[1, 0], xM_s[1, 1]
                xe_2, pxe_2 = xM_e[1, 0], xM_e[1, 1]

                # 竖直两个试探粒子
                zs_1, pzs_1 = xM_s[2, 2], xM_s[2, 3]
                ze_1, pze_1 = xM_e[2, 2], xM_e[2, 3]

                zs_2, pzs_2 = xM_s[3, 2], xM_s[3, 3]
                ze_2, pze_2 = xM_e[3, 2], xM_e[3, 3]

                # ====================================================
                # 诊断：初始扰动矩阵不能接近奇异
                # ====================================================
                det_x_s = xs_1 * pxs_2 - xs_2 * pxs_1
                det_z_s = zs_1 * pzs_2 - zs_2 * pzs_1

                det_x_min = min(det_x_min, abs(det_x_s))
                det_z_min = min(det_z_min, abs(det_z_s))

                if abs(det_x_s) < 1.0e-20:
                    print(
                        "Warning: horizontal initial local matrix nearly singular:",
                        "index_Ek =", index_Ek,
                        "index_n =", index_n,
                        "det_x_s =", det_x_s,
                    )

                if abs(det_z_s) < 1.0e-20:
                    print(
                        "Warning: vertical initial local matrix nearly singular:",
                        "index_Ek =", index_Ek,
                        "index_n =", index_n,
                        "det_z_s =", det_z_s,
                    )

                # ====================================================
                # 由两个水平试探粒子求水平 2x2 传输矩阵
                # ====================================================
                a11, a12, a21, a22 = self.find_coef(
                    xs_1, pxs_1, xe_1, pxe_1,
                    xs_2, pxs_2, xe_2, pxe_2,
                )

                # ====================================================
                # 由两个竖直试探粒子求竖直 2x2 传输矩阵
                # ====================================================
                b11, b12, b21, b22 = self.find_coef(
                    zs_1, pzs_1, ze_1, pze_1,
                    zs_2, pzs_2, ze_2, pze_2,
                )

                # ====================================================
                # 7. 计算 fi0 处的一圈传输矩阵相位
                # ====================================================
                cos_ur = (a11 + a22) / 2.0
                cos_uz = (b11 + b22) / 2.0

                cos_ur_clip = np.clip(cos_ur, -1.0, 1.0)
                cos_uz_clip = np.clip(cos_uz, -1.0, 1.0)

                sin_ur = np.sqrt(max(0.0, 1.0 - cos_ur_clip ** 2))
                sin_uz = np.sqrt(max(0.0, 1.0 - cos_uz_clip ** 2))

                # 选择 sin_mu 的符号，使 beta = M12 / sin_mu 为正
                if a12 * sin_ur < 0.0:
                    sin_ur = -sin_ur

                if b12 * sin_uz < 0.0:
                    sin_uz = -sin_uz

                # ====================================================
                # 8. 计算 beta_r, alpha_r, beta_z, alpha_z
                # ====================================================
                if np.abs(sin_ur) < 1.0e-14:
                    beta_r_temp = np.nan
                    alpha_r_temp = np.nan
                else:
                    beta_r_temp = a12 / sin_ur
                    alpha_r_temp = (a11 - a22) / (2.0 * sin_ur)

                if np.abs(sin_uz) < 1.0e-14:
                    beta_z_temp = np.nan
                    alpha_z_temp = np.nan
                else:
                    beta_z_temp = b12 / sin_uz
                    alpha_z_temp = (b11 - b22) / (2.0 * sin_uz)

                # ====================================================
                # 9. 存储
                # ====================================================
                beta_r_arr[index_Ek, index_n] = beta_r_temp
                beta_z_arr[index_Ek, index_n] = beta_z_temp

                alpha_r_arr[index_Ek, index_n] = alpha_r_temp
                alpha_z_arr[index_Ek, index_n] = alpha_z_temp

        print("Minimum abs horizontal initial determinant =", det_x_min)
        print("Minimum abs vertical initial determinant   =", det_z_min)

        # ============================================================
        # 10. 保存 beta / alpha
        # ============================================================
        beta_fi0_deg = np.rad2deg(beta_fi0)

        beta_r_arr_save = np.row_stack((beta_fi0_deg, beta_r_arr))
        beta_z_arr_save = np.row_stack((beta_fi0_deg, beta_z_arr))

        alpha_r_arr_save = np.row_stack((beta_fi0_deg, alpha_r_arr))
        alpha_z_arr_save = np.row_stack((beta_fi0_deg, alpha_z_arr))

        Bz_trajectory_save = np.row_stack((
            np.reshape(fi_trajectory, (1, -1)),
            Bz_trajectory[::5, :],
        ))

        Bf_trajectory_save = np.row_stack((
            np.reshape(fi_trajectory, (1, -1)),
            Bf_trajectory[3::5, :],
        ))

        Br_trajectory_save = np.row_stack((
            np.reshape(fi_trajectory, (1, -1)),
            Br_trajectory[3::5, :],
        ))

        a = np.insert(Ek_arr, 0, 0)

        beta_r_arr_save = np.column_stack((a, beta_r_arr_save))
        beta_z_arr_save = np.column_stack((a, beta_z_arr_save))

        alpha_r_arr_save = np.column_stack((a, alpha_r_arr_save))
        alpha_z_arr_save = np.column_stack((a, alpha_z_arr_save))

        Bz_trajectory_save = np.column_stack((a, Bz_trajectory_save))
        Bf_trajectory_save = np.column_stack((a, Bf_trajectory_save))
        Br_trajectory_save = np.column_stack((a, Br_trajectory_save))

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:

            BetaFuncFileName = os.path.join(SEOPATHName, "BetaFuncR.txt")
            with open(BetaFuncFileName, "w") as fid:
                fid.write("unitEk=MeV, unitFi=deg, unitBeta=m\n")
            with open(BetaFuncFileName, "a") as fid:
                np.savetxt(fid, beta_r_arr_save)

            BetaFuncFileName = os.path.join(SEOPATHName, "BetaFuncZ.txt")
            with open(BetaFuncFileName, "w") as fid:
                fid.write("unitEk=MeV, unitFi=deg, unitBeta=m\n")
            with open(BetaFuncFileName, "a") as fid:
                np.savetxt(fid, beta_z_arr_save)

            AlphaFuncFileName = os.path.join(SEOPATHName, "AlphaFuncR.txt")
            with open(AlphaFuncFileName, "w") as fid:
                fid.write("unitEk=MeV, unitFi=deg, unitAlpha=1\n")
            with open(AlphaFuncFileName, "a") as fid:
                np.savetxt(fid, alpha_r_arr_save)

            AlphaFuncFileName = os.path.join(SEOPATHName, "AlphaFuncZ.txt")
            with open(AlphaFuncFileName, "w") as fid:
                fid.write("unitEk=MeV, unitFi=deg, unitAlpha=1\n")
            with open(AlphaFuncFileName, "a") as fid:
                np.savetxt(fid, alpha_z_arr_save)

            BzFileName = os.path.join(SEOPATHName, "Bz.txt")
            with open(BzFileName, "w") as fid:
                fid.write("unitEk=MeV, unitFi=deg, unitB=T\n")
            with open(BzFileName, "a") as fid:
                np.savetxt(fid, Bz_trajectory_save)

            BrFileName = os.path.join(SEOPATHName, "Br.txt")
            with open(BrFileName, "w") as fid:
                fid.write("unitEk=MeV, unitFi=deg, unitB=T\n")
            with open(BrFileName, "a") as fid:
                np.savetxt(fid, Br_trajectory_save)

            BfFileName = os.path.join(SEOPATHName, "Bf.txt")
            with open(BfFileName, "w") as fid:
                fid.write("unitEk=MeV, unitFi=deg, unitB=T\n")
            with open(BfFileName, "a") as fid:
                np.savetxt(fid, Bf_trajectory_save)

            # ========================================================
            # 11. 绘图
            # ========================================================
            NSectors_plot = 1

            plt.figure()
            for index_Ek in range(Ek_arr_len):
                ThisLabel = f"Ek = {Ek_arr[index_Ek]} MeV"
                plt.plot(beta_fi0_deg, beta_r_arr[index_Ek, :], label=ThisLabel)
            plt.legend()
            plt.xlim([0.0, 360.0 / NSectors_plot])
            plt.xlabel("fi (deg)", fontsize=16)
            plt.ylabel("beta R (m)", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()

            plt.figure()
            for index_Ek in range(Ek_arr_len):
                ThisLabel = f"Ek = {Ek_arr[index_Ek]} MeV"
                plt.plot(beta_fi0_deg, beta_z_arr[index_Ek, :], label=ThisLabel)
            plt.legend()
            plt.xlim([0.0, 360.0 / NSectors_plot])
            plt.xlabel("fi (deg)", fontsize=16)
            plt.ylabel("beta Z (m)", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()

            plt.figure()
            for index_Ek in range(Ek_arr_len):
                ThisLabel = f"Ek = {Ek_arr[index_Ek]} MeV"
                plt.plot(beta_fi0_deg, alpha_r_arr[index_Ek, :], label=ThisLabel)
            plt.legend()
            plt.xlim([0.0, 360.0 / NSectors_plot])
            plt.xlabel("fi (deg)", fontsize=16)
            plt.ylabel("alpha R", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()

            plt.figure()
            for index_Ek in range(Ek_arr_len):
                ThisLabel = f"Ek = {Ek_arr[index_Ek]} MeV"
                plt.plot(beta_fi0_deg, alpha_z_arr[index_Ek, :], label=ThisLabel)
            plt.legend()
            plt.xlim([0.0, 360.0 / NSectors_plot])
            plt.xlabel("fi (deg)", fontsize=16)
            plt.ylabel("alpha Z", fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()

            plt.show()


if __name__ == "__main__":
    SEO_FilePath = "./resultsSEO/map_p10_k5_s50-2023-09-23-23h-36m-22s"
    Ek_list = [30, 40, 50]
    FFAG_BetaFuncCalc().CalcBetaFunc(SEO_FilePath, Ek_list)
