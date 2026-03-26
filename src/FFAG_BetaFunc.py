import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from FFAG_MathTools import FFAG_interpolation
from FFAG_ParasAndConversion import FFAG_ConversionTools, FFAG_GlobalParameters
from FFAG_track import FFAG_RungeKutta, FunctionForSEOVectBz
from FFAG_Field import FFAG_Bfield_analytical



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
        BMap = FFAG_Bfield_analytical(os.path.join(BmapPath, "config_Bmap.json"), 0, flag3D=True)

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
        # t_end = np.pi * 2 * 2 + t_start
        t_end = np.pi * 2 * 2 / BMap.Nsectors + t_start
        Ini_start = ini_matrix
        GlobalParas = FFAG_GlobalParameters()
        GlobalParas.AddBMap(BMap)

        _, tr_points_AllSteps, Bz_trajectory, Br_trajectory, Bf_trajectory, fi_trajectory = FFAG_RungeKutta().rk4_solve_vect_3DMatrix(
            FunctionForSEOVectBz, t_start, t_end, Ini_start, h, GlobalParas)



        # 5. 得到r, pr, z, pz相对于fi的插值函数，共5*n组,n为能量值
        a_axis0 = tr_points_AllSteps[:, 0, 0]  # steps
        a_axis1 = tr_points_AllSteps[0, :, -1]  # IDs
        func_interp = FFAG_interpolation().My2p5DInterp(tr_points_AllSteps, a_axis0, a_axis1)

        # 6. 给定任意fi0作为起始点，得到其起始点坐标，插值得到其1圈后的终点坐标
        beta_n = 1000
        beta_fi0 = np.linspace(0, 2 * np.pi / BMap.Nsectors, beta_n)
        beta_r_arr, beta_z_arr = np.zeros((Ek_arr_len, beta_n)), np.zeros((Ek_arr_len, beta_n))
        alpha_r_arr, alpha_z_arr = np.zeros((Ek_arr_len, beta_n)), np.zeros((Ek_arr_len, beta_n))

        for index_n in range(beta_n):
            # 遍历方位角
            t_s_oneTurn = beta_fi0[index_n]
            t_e_oneTurn = t_s_oneTurn + np.pi * 2 / BMap.Nsectors

            fi_s = np.ones(5 * Ek_arr_len) * t_s_oneTurn
            fi_e = np.ones(5 * Ek_arr_len) * t_e_oneTurn
            ID_s = a_axis1
            ID_e = a_axis1

            # 0~4对应第一个Ek, 5~9对应第二个Ek
            r1_s, _ = func_interp(fi_s, ID_s)  # axis0(fi_steps), axis1(IDs)
            r1_e, _ = func_interp(fi_e, ID_e)  # axis0(fi_steps), axis1(IDs)

            # 遍历能量值Ek
            for index_Ek in range(Ek_arr_len):

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

            NSectors = BMap.Nsectors
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





if __name__ == "__main__":
    SEO_FilePath = "./resultsSEO/map_p10_k5_s50-2023-09-23-23h-36m-22s"
    Ek_list = [30, 40, 50]
    FFAG_BetaFuncCalc().CalcBetaFunc(SEO_FilePath, Ek_list)
