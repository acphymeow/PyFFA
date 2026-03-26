import numpy as np
import os
import shutil
import copy
from numba import njit, types, prange
import math


def StepInterpNew(pre, post, fi_threshold):
    # Interp on the adjacent time steps

    rownum, colnum = np.size(pre, 0), np.size(pre, 1)

    x0 = pre[:, -1].reshape((-1, 1))
    x1 = post[:, -1].reshape((-1, 1))
    x = fi_threshold.reshape((-1, 1))
    y0 = copy.deepcopy(pre)
    y1 = copy.deepcopy(post)

    x0 = np.tile(x0, (1, colnum))
    x1 = np.tile(x1, (1, colnum))
    x = np.tile(x, (1, colnum))

    y = (x - x0) / (x1 - x0) * y1 + (x - x1) / (x0 - x1) * y0
    return y


class FFAG_GeometryCalc:
    def __init__(self):
        pass

    def doBoundingBoxesOverlap(self, A1x, A1y, A2x, A2y, B1x, B1y, B2x, B2y):
        # 快速排斥
        # A ---> (A1x, A1y, A2x, A2y)
        # B ---> (B1x, B1y, B2x, B2y)
        NonOverlap = ((np.minimum(A1x, A2x) >= np.maximum(B1x, B2x)) +
                      (np.maximum(A1x, A2x) <= np.minimum(B1x, B2x)) +
                      (np.minimum(A1y, A2y) >= np.maximum(B1y, B2y)) +
                      (np.maximum(A1y, A2y) <= np.minimum(B1y, B2y)))

        return NonOverlap

    # def doBoundingBoxesOverlap_Fast(self, A1x, A1y, A2x, A2y, B1x, B1y, B2x, B2y):
    #     # 快速排斥
    #     # A ---> (A1x, A1y, A2x, A2y)
    #     # B ---> (B1x, B1y, B2x, B2y)
    #     # 4个边界都不重叠，返回NonOverlap为True
    #     NonOverlap_xleft = (np.minimum(A1x, A2x) >= np.maximum(B1x, B2x))
    #     NonOverlap_xright = (np.maximum(A1x, A2x) <= np.minimum(B1x, B2x))
    #     NonOverlap_yup = (np.minimum(A1y, A2y) >= np.maximum(B1y, B2y))
    #     NonOverlap_ydown = (np.maximum(A1y, A2y) <= np.minimum(B1y, B2y))
    #     NonOverlap = NonOverlap_xleft + NonOverlap_xright + NonOverlap_yup + NonOverlap_ydown
    #     return NonOverlap

    def doLinesIntersect(self, A1x, A1y, A2x, A2y, B1x, B1y, B2x, B2y):
        # 跨立实验

        # The length of A, B
        LengthA = np.sqrt((A2x - A1x) ** 2 + (A2y - A1y) ** 2)
        LengthB = np.sqrt((B2x - B1x) ** 2 + (B2y - B1y) ** 2)
        # (B1B2 cross B1A1)/ norm(B1B2) = distance from A1 to B1B2
        # (B1B2 cross B1A2)/ norm(B1B2) = distance from A2 to B1B2

        # 以B为基准，判断点A1A2是否在线段B两侧
        # B1B2 cross B1A1
        # B1B2=((B2[0]-B1[0]), (B2[1]-B1[1]))
        # B1A1=((A1[0]-B1[0]), (A1[1]-B1[1]))
        cross_product_A1 = (B2x - B1x) * (A1y - B1y) - (B2y - B1y) * (A1x - B1x)
        # D_A1_to_B = cross_product_A1 / LengthB
        D_A1_to_B = np.divide(cross_product_A1, LengthB,
                              out=np.full_like(LengthB, np.nan, dtype=float), where=LengthB != 0)

        # B1B2 cross B1A2
        # B1B2=((B2[0]-B1[0]), (B2[1]-B1[1]))
        # B1A2=((A2[0]-B1[0]), (A2[1]-B1[1]))
        cross_product_A2 = (B2x - B1x) * (A2y - B1y) - (B2y - B1y) * (A2x - B1x)
        # D_A2_to_B = cross_product_A2 / LengthB
        D_A2_to_B = np.divide(cross_product_A2, LengthB,
                              out=np.full_like(LengthB, np.nan, dtype=float), where=LengthB != 0)

        # 以A为基准，判断点B1B2是否在线段A两侧
        # A1A2 cross A1B1
        # A1A2 = ((A2[0]-A1[0]), (A2[1]-A1[1]))
        # A1B1 = ((B1[0]-A1[0]), (B1[1]-A1[1]))
        cross_product_B1 = (A2x - A1x) * (B1y - A1y) - (A2y - A1y) * (B1x - A1x)
        # D_B1_to_A = cross_product_B1 / LengthA
        D_B1_to_A = np.divide(cross_product_B1, LengthA,
                              out=np.full_like(LengthA, np.nan, dtype=float), where=LengthA != 0)

        # A1A2 cross A1B2
        # A1A2 = ((A2[0] - A1[0]), (A2[1] - A1[1]))
        # A1B2 = ((B2[0] - A1[0]), (B2[1] - A1[1]))
        cross_product_B2 = (A2x - A1x) * (B2y - A1y) - (A2y - A1y) * (B2x - A1x)
        # D_B2_to_A = cross_product_B2 / LengthA
        D_B2_to_A = np.divide(cross_product_B2, LengthA,
                              out=np.full_like(LengthA, np.nan, dtype=float), where=LengthA != 0)

        crossA = cross_product_A1 * cross_product_A2
        crossB = cross_product_B1 * cross_product_B2

        # if crossA<0 and crossB<0, return true
        # else return false
        flagA = crossA < 0
        flagB = crossB < 0
        flag = flagA * flagB

        return flag, D_A1_to_B, D_A2_to_B, D_B1_to_A, D_B2_to_A

    def FindIntersectionVect(self, A, B):
        """
        同时判断多个线段与多个线段是否相交,A,B代表多条线段
        # A ---> (A1x, A1y, A2x, A2y)
        # B ---> (B1x, B1y, B2x, B2y)
        # test code: test_cross_spiral.py
        """
        # reshape A, B
        A = np.atleast_2d(A)
        B = np.atleast_2d(B)

        RowsOfA = np.size(A, 0)
        RowsOfB = np.size(B, 0)

        A = np.repeat(A, RowsOfB, axis=0)
        B = np.tile(B, (RowsOfA, 1))

        (A1x, A1y, A2x, A2y) = (A[:, 0], A[:, 1], A[:, 2], A[:, 3])
        (B1x, B1y, B2x, B2y) = (B[:, 0], B[:, 1], B[:, 2], B[:, 3])

        OverlapFlag = ~self.doBoundingBoxesOverlap(A1x, A1y, A2x, A2y, B1x, B1y, B2x, B2y)
        # 并不需要每个step进行doLinesIntersect, 若OverlapFlag全为False则跳过该步骤
        NoneParticleOverlap = np.sum(OverlapFlag) == 0

        if NoneParticleOverlap:
            IntersectFlag, D_A1_B, D_A2_B, D_B1_A, D_B2_A = (
                np.full_like(OverlapFlag, False, dtype=bool),
                np.full_like(OverlapFlag, False, dtype=bool),
                np.full_like(OverlapFlag, False, dtype=bool),
                np.full_like(OverlapFlag, False, dtype=bool),
                np.full_like(OverlapFlag, False, dtype=bool))
        else:
            IntersectFlag, D_A1_B, D_A2_B, D_B1_A, D_B2_A = (
                self.doLinesIntersect(A1x, A1y, A2x, A2y, B1x, B1y, B2x, B2y))


        Flag = OverlapFlag * IntersectFlag
        Flag_reshape = np.reshape(Flag, (RowsOfA, RowsOfB))
        D_A1_B_matrix = np.reshape(D_A1_B, (RowsOfA, RowsOfB))
        D_A2_B_matrix = np.reshape(D_A2_B, (RowsOfA, RowsOfB))
        D_B1_A_matrix = np.reshape(D_B1_A, (RowsOfA, RowsOfB))
        D_B2_A_matrix = np.reshape(D_B2_A, (RowsOfA, RowsOfB))

        return Flag_reshape, D_A1_B_matrix, D_A2_B_matrix, D_B1_A_matrix, D_B2_A_matrix


    def GetVectAngle(self, VectA, VectB):
        # get angles between VectA and VectB.
        # dimensions of inputs: (n,2) ndarray
        dotAB = VectA[:, 0] * VectB[:, 0] + VectA[:, 1] * VectB[:, 1]
        normAB = np.sqrt(VectA[:, 0] ** 2 + VectA[:, 1] ** 2) * np.sqrt(VectB[:, 0] ** 2 + VectB[:, 1] ** 2)
        # angle_radians = np.arccos(dotAB / normAB)
        angle_radians = np.arccos(np.clip((dotAB / normAB), -1, 1))
        angle_degree = np.rad2deg(angle_radians)
        CrossValue = VectA[:, 0] * VectB[:, 1] - VectA[:, 1] * VectB[:, 0]
        Sign = np.ones_like(CrossValue)
        Sign[CrossValue < 0] = -1
        angle_radians_WithSign, angle_degree_WithSign = angle_radians * Sign, angle_degree * Sign
        return angle_radians_WithSign, angle_degree_WithSign

    def CheckCrossFiVectNew(self, fi, fi_threshold):
        """
        检查当前step是否穿越了给定方位角
        test code: test_cross_Fi2.py
        """
        azimuth_PreStep, azimuth_CurrentStep = fi[:, -2], fi[:, -1]
        azimuth_threshold = np.ones_like(azimuth_PreStep) * fi_threshold

        # 映射到单位圆上3点
        a = np.column_stack((np.cos(azimuth_PreStep), np.sin(azimuth_PreStep)))
        b = np.column_stack((np.cos(azimuth_CurrentStep), np.sin(azimuth_CurrentStep)))
        c = np.column_stack((np.cos(azimuth_threshold), np.sin(azimuth_threshold)))

        # 判断点c是否在圆弧上
        angle_ab, _ = self.GetVectAngle(a, b)
        angle_ac, _ = self.GetVectAngle(a, c)
        is_on_arc = (angle_ac - angle_ab) * (angle_ac - 0) < 0

        angle_a = np.zeros_like(angle_ab)
        angle_b, angle_c = angle_a + angle_ab, angle_a + angle_ac

        return is_on_arc, angle_a, angle_b, angle_c

    def calculate_angle(self, point1, point2, point3):
        vector1 = point1 - point2
        vector2 = point3 - point2

        # dot_product = np.dot(vector1, vector2)
        # norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        dot_product_vect = np.sum(vector1 * vector2, axis=1)
        norm_product_vect = np.sqrt(vector1[:, 0] ** 2 + vector1[:, 1] ** 2) * np.sqrt(
            vector2[:, 0] ** 2 + vector2[:, 1] ** 2)

        ratio = dot_product_vect / norm_product_vect  # 可能产生浮点超范围
        ratio = np.clip(ratio, -1.0, 1.0)  # 确保在 [-1,1] 之间
        angle = np.arccos(ratio)
        angleWithSign = copy.deepcopy(angle)

        # a1, a2, a3 = vector1[0], vector1[1], 0.0
        # b1, b2, b3 = vector2[0], vector2[1], 0.0
        vector_cross = vector1[:, 0] * vector2[:, 1] - vector1[:, 1] * vector2[:, 0]

        # flag1 = vector_cross != 0
        flag1 = ~np.isclose(vector_cross, 0)

        flag2_1 = angle > np.deg2rad(179)
        flag3_1 = angle < np.deg2rad(1)

        angleWithSign[flag1] = angle[flag1] * vector_cross[flag1] / np.abs(vector_cross[flag1])

        flag2 = (~flag1) & flag2_1
        flag3 = (~flag1) & flag3_1

        angleWithSign[flag2] = np.pi
        angleWithSign[flag3] = 0

        return angleWithSign

    # @profile
    def point_in_convex_polygon(self, polygon_points, point_to_check):
        """
        判断点是否在凸多边形内部

        参数：
        scatter_points：散点的坐标，二维NumPy数组，每一行代表一个散点的坐标。
        point_to_check：待检测点的坐标，一维NumPy数组或列表。

        返回值：
        如果待检测点在凸多边形内部，返回True；否则，返回False。
        """
        # 将散点坐标转换为NumPy数组
        polygon_points_closed = np.vstack((polygon_points, polygon_points[0, :]))

        NumOfPointsToCheck = np.size(point_to_check, 0)
        NumOfPointsToPolygon = np.size(polygon_points, 0)

        polygon_points_left = polygon_points_closed[:-1, :]
        polygon_points_right = polygon_points_closed[1:, :]

        point_i_repeat = np.repeat(point_to_check, NumOfPointsToPolygon, axis=0)
        polygon_points_left_repeat = np.tile(polygon_points_left, (NumOfPointsToCheck, 1))
        polygon_points_right_repeat = np.tile(polygon_points_right, (NumOfPointsToCheck, 1))

        angleWithSign = self.calculate_angle(polygon_points_left_repeat, point_i_repeat, polygon_points_right_repeat)
        angleWithSign_splits = np.split(angleWithSign, NumOfPointsToCheck)
        sums = np.sum(angleWithSign_splits, axis=1)

        flag_in_polygon = np.abs(sums) > np.deg2rad(350.0)
        return flag_in_polygon

    # @profile
    def point_in_convex_polygon_fast(self, r, fi, polygon, x_min, x_max, y_min, y_max):
        inside = point_in_convex_polygon_batch(r, fi, polygon, x_min, x_max, y_min, y_max)
        return inside


class FFAG_SegmentTools:
    def __init__(self):
        pass

    def MatrixDotCross(self, matrix1, matrix2):
        DotValue = matrix1[:, 0] * matrix2[:, 0] + matrix1[:, 1] * matrix2[:, 1]
        CrossValue = matrix1[:, 0] * matrix2[:, 1] - matrix1[:, 1] * matrix2[:, 0]
        return DotValue, CrossValue

    def MatrixNorm(self, matrix1):
        norm1 = np.sqrt(matrix1[:, 0] ** 2 + matrix1[:, 1] ** 2)
        return norm1

    def CentralLine2Segments(self, cl):
        # 将曲线上的连续散点转换为折线端点（n,2）--->(n-1,4)
        cl_mid = cl[1:-1, :]
        cl_mid_repeat = np.repeat(cl_mid, 2, axis=0)
        cl_mid_repeat_flat = np.reshape(cl_mid_repeat, (1, -1))
        cl_mid_repeat_flat_full = (
            np.column_stack((np.atleast_2d(cl[0]), cl_mid_repeat_flat, np.atleast_2d(cl[-1]))))
        cl_segment = np.reshape(cl_mid_repeat_flat_full, (-1, 4))
        return cl_segment

    def SegmentPointMin(self, line1, sample_point):
        # find the minimum distance from a point to a segment
        # test code : test_segment_min.py
        cl_segment = self.CentralLine2Segments(line1)

        # 计算直线上的向量
        point1, point2 = cl_segment[:, 0:2], cl_segment[:, 2:4]  # 折线的起始点、终点
        line_vector = point2 - point1
        line_vector = np.atleast_2d(line_vector)

        # 计算样本点到直线的向量
        sample_vector = sample_point - point1
        sample_vector = np.atleast_2d(sample_vector)

        # 计算最短距离的点
        ratio = self.MatrixDotCross(sample_vector, line_vector)[0] / self.MatrixDotCross(line_vector, line_vector)[0]
        flag0 = ratio < 0
        flag1 = ratio > 1
        ratio[flag0], ratio[flag1] = 0, 1
        ratio = np.reshape(np.repeat(ratio, 2, axis=0), (-1, 2))
        projection_point = point1 + ratio * line_vector

        # 计算全局最短距离的点
        D_proj_samp = self.MatrixNorm(projection_point - sample_point)
        min_arg = np.argmin(D_proj_samp)
        projection_min, D_min = projection_point[min_arg, :], D_proj_samp[min_arg]

        return projection_min, D_min, projection_point

    def SegmentGridMin(self, line1, grid_points_x, grid_points_y):
        # find the minimum distance from grid points to a segment
        # test code : test_segment_min.py
        cl_segment = self.CentralLine2Segments(line1)
        cl_point_s, cl_point_e = cl_segment[:, 0:2], cl_segment[:, 2:4]  # 折线的起始点、终点

        n_grid_x, n_grid_y = np.size(grid_points_x, 0), np.size(grid_points_x, 1)
        n_line_d = np.size(cl_segment, 0)

        # 分离折线的xy坐标
        cl_point_s_x_raw, cl_point_s_y_raw = cl_point_s[:, 0], cl_point_s[:, 1]
        cl_point_e_x_raw, cl_point_e_y_raw = cl_point_e[:, 0], cl_point_e[:, 1]
        # 切向向量
        tan_direct_x = cl_point_e_x_raw - cl_point_s_x_raw
        tan_direct_y = cl_point_e_y_raw - cl_point_s_y_raw
        # broadcast
        cl_point_s_x = np.tile(cl_point_s_x_raw[np.newaxis, np.newaxis, :], (n_grid_x, n_grid_y, 1))
        cl_point_s_y = np.tile(cl_point_s_y_raw[np.newaxis, np.newaxis, :], (n_grid_x, n_grid_y, 1))
        cl_point_e_x = np.tile(cl_point_e_x_raw[np.newaxis, np.newaxis, :], (n_grid_x, n_grid_y, 1))
        cl_point_e_y = np.tile(cl_point_e_y_raw[np.newaxis, np.newaxis, :], (n_grid_x, n_grid_y, 1))

        grid_x_broadcast = np.tile(grid_points_x[..., np.newaxis], (1, 1, n_line_d))
        grid_y_broadcast = np.tile(grid_points_y[..., np.newaxis], (1, 1, n_line_d))

        # 计算网格点到折线的向量(折线的每段有2点，2个二维坐标，4个xy)
        line_vector_x = cl_point_e_x - cl_point_s_x
        line_vector_y = cl_point_e_y - cl_point_s_y
        grid_vector_s_x = grid_x_broadcast - cl_point_s_x
        grid_vector_s_y = grid_y_broadcast - cl_point_s_y

        # calculate dot(line_vector, grid_vector)
        Dot_line_line = line_vector_x * line_vector_x + line_vector_y * line_vector_y
        Dot_grid_line = grid_vector_s_x * line_vector_x + grid_vector_s_y * line_vector_y

        # 计算到折线各分段最短距离
        ratio = Dot_grid_line / Dot_line_line
        flag0 = ratio < 0
        flag1 = ratio > 1
        ratio[flag0], ratio[flag1] = 0, 1
        projection_point_x = cl_point_s_x + ratio * line_vector_x
        projection_point_y = cl_point_s_y + ratio * line_vector_y

        # 计算全局最短距离的点
        D_proj_samp = np.sqrt(
            (projection_point_x - grid_x_broadcast) ** 2 + (projection_point_y - grid_y_broadcast) ** 2)
        min_arg_axis2 = np.argmin(D_proj_samp, axis=2)
        min_arg_axis0, min_arg_axis1 = np.indices((n_grid_x, n_grid_y))[0], np.indices((n_grid_x, n_grid_y))[1]

        projection_x_min = projection_point_x[min_arg_axis0, min_arg_axis1, min_arg_axis2]
        projection_y_min = projection_point_y[min_arg_axis0, min_arg_axis1, min_arg_axis2]
        D_min = D_proj_samp[min_arg_axis0, min_arg_axis1, min_arg_axis2]

        # find the tang direction of the projection points
        projection_tan_x = tan_direct_x[min_arg_axis2]
        projection_tan_y = tan_direct_y[min_arg_axis2]

        return projection_x_min, projection_y_min, D_min, projection_point_x, projection_point_y, projection_tan_x, projection_tan_y

    def find_circle_line_intersections(self, line_points, circle_x, circle_y, radius):
        # 二维平面上有若干散点构成的折线段，给定一个半径值R，求以R为半径的圆与折线的交点坐标
        # test code: test_segment_circle_cross.py
        intersections = []
        for i in range(len(line_points) - 1):
            x1, y1 = line_points[i]
            x2, y2 = line_points[i + 1]
            # 计算线段的方向向量
            dx = x2 - x1
            dy = y2 - y1

            # 计算直线的参数
            A = dx ** 2 + dy ** 2
            B = 2 * (dx * (x1 - circle_x) + dy * (y1 - circle_y))
            C = (x1 - circle_x) ** 2 + (y1 - circle_y) ** 2 - radius ** 2

            # 计算判别式
            discriminant = B ** 2 - 4 * A * C

            # 如果判别式小于零，没有交点
            if discriminant < 0:
                continue

            # 计算两个交点
            t1 = (-B + np.sqrt(discriminant)) / (2 * A)
            t2 = (-B - np.sqrt(discriminant)) / (2 * A)

            # 检查是否交点在线段内
            if 0 <= t1 <= 1:
                intersection_x1 = x1 + t1 * dx
                intersection_y1 = y1 + t1 * dy
                intersections.append((intersection_x1, intersection_y1))

            if 0 <= t2 <= 1:
                intersection_x2 = x1 + t2 * dx
                intersection_y2 = y1 + t2 * dy
                intersections.append((intersection_x2, intersection_y2))

        return intersections

    def segment_interp(self, points, ratio):
        """
        计算折线的长度并进行比例剖分，返回插值点和输入点的坐标，并将它们按距离进行排序。
        test code: test_segment_interp.py

        Parameters:
        points (numpy.ndarray): 二维数组，包含按序排序的散点坐标，每行表示一个点。
        ratio (numpy.ndarray): 包含要进行等距剖分的位置的比例值。

        Returns:
        xyd_interp (numpy.ndarray): 包含等距剖分的插值点坐标和对应的距离。
        xyd_input (numpy.ndarray): 包含输入的散点坐标和对应的距离。
        xyd_all_sorted (numpy.ndarray): 包含所有点（插值点和输入点）按距离排序后的坐标。

        """
        flag_ratio_less_0 = ratio <= 0
        flag_ratio_larger_1 = ratio >= 1
        ratio[flag_ratio_less_0] = 0.000001
        ratio[flag_ratio_larger_1] = 1

        # 计算每个线段的长度
        length = np.linalg.norm(np.diff(points, axis=0), axis=1)

        # 计算每个点到起点的距离
        distances = np.cumsum(np.insert(length, 0, 0))

        # 计算整条线的总长度
        total_length = np.sum(length)

        # 根据所给的比例值计算插值点
        idx_e = np.searchsorted(distances, total_length * ratio)
        idx_s = idx_e - 1

        distances_interp = total_length * ratio
        mod_length_ratio = (distances_interp - distances[idx_s]) / length[idx_s]
        mod_length_ratio_repeat = np.repeat(mod_length_ratio, 2, axis=0)
        mod_length_ratio_reshape = np.reshape(mod_length_ratio_repeat, (-1, 2))
        xy_s, xy_e = points[idx_s, :], points[idx_e, :]

        xy_interp = xy_s + (xy_e - xy_s) * mod_length_ratio_reshape

        xyd_interp = np.column_stack((xy_interp, distances_interp))
        xyd_input = np.column_stack((points, distances))
        xyd_all = np.row_stack((xyd_input, xyd_interp))

        # 获取根据最后一列（distance）排序后的索引
        sorted_indices = np.argsort(xyd_all[:, -1])
        # 使用索引对 xyd_all 进行排序
        xyd_all_sorted = xyd_all[sorted_indices]

        return xyd_interp, xyd_input, xyd_all_sorted


def dynamic_insert_3D(matrix_a, indices_i, indices_j, value_b):
    # indices_i and indices_j are (1, n) arrays
    # value_b is a (n, l) array
    # matrix_a is a (j, k, l) array
    rows, cols, deepth = np.size(matrix_a, 0), np.size(matrix_a, 1), np.size(matrix_a, 2)
    # steps, number of particles, dims
    max_i = np.max(indices_i)

    # Extend matrix_a if needed
    if max_i + 1 > rows:
        extension = np.ones((4, cols, deepth)) * (-1)
        matrix_a = np.concatenate((matrix_a, extension), axis=0)

    matrix_a[indices_i, indices_j, :] = value_b

    return matrix_a


def append_with_limit(lst, new_data, limit):
    lst.append(new_data)
    if len(lst) > limit:
        del lst[0]


def concatenate_with_limit(tr_points_AllSteps, tr_points_ThisStep, max_length):
    # 将tr_points_ThisStep附加到tr_points_AllSteps
    tr_points_AllSteps = np.concatenate((tr_points_AllSteps, tr_points_ThisStep[np.newaxis, :, :]), axis=0)

    # 如果tr_points_AllSteps的长度已经超过了max_length，删除前面的层
    CurrentLength = np.size(tr_points_AllSteps, 0)
    if CurrentLength > max_length:
        tr_points_AllSteps = tr_points_AllSteps[-max_length:, :, :]

    return tr_points_AllSteps


class FFAG_FileOperation:
    def __init__(self):
        pass

    def CreatAEmptyFold(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"The folder '{folder_path}' has been created.")
        else:
            print(f"The folder '{folder_path}' already exists.")
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    else:
                        os.remove(file_path)
                except Exception as e:
                    print(f"Failed to delete file: {file_path} ({e})")
            print(f"The folder '{folder_path}' has been cleared.")

    def CreatAEmptyFold_safe(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"The folder '{folder_path}' has been created.")
        else:
            user_input = input(f"Are you sure you want to clear the folder '{folder_path}'? (Y/N): ")

            if user_input.upper() == 'Y':
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        if os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                        else:
                            os.remove(file_path)
                    except Exception as e:
                        print(f"Failed to delete file: {file_path} ({e})")
                print(f"The folder '{folder_path}' has been cleared.")
            else:
                print("Operation canceled by the user.")

    def CreatAEmptyFoldUncovered(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"The folder '{folder_path}' has been created.")

        i = 1
        while True:
            new_folder_name = str(i)
            new_folder_path = os.path.normpath(os.path.join(folder_path, new_folder_name))
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
                print(f"Created subfolder '{new_folder_name}' in '{folder_path}'.")
                break
            else:
                i += 1

        return new_folder_path


class FFAG_DynamicMatrix:
    """
    test code: test_dynamicMatrixWithLimit.py
    """
    def __init__(self, dim2, dim3):
        self.MatrixCapability = 16
        self.Matrix3D = -1 * np.ones((self.MatrixCapability, dim2, dim3))
        self.Insert_index = 0
        self.dim2, self.dim3 = dim2, dim3
        self.Matrix3DValid = None

    def find_power_to_exceed(self, a, b):
        # 计算MatrixCapability翻倍次数，向上取整
        power = int(np.ceil(np.log2(b / a)))
        bDoubleN = a * (2 ** power)
        bExpand = bDoubleN - a
        return power, bDoubleN, bExpand

    def dynamic_insert(self, Matrix2D):
        # Expand matrix if needed
        if self.Insert_index > self.MatrixCapability - 1:
            _, _, CapabilityExpand = self.find_power_to_exceed(self.MatrixCapability, self.Insert_index+1)
            MatrixExtend = -1 * np.ones((CapabilityExpand, self.dim2, self.dim3))
            self.Matrix3D = np.concatenate((self.Matrix3D, MatrixExtend), axis=0)
            self.MatrixCapability += CapabilityExpand

        self.Matrix3D[self.Insert_index, :, :] = Matrix2D
        self.Insert_index += 1

    def RestoreMatrix(self,):
        self.Matrix3DValid = self.Matrix3D[:self.Insert_index, :, :]
        return self.Matrix3DValid


class FFAG_DynamicMatrixWithLimit:

    def __init__(self, dim2, dim3, SizeLimit):
        self.SizeLimit = SizeLimit
        self.Matrix3D = -1 * np.ones((SizeLimit, dim2, dim3))
        self.CurrentTurns = 0
        self.CurrentRows = 0
        self.Insert_index = 0
        self.Matrix3DValid = None

    def dynamic_insert_withLimit(self, Matrix2D):
        self.CurrentTurns = int(self.Insert_index / self.SizeLimit)
        self.CurrentRows = self.Insert_index - self.CurrentTurns * self.SizeLimit
        self.Matrix3D[self.CurrentRows, :, :] = Matrix2D
        self.Insert_index += 1

    def RestoreMatrix(self):
        if self.CurrentTurns == 0:
            self.Matrix3DValid = self.Matrix3D[:self.CurrentRows + 1, :, :]
        else:
            self.PrePart = copy.deepcopy(self.Matrix3D[self.CurrentRows + 1:, :, :])
            self.PostPart = copy.deepcopy(self.Matrix3D[:self.CurrentRows + 1, :, :])
            self.Matrix3DValid = np.row_stack((self.PrePart, self.PostPart))
        return self.Matrix3DValid


class FFAG_FlatBunch():
    def __init__(self):
        pass

    def expand_angles(self, source_angles, expansion_factor=0.4):
        """
        按指定倍数扩展原始角度区间，例如 [0,30] 扩展为 [-15,45]。
        扩展后的角度区间包括原区间及其对称延伸，保持原采样步长。

        参数:
            source_angles  : 原始角度数组
            expansion_factor : 扩展倍数，默认为 1.0（即 100% 扩展）

        返回:
            extended_angles : 扩展后的角度数组
        """
        # 原区间的最大值和跨度
        angle_min, angle_max = np.min(source_angles), np.max(source_angles)
        angle_range = (angle_max - angle_min)

        # 扩展点的数量
        n = len(source_angles)
        left_count = int(n * expansion_factor // 2)  # 左侧扩展点数量
        right_count = int(n * expansion_factor // 2)  # 右侧扩展点数量

        left_count, right_count = max(left_count, 3), max(right_count, 3)

        # 索引构造：左侧和右侧扩展
        indices_left = np.arange(-left_count, -1)  # 左侧扩展部分索引
        indices_right = np.arange(1, right_count)  # 右侧扩展部分索引

        # # 索引构造：左侧和右侧扩展
        # n = len(source_angles)
        # indices_left = np.arange(-n // 2, -1)  # 左侧扩展部分索引
        # indices_right = np.arange(1, n - (n // 2))  # 右侧扩展部分索引

        # 左扩展：映射到周期内，取末尾元素减去 N
        left_extension = np.take(source_angles, indices_left, mode='wrap') - angle_range

        # 中间部分：原始区间保持不变
        middle = source_angles

        # 右扩展：映射到周期内，取开头元素加上 N
        right_extension = np.take(source_angles, indices_right, mode='wrap') + angle_range

        # 合并扩展区间
        extended_angles = np.concatenate([left_extension, middle, right_extension])

        # 源数组在扩展数组中的位置索引
        original_indices = np.arange(len(left_extension), len(left_extension) + len(middle))

        return extended_angles, original_indices, indices_left, indices_right

    def get_expand_index(self, source_angles):
        """
        扩展原始数据并计算扩展后的角度、索引、数值数组。
        1. 计算扩展后的角度数组 expanded_angles (扩展范围为原区间长度的1倍)；
        2. 对 expanded_angles 中的每个角度，基于周期性映射 (mod 360) + 最近点匹配，计算其索引和值；
        3. 如果角度不在原区间内，索引为 -1，值为 0。

        参数:
            source_angles  : 原始角度数组
            source_indices : 原始索引数组
            source_values  : 原始数值数组

        返回:
            expanded_angles  : 扩展后的角度数组
            expanded_indices : 扩展后的索引数组
            expanded_values  : 扩展后的数值数组
        """
        # 计算扩展后的角度数组
        expanded_angles, original_angle_indices, indices_left, indices_right = self.expand_angles(source_angles)
        expanded_left_indices = []
        expanded_right_indices = []
        expanded_ratios = []
        expanded_coeffs = []

        # 对扩展角度进行索引和比例计算
        for ang in expanded_angles:
            left_idx, right_idx, ratio, coeff = self.angle_to_index_and_value(ang, source_angles)
            expanded_left_indices.append(left_idx)
            expanded_right_indices.append(right_idx)
            expanded_ratios.append(ratio)
            expanded_coeffs.append(coeff)

        # 转为 numpy 数组
        expanded_left_indices = np.array(expanded_left_indices, dtype=int)
        expanded_right_indices = np.array(expanded_right_indices, dtype=int)
        expanded_ratios = np.array(expanded_ratios, dtype=float)
        expanded_coeffs = np.array(expanded_coeffs, dtype=float)

        return expanded_angles, expanded_left_indices, expanded_right_indices, expanded_ratios, expanded_coeffs, original_angle_indices

    def angle_to_index_and_value(self, angle, source_angles):
        """
        根据输入角度 angle 计算其对应的左索引、右索引和比例
        对输入角度进行周期性处理 (mod 360deg)
        如果映射后角度落在原区间内，则返回左索引、右索引、比例
        如果超出原区间，则返回 -1, -1, 0

        参数:
            angle         : 输入角度
            source_angles : 单调递增的原始角度数组

        返回:
            left_index  : 左侧索引
            right_index : 右侧索引
            ratio       : angle 在区间中的比例 (0.0 - 1.0)
        """

        source_angles_norm = source_angles - source_angles[0]
        angle_norm = angle - source_angles[0]
        valid_start = source_angles_norm[0]
        valid_end = source_angles_norm[-1]

        # 周期性处理
        angle_norm_mod = angle_norm % (np.pi*2)

        # 判断是否落在有效区间内
        if valid_start <= angle_norm_mod <= valid_end:
            # 找到插入位置
            insert_idx = np.searchsorted(source_angles_norm, angle_norm_mod)

            # 边界处理
            if insert_idx == 0:  # 小于最小值
                return 0, 1, 0, 1
            elif insert_idx == len(source_angles):  # 大于等于最大值
                return -2, -2, 1/2, 2

            # 左、右索引
            left_index = insert_idx - 1
            right_index = insert_idx

            # 左、右值
            left_value = source_angles[left_index]
            right_value = source_angles[right_index]

            # 计算比例
            ratio = (angle_norm_mod - left_value) / (right_value - left_value)

            return left_index, right_index, ratio, 1
        else:
            return -1, -1, 0.0, 1

    def expand_axis(self, source_angles, r_grid_flat, z_grid_flat):

        f_grid_flat_expand, original_indices, indices_left, indices_right = self.expand_angles(source_angles)

        # # 左扩展：映射到周期内，取末尾元素减去 N
        # r_left_extension = np.take(r_grid_flat, indices_left, mode='wrap')
        # z_left_extension = np.take(z_grid_flat, indices_left, mode='wrap')
        #
        # # 中间部分：原始区间保持不变
        # r_middle = r_grid_flat
        # z_middle = z_grid_flat
        #
        # # 右扩展：映射到周期内，取开头元素加上 N
        # r_right_extension = np.take(r_grid_flat, indices_right, mode='wrap')
        # z_right_extension = np.take(z_grid_flat, indices_right, mode='wrap')

        # 合并扩展区间
        r_grid_flat_expand = r_grid_flat
        z_grid_flat_expand = z_grid_flat

        return f_grid_flat_expand, r_grid_flat_expand, z_grid_flat_expand

    def expand_matrix(self, source_angles, Xmatrix, Ymatrix, Zmatrix):
        """
        对输入的 3D 坐标矩阵 Xmatrix, Ymatrix, Zmatrix 沿 axis=1 进行拓展，
        其中 Ymatrix 的扩展需要考虑周期性边界条件。

        参数:
            source_angles : 原始角度数组（用于定义周期性边界条件）
            Xmatrix      : X 方向的 3D 坐标矩阵
            Ymatrix      : Y 方向的 3D 坐标矩阵（需周期性扩展）
            Zmatrix      : Z 方向的 3D 坐标矩阵

        返回:
            extended_Xmatrix : 拓展后的 X 方向 3D 坐标矩阵
            extended_Ymatrix : 拓展后的 Y 方向 3D 坐标矩阵
            extended_Zmatrix : 拓展后的 Z 方向 3D 坐标矩阵
        """

        # 确定 Ymatrix 的最小值和最大值（用于计算周期性边界跨度）
        Ymin, Ymax = np.min(Ymatrix), np.max(Ymatrix)

        _, _, indices_left, indices_right = self.expand_angles(source_angles)

        # 左扩展：周期性延伸
        X_left_extension = np.take(Xmatrix, indices_left, axis=1, mode='wrap')
        Y_left_extension = np.take(Ymatrix, indices_left, axis=1, mode='wrap') - (Ymax - Ymin)
        Z_left_extension = np.take(Zmatrix, indices_left, axis=1, mode='wrap')

        # 中间部分：原始矩阵保持不变
        X_middle = Xmatrix
        Y_middle = Ymatrix
        Z_middle = Zmatrix

        # 右扩展：周期性延伸
        X_right_extension = np.take(Xmatrix, indices_right, axis=1, mode='wrap')
        Y_right_extension = np.take(Ymatrix, indices_right, axis=1, mode='wrap') + (Ymax - Ymin)
        Z_right_extension = np.take(Zmatrix, indices_right, axis=1, mode='wrap')

        # 合并扩展结果
        extended_Xmatrix = np.concatenate([X_left_extension, X_middle, X_right_extension], axis=1)
        extended_Ymatrix = np.concatenate([Y_left_extension, Y_middle, Y_right_extension], axis=1)
        extended_Zmatrix = np.concatenate([Z_left_extension, Z_middle, Z_right_extension], axis=1)

        return extended_Xmatrix, extended_Ymatrix, extended_Zmatrix

    def expand_data(self, source_angles, charge_distribution):
        """
        将 y_grid 扩展一倍，并利用扩展后的索引和比例扩展 Xmatrix, Ymatrix, Zmatrix, charge_distribution 沿 y 方向扩展一倍。

        参数:
            y_grid   : 原始 y 网格数组
            Xmatrix  : 原始 X 矩阵
            Ymatrix  : 原始 Y 矩阵
            Zmatrix  : 原始 Z 矩阵

        返回:
            expanded_y_grid : 扩展后的 y 网格数组
            Xmatrix_expand  : 沿 y 方向扩展后的 X 矩阵
            Ymatrix_expand  : 沿 y 方向扩展后的 Y 矩阵
            Zmatrix_expand  : 沿 y 方向扩展后的 Z 矩阵
        """

        # 1) 归一化 y_grid
        y_grid_norm = source_angles - source_angles[0]

        # 2) 通过 expand_data(y_grid_norm) 得到 expanded_angles, expanded_left_indices, expanded_right_indices, expanded_ratios
        expanded_angles, left_indices, right_indices, ratios, coeffs, original_angle_indices = self.get_expand_index(y_grid_norm)

        # 3) 构造一个零矩阵 zeros_matrix
        zeros_matrix = np.zeros((charge_distribution.shape[0], 1, charge_distribution.shape[2]))

        # 4)
        charge_distribution_append0 = np.append(charge_distribution, zeros_matrix, axis=1)

        # 5) 根据左索引、右索引和比例插值
        n_expanded = len(expanded_angles)  # 扩展后的角度数量
        charge_distribution_expand = np.zeros((charge_distribution.shape[0], n_expanded, charge_distribution.shape[2]))

        for i, (li, ri, ratio, coeff) in enumerate(zip(left_indices, right_indices, ratios, coeffs)):
            if li == -1 or ri == -1:  # 超出范围的点，填充零
                charge_distribution_expand[:, i, :] = 0
            else:  # 插值计算
                charge_distribution_expand[:, i, :] = ((1 - ratio) * charge_distribution_append0[:, li, :]
                                                       + ratio * charge_distribution_append0[:, ri, :]) * coeff

        return charge_distribution_expand, original_angle_indices

#
# @njit(parallel=True)
# def point_in_convex_polygon_binary_njit(polygon, points):
#     N = polygon.shape[0]
#     M = points.shape[0]
#     V0x = polygon[0, 0]
#     V0y = polygon[0, 1]
#     vecs = np.empty((N, 2), dtype=np.float64)
#     for i in range(N):
#         vecs[i, 0] = polygon[i, 0] - V0x
#         vecs[i, 1] = polygon[i, 1] - V0y
#     res = np.empty(M, dtype=np.bool_)
#     for j in prange(M):
#         px = points[j, 0]
#         py = points[j, 1]
#         vpx = px - V0x
#         vpy = py - V0y
#         if vecs[1, 0] * vpy - vecs[1, 1] * vpx < 0.0 or vecs[N - 1, 0] * vpy - vecs[N - 1, 1] * vpx > 0.0:
#             res[j] = False
#             continue
#         left_idx = 1
#         right_idx = N - 1
#         while right_idx - left_idx > 1:
#             mid = (left_idx + right_idx) // 2
#             if vecs[mid, 0] * vpy - vecs[mid, 1] * vpx >= 0.0:
#                 left_idx = mid
#             else:
#                 right_idx = mid
#         x1 = polygon[left_idx, 0] - px
#         y1 = polygon[left_idx, 1] - py
#         x2 = polygon[right_idx, 0] - px
#         y2 = polygon[right_idx, 1] - py
#         c = x1 * y2 - y1 * x2
#         res[j] = c >= 0.0
#     return res
#
#
# # @njit(parallel=True)
# @profile
# def polar_filter_and_convert(r, fi, phi_start, phi_end):
#     M = r.shape[0]
#     mask = np.zeros(M, dtype=np.bool_)
#     x = np.empty(M, dtype=np.float64)
#     y = np.empty(M, dtype=np.float64)
#
#     phi_start = phi_start % (2 * np.pi)
#     phi_end = phi_end % (2 * np.pi)
#
#     for i in prange(M):
#         ri = r[i]
#         fi_norm = fi[i] % (2 * np.pi)
#
#         if phi_start <= phi_end:
#             in_phi = phi_start <= fi_norm <= phi_end
#         else:
#             in_phi = (fi_norm >= phi_start) or (fi_norm <= phi_end)
#
#         if in_phi:
#             mask[i] = True
#             x[i] = ri * math.cos(fi_norm)
#             y[i] = ri * math.sin(fi_norm)
#
#     return x, y, mask

@njit(parallel=True)
def point_in_convex_polygon_batch(r, fi, polygon, x_min, x_max, y_min, y_max):
    """
    输入为极坐标 (r, fi)，输出每个点是否在 polygon 内（凸）
    """
    M = r.shape[0]
    N = polygon.shape[0]
    result = np.zeros(M, dtype=np.bool_)

    for i in prange(M):
        # 极坐标 -> 笛卡尔
        x = r[i] * np.cos(fi[i])
        y = r[i] * np.sin(fi[i])

        #快速排斥
        if x < x_min or x > x_max or y < y_min or y > y_max:
            continue

        # 全叉积判断
        prev_cross = 0.0
        inside = True
        for j in range(N):
            x0, y0 = polygon[j]
            x1, y1 = polygon[(j + 1) % N]

            dx1 = x1 - x0
            dy1 = y1 - y0
            dx2 = x - x0
            dy2 = y - y0
            cross = dx1 * dy2 - dy1 * dx2

            if j == 0:
                prev_cross = cross
            else:
                if cross * prev_cross < 0.0:
                    inside = False
                    break

        result[i] = inside

    return result


@njit(parallel=True)
def emap_kernel(r, fi,
                acc_Etot, theta_n,
                poly_x, poly_y,
                box_x_min, box_x_max,
                box_y_min, box_y_max,
                Er, Efi, Ez, Malloc_RF_shift, gap_index):
    """
    同上，对每个粒子判定是否落在凸多边形内
    若在内 → 计算 Δθ 并写 Er/Efi
    若不在 → Er/Efi 置 0
    """
    Nv = poly_x.shape[0]
    Np = r.shape[0]

    for k in prange(Np):

        # -------- 极坐标 → (x,y) ----------
        x = r[k] * np.cos(fi[k])
        y = r[k] * np.sin(fi[k])

        # -------- 包围盒快速排斥 ----------
        if (x < box_x_min or x > box_x_max or
            y < box_y_min or y > box_y_max):
            Er[k]  = 0.0
            Efi[k] = 0.0
            Ez[k] = 0.0
            continue

        # -------- 凸多边形同侧法 ----------
        inside = True
        prev   = 0.0
        for j in range(Nv):
            x0 = poly_x[j];       y0 = poly_y[j]
            x1 = poly_x[(j+1)%Nv];y1 = poly_y[(j+1)%Nv]

            cross = (x1 - x0)*(y - y0) - (y1 - y0)*(x - x0)
            if j == 0:
                prev = cross
            elif cross * prev < 0.0:   # 符号变 → 在边界外
                inside = False
                break

        if inside:
            dtheta   = theta_n - fi[k]
            Er[k]    = acc_Etot * np.cos(dtheta)
            Efi[k]   = acc_Etot * np.sin(dtheta)
            Ez[k] = 0.0
            Malloc_RF_shift[k] = gap_index
        else:
            Er[k]  = 0.0
            Efi[k] = 0.0
            Ez[k] = 0.0
            Malloc_RF_shift[k] = 0

# @njit(parallel=True)
# def emap_kernel(r, fi,
#                 acc_Etot, theta_n,
#                 poly_x, poly_y,
#                 box_x_min, box_x_max,
#                 box_y_min, box_y_max,
#                 Er, Efi, Ez):
#     """
#     · 同上，对每个粒子判定是否落在凸多边形内
#     · 若在内 → 计算 Δθ 并写 Er/Efi
#     · 若不在 → Er/Efi 置 0
#     """
#     Nv = poly_x.shape[0]
#     Np = r.shape[0]
#
#     for k in prange(Np):
#
#         # -------- 极坐标 → (x,y) ----------
#         x = r[k] * np.cos(fi[k])
#         y = r[k] * np.sin(fi[k])
#
#         # -------- 包围盒快速排斥 ----------
#         if (x < box_x_min or x > box_x_max or
#             y < box_y_min or y > box_y_max):
#             Er[k]  = 0.0
#             Efi[k] = 0.0
#             Ez[k] = 0.0
#             continue
#
#         # -------- 凸多边形同侧法 ----------
#         inside = True
#         prev   = 0.0
#         for j in range(Nv):
#             x0 = poly_x[j];       y0 = poly_y[j]
#             x1 = poly_x[(j+1)%Nv];y1 = poly_y[(j+1)%Nv]
#
#             cross = (x1 - x0)*(y - y0) - (y1 - y0)*(x - x0)
#             if j == 0:
#                 prev = cross
#             elif cross * prev < 0.0:   # 符号变 → 在边界外
#                 inside = False
#                 break
#
#         if inside:
#             dtheta   = theta_n - fi[k]
#             Er[k]    = acc_Etot * np.cos(dtheta)
#             Efi[k]   = acc_Etot * np.sin(dtheta)
#             Ez[k] = 0.0
#         else:
#             Er[k]  = 0.0
#             Efi[k] = 0.0
#             Ez[k] = 0.0
