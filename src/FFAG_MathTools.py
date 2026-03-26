import numpy as np
import math
import copy
from numba import njit, types, prange
from numba.np.extensions import cross2d
from mpi4py import MPI


class FFAG_interpolation:
    def __init__(self):
        pass

    def find_two_points(self, X, xi):
        """
        Find the nearest two sample points around xi and return their positions
        Input: X: an array of sample point positions (sorted in ascending order)
               xi: the target value
        Output: left_point, right_point: two positions of the closest points
                flag: a flag indicating whether xi lies within the range of X (1) or not (0)
                Note: left_point <= xi <= right_point
        """
        delta_x = X[1] - X[0]
        idx = math.floor((xi - X[0]) / delta_x)
        if idx < 0:
            flag = 1
            idx_array = np.array((0, 1))
        elif idx >= len(X) - 2:
            flag = 1
            idx_array = np.array((len(X) - 2, len(X) - 1))
        else:
            flag = 0
            idx_array = np.array((idx, idx + 1))
        return idx_array, flag

    def find_two_points_non_uniform(self, X, xi):
        """
        Find the nearest two sample points around xi and return their positions
        Input: X: an array of sample point positions (sorted in ascending order)
               xi: the target value
        Output: left_point, right_point: two positions of the closest points
                flag: a flag indicating whether xi lies within the range of X (1) or not (0)
                Note: left_point <= xi <= right_point
        """
        right_point = np.searchsorted(X, xi, side='right')
        left_point = right_point - 1
        # Check if xi is out of the range of X
        flag = 0
        if left_point < 0:
            idx_array = np.array((0, 1))
            flag = 1
        elif right_point > len(X) - 1:
            idx_array = np.array((len(X) - 2, len(X) - 1))
            flag = 1
        else:
            idx_array = np.array([left_point, right_point])

        return idx_array, flag

    def find_two_points_non_uniform_vect(self, X, xi):

        right_point = np.searchsorted(X, xi, side='right')
        left_point = right_point - 1

        # Check if xi is out of the range of X
        idx_array_0, idx_array_1, flag = (
            copy.deepcopy(left_point), copy.deepcopy(right_point), np.zeros_like(xi, dtype=bool))

        # if left_point < 0
        flag_left_point_less_0 = left_point < 0
        idx_array_0[flag_left_point_less_0] = 0
        idx_array_1[flag_left_point_less_0] = 1
        flag[flag_left_point_less_0] = 1

        # if right_right_point > len(X) - 1
        flag_right_point_larger_len = right_point > len(X) - 1
        idx_array_0[flag_right_point_larger_len] = len(X) - 2
        idx_array_1[flag_right_point_larger_len] = len(X) - 1
        flag[flag_right_point_larger_len] = 1

        idx_array = np.vstack((idx_array_0, idx_array_1))

        return idx_array, flag

    def find_four_points_non_uniform(self, X, xi):
        """
        Find the nearest four sample points around xi and return their positions
        Input: X: an array of sample point positions (sorted in ascending order)
               xi: the target value
        Output: left_point, right_point: two positions of the closest points
                flag: a flag indicating whether xi lies within the range of X (1) or not (0)
                Note: left_point <= xi <= right_point
        """
        right_point = np.searchsorted(X, xi, side='right')
        left_point = right_point - 1
        right_right_point = right_point + 1
        left_left_point = left_point - 1
        # Check if xi is out of the range of X
        flag = 0
        if left_left_point < 0:
            idx_array = np.array((0, 1, 2, 3))
            flag = 1
        elif right_right_point > len(X) - 1:
            idx_array = np.array((len(X) - 4, len(X) - 3, len(X) - 2, len(X) - 1))
            flag = 1
        else:
            idx_array = np.array((left_left_point, left_point, right_point, right_right_point))
        return idx_array, flag

    def find_four_points_non_uniform_vector(self, X, xi):
        """
        Find the nearest 4 sample points around xi and return their positions.
        This is a vectorized function.
        """
        right_point = np.searchsorted(X, xi, side='right')
        left_point = right_point - 1
        right_right_point = right_point + 1
        left_left_point = left_point - 1
        # Check if xi is out of the range of X
        flag_left_point_less_than_0 = left_left_point < 0
        flag_right_point_larger_than_len = right_right_point > len(X) - 1
        idx_array_0, idx_array_1, idx_array_2, idx_array_3, flag = \
            copy.deepcopy(left_left_point), copy.deepcopy(left_point), \
                copy.deepcopy(right_point), copy.deepcopy(right_right_point), \
                np.zeros_like(xi, dtype=bool)

        # if left_left_point < 0
        idx_array_0[flag_left_point_less_than_0] = 0
        idx_array_1[flag_left_point_less_than_0] = 1
        idx_array_2[flag_left_point_less_than_0] = 2
        idx_array_3[flag_left_point_less_than_0] = 3
        flag[flag_left_point_less_than_0] = 1

        # if right_right_point > len(X) - 1
        idx_array_0[flag_right_point_larger_than_len] = len(X) - 4
        idx_array_1[flag_right_point_larger_than_len] = len(X) - 3
        idx_array_2[flag_right_point_larger_than_len] = len(X) - 2
        idx_array_3[flag_right_point_larger_than_len] = len(X) - 1
        flag[flag_right_point_larger_than_len] = 1

        flag_left_point_equal_0 = left_point == 0
        flag_left_point_equal_minus2 = left_point == len(X) - 2
        flag[flag_left_point_equal_0] = 0
        flag[flag_left_point_equal_minus2] = 0

        idx_array = np.vstack((idx_array_0, idx_array_1, idx_array_2, idx_array_3))

        return idx_array, flag

    def linear_interpolation(self, X, Y, NonUniform=True):
        def interpolator(xi):
            if NonUniform:
                idx_array, OutRangeFlag = self.find_two_points_non_uniform(X, xi)
            else:
                idx_array, OutRangeFlag = self.find_two_points(X, xi)
            x = X[idx_array]
            y = Y[idx_array]
            x0, x1, y0, y1 = x[0], x[1], y[0], y[1]
            return y0 + (xi - x0) / (x1 - x0) * (y1 - y0)

        return interpolator

    def linear_interpolation_vect(self, X, Y, NonUniform=True):
        def interpolator(xi):
            if NonUniform:
                idx_array, OutRangeFlag = self.find_two_points_non_uniform_vect(X, xi)
            else:
                idx_array, OutRangeFlag = self.find_two_points(X, xi)
            x = X[idx_array]
            y = Y[idx_array]
            x0, x1, y0, y1 = x[0], x[1], y[0], y[1]
            return y0 + (xi - x0) / (x1 - x0) * (y1 - y0)

        return interpolator

    def derivate_interpolation(self, X, Y):
        def derivate_interpolator(xi):
            idx_array, OutRangeFlag = self.find_four_points_non_uniform_vector(X, xi)
            x = X[idx_array]
            y = Y[idx_array]
            coefficients = np.polyfit(x, y, deg=3)
            a3, a2, a1, a0 = coefficients[0], coefficients[1], coefficients[2], coefficients[3]
            return 3 * a3 * xi ** 2 + 2 * a2 * xi + a1

        return derivate_interpolator

    def find_four_points(self, X, xi):
        """
        Find the four sample points around xi and return them
        Input: X: array of sample point positions
               xi: the interpolation point
        Output: arrays of X positions and values of four closest sample points
                flag: an indicator in case xi falls outside range of X values (1) or not (0)
        """
        delta_x = X[1] - X[0]
        idx = math.floor((xi - X[0]) / delta_x)
        if idx < 1:
            flag = 1
            idx_array = np.array((0, 1, 2, 3))
        elif idx >= len(X) - 2:
            flag = 1
            # idx = len(X) - 2
            idx_array = np.array((len(X) - 2 - 2, len(X) - 2 - 1, len(X) - 2, len(X) - 2 + 1))
        else:
            flag = 0
            idx_array = np.array((idx - 1, idx, idx + 1, idx + 2))

        if idx == 0 or idx == len(X) - 2:
            flag = 0

        return idx_array, flag

    def polynomial_interpolation(self, X, Y, xi):
        """
        Polynomial interpolation function
        Input: X, Y: arrays of sample point positions and values
               xi: the interpolation point
        Output: yi: interpolated value at xi
                OutRangeFlag: an indicator in case xi falls outside range of X values (1) or not (0)
        """
        idx_array, OutRangeFlag = self.find_four_points(X, xi)
        x = X[idx_array]
        y = Y[idx_array]
        x_i_0, x_i_1, x_i_2, x_i_3 = xi - x[0], xi - x[1], xi - x[2], xi - x[3]
        x_0_1, x_0_2, x_0_3 = x[0] - x[1], x[0] - x[2], x[0] - x[3]
        x_1_2, x_1_3 = x[1] - x[2], x[1] - x[3]
        x_2_3 = x[2] - x[3]
        # L0 = y[0] * (xi - x[1]) * (xi - x[2]) * (xi - x[3]) / (x[0] - x[1]) / (x[0] - x[2]) / (x[0] - x[3])
        # L1 = y[1] * (xi - x[0]) * (xi - x[2]) * (xi - x[3]) / (x[1] - x[0]) / (x[1] - x[2]) / (x[1] - x[3])
        # L2 = y[2] * (xi - x[0]) * (xi - x[1]) * (xi - x[3]) / (x[2] - x[0]) / (x[2] - x[1]) / (x[2] - x[3])
        # L3 = y[3] * (xi - x[0]) * (xi - x[1]) * (xi - x[2]) / (x[3] - x[0]) / (x[3] - x[1]) / (x[3] - x[2])
        L0 = y[0] * x_i_1 * x_i_2 * x_i_3 / x_0_1 / x_0_2 / x_0_3
        L1 = -1 * y[1] * x_i_0 * x_i_2 * x_i_3 / x_0_1 / x_1_2 / x_1_3
        L2 = y[2] * x_i_0 * x_i_1 * x_i_3 / x_0_2 / x_1_2 / x_2_3
        L3 = -1 * y[3] * x_i_0 * x_i_1 * x_i_2 / x_0_3 / x_1_3 / x_2_3
        Value = L0 + L1 + L2 + L3
        return Value, OutRangeFlag

    def polynomial_interpolation_2D(self, X, Y, Z, Xi, Yi):
        """
        2D polynomial interpolation function
        Input: X, Y: arrays of sample point positions in the vertical(r_axis) and horizontal(fi_axis) directions, respectively
               Z: 2D array of function values at the sample points
               Xi, Yi: coordinates of the interpolation point to be evaluated
        Output: Value: interpolated function value at the point (Xi, Yi)
                OutRangeFlag: an indicator in case (Xi, Yi) falls outside the range of sample point positions
                (1) or not (0)
        """
        # First, use the find_four_points function to find the positions of the four closest sample points in the
        # vertical direction around the given interpolation point with horizontal position xi
        idx_y, flag = self.find_four_points(Y, Yi)

        # Use polynomial_interpolation function to perform 1D interpolation on the four closest sample points in the
        # horizontal direction around the given interpolation point with vertical position Yi, and obtain the values of these
        # four interpolated points
        SampleY0, SampleY1, SampleY2, SampleY3 = Z[:, idx_y[0]], Z[:, idx_y[1]], Z[:, idx_y[2]], Z[:, idx_y[3]]
        SampleX0, OutRangeFlag0 = self.polynomial_interpolation(X, SampleY0, Xi)
        SampleX1, OutRangeFlag1 = self.polynomial_interpolation(X, SampleY1, Xi)
        SampleX2, OutRangeFlag2 = self.polynomial_interpolation(X, SampleY2, Xi)
        SampleX3, OutRangeFlag3 = self.polynomial_interpolation(X, SampleY3, Xi)

        # Combine the values of the four interpolated points into a new array SampleX
        SampleX = np.array([SampleX0, SampleX1, SampleX2, SampleX3])

        # Use polynomial_interpolation function to perform 1D interpolation on the four closest sample points in the
        # vertical direction around the given interpolation point with horizontal position xi, using SampleX as the values.
        # Obtain the value of the final interpolation point
        Value, _ = self.polynomial_interpolation(
            np.array([Y[idx_y[0]], Y[idx_y[1]], Y[idx_y[2]], Y[idx_y[3]]]), SampleX, Yi)
        OutRangeFlag = flag + OutRangeFlag0 + OutRangeFlag1 + OutRangeFlag2 + OutRangeFlag3
        # Return the value of the final interpolation point and a flag indicating whether the interpolation point is within
        # the range of sample point positions or not
        return Value, OutRangeFlag

    def Lagrange_interp_2D_vect(self, X, Y, Z, Xi, Yi):
        """
        Input: X, Y: arrays of sample point positions in the vertical(r_axis) and horizontal(fi_axis) directions, respectively
               Z: 2D array of function values at the sample points
               Xi, Yi: coordinates of the interpolation point to be evaluated
        """
        idx_array_x, flag_x = self.find_four_points_non_uniform_vector(X, Xi)
        idx_array_y, flag_y = self.find_four_points_non_uniform_vector(Y, Yi)
        OutRangeFlag = np.logical_or(flag_x, flag_y)
        # shape of idx_array_x, idx_array_y: 4 * nSamplePoints
        nSamplePoints = np.size(Xi, 0)

        xs, ys = X[idx_array_x], Y[idx_array_y]  # x and y coordinates of the surrounding points
        idx_grid_x, idx_grid_y = self.My3DMeshgrid(idx_array_x, idx_array_y)
        zs = Z[idx_grid_x, idx_grid_y]

        Lx, Ly, Lxy = np.zeros((4, nSamplePoints)), np.zeros((4, nSamplePoints)), np.zeros((4, 4, nSamplePoints))

        Lx[0, :] = (Xi - xs[1, :]) * (Xi - xs[2, :]) * (Xi - xs[3, :]) / (xs[0, :] - xs[1, :]) / (
                xs[0, :] - xs[2, :]) / (xs[0, :] - xs[3, :])
        Lx[1, :] = (Xi - xs[0, :]) * (Xi - xs[2, :]) * (Xi - xs[3, :]) / (xs[1, :] - xs[0, :]) / (
                xs[1, :] - xs[2, :]) / (xs[1, :] - xs[3, :])
        Lx[2, :] = (Xi - xs[0, :]) * (Xi - xs[1, :]) * (Xi - xs[3, :]) / (xs[2, :] - xs[0, :]) / (
                xs[2, :] - xs[1, :]) / (xs[2, :] - xs[3, :])
        Lx[3, :] = (Xi - xs[0, :]) * (Xi - xs[1, :]) * (Xi - xs[2, :]) / (xs[3, :] - xs[0, :]) / (
                xs[3, :] - xs[1, :]) / (xs[3, :] - xs[2, :])

        Ly[0, :] = (Yi - ys[1, :]) * (Yi - ys[2, :]) * (Yi - ys[3, :]) / (ys[0, :] - ys[1, :]) / (
                ys[0, :] - ys[2, :]) / (ys[0, :] - ys[3, :])
        Ly[1, :] = (Yi - ys[0, :]) * (Yi - ys[2, :]) * (Yi - ys[3, :]) / (ys[1, :] - ys[0, :]) / (
                ys[1, :] - ys[2, :]) / (ys[1, :] - ys[3, :])
        Ly[2, :] = (Yi - ys[0, :]) * (Yi - ys[1, :]) * (Yi - ys[3, :]) / (ys[2, :] - ys[0, :]) / (
                ys[2, :] - ys[1, :]) / (ys[2, :] - ys[3, :])
        Ly[3, :] = (Yi - ys[0, :]) * (Yi - ys[1, :]) * (Yi - ys[2, :]) / (ys[3, :] - ys[0, :]) / (
                ys[3, :] - ys[1, :]) / (ys[3, :] - ys[2, :])

        for idx_Lx in range(0, 4):
            for idx_Ly in range(0, 4):
                Lxy[idx_Lx, idx_Ly, :] = Lx[idx_Lx, :] * Ly[idx_Ly, :]

        interp_value = np.sum(Lxy * zs, axis=(0, 1))
        return interp_value, OutRangeFlag

    def My3DMeshgrid(self, x_2D, y_2D):
        # shape of x_2D and y_2D: D*N, N=number of sample points
        # D = dimension of the coordinate of the sample points
        DimensionSamplePoints = np.size(x_2D, 0)
        x_expand, y_expand = x_2D[:, np.newaxis, :], y_2D[np.newaxis, :, :]
        # Create a meshgrid for x
        X = np.tile(x_expand, (1, DimensionSamplePoints, 1))
        # Create a meshgrid for y
        Y = np.tile(y_expand, (DimensionSamplePoints, 1, 1))
        # X[:,:,0] is same in the row, Y[:,:,0] is same in the column
        return X, Y

    def My2p5DInterp(self, Matrix3D, axis_dim0, axis_dim1):
        # axis0: steps of azimuth
        # axis1: PID
        # axis2: coordinates
        # 在axis0, axis1上进行插值
        LenDim3 = np.size(Matrix3D, 2)

        def InterpFunc2D(fi0, PID_n):
            idx_array_x, flag_x = self.find_two_points_non_uniform_vect(axis_dim0, fi0)
            idx_array_y, flag_y = self.find_two_points_non_uniform_vect(axis_dim1, PID_n)
            OutRangeFlag = np.logical_or(flag_x, flag_y)
            idx_x0, idx_x1 = idx_array_x[0, :], idx_array_x[1, :]
            idx_y0, idx_y1 = idx_array_y[0, :], idx_array_y[1, :]

            x0, x1 = axis_dim0[idx_x0], axis_dim0[idx_x1]
            y0, y1 = axis_dim1[idx_y0], axis_dim1[idx_y1]

            z00 = Matrix3D[idx_x0, idx_y0, :]  # at x0,y0
            z01 = Matrix3D[idx_x0, idx_y1, :]  # at x0,y1
            z10 = Matrix3D[idx_x1, idx_y0, :]  # at x1,y0
            z11 = Matrix3D[idx_x1, idx_y1, :]  # at x1,y1
            L00 = (fi0 - x1) * (PID_n - y1) / (x0 - x1) / (y0 - y1)
            L01 = (fi0 - x1) * (PID_n - y0) / (x0 - x1) / (y1 - y0)
            L10 = (fi0 - x0) * (PID_n - y1) / (x1 - x0) / (y0 - y1)
            L11 = (fi0 - x0) * (PID_n - y0) / (x1 - x0) / (y1 - y0)
            L00_Ext = np.tile(L00, (LenDim3, 1)).T
            L01_Ext = np.tile(L01, (LenDim3, 1)).T
            L10_Ext = np.tile(L10, (LenDim3, 1)).T
            L11_Ext = np.tile(L11, (LenDim3, 1)).T

            # fi0 equals to x0, PID_n equals to y
            result = L00_Ext * z00 + L01_Ext * z01 + L10_Ext * z10 + L11_Ext * z11

            return result, OutRangeFlag

        return InterpFunc2D

    def My1p5DInterp(self, X_values, Y_arrs):
        # x是一维数组，y是二维矩阵，x中的每一个数对应y的一行
        def Interp1p5D(xi):
            idx_array, flag = self.find_two_points_non_uniform_vect(X_values, xi)
            x0, x1 = X_values[idx_array[0]], X_values[idx_array[1]]
            y0, y1 = Y_arrs[idx_array[0], :], Y_arrs[idx_array[1], :]
            y = (xi-x1)/(x0-x1)*y0 + (xi-x0)/(x1-x0)*y1
            return y, flag

        return Interp1p5D


class FFAG_derivative:
    def __init__(self):
        pass

    def myderivative(self, x, y):
        # First-order derivative in difference form
        # test code:
        # Declare an array deriv of zeros of length n
        n = len(x)
        deriv = np.zeros(n)

        # Calculate first derivative at interior points
        # using central divided differences
        for i in range(1, n - 1):
            deriv[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
        # Calculate first derivative at end points using
        # forward and backward divided differences
        deriv[0] = (y[1] - y[0]) / (x[1] - x[0])
        deriv[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])

        return deriv


    def myderivative_smooth(self, x, y):
        # Declare an array deriv of zeros of length n
        n = len(x)
        deriv = np.zeros(n)

        # Calculate first derivative at interior points
        # using central divided differences
        for i in range(1, n - 1):
            deriv[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
        # Calculate first derivative at end points using
        # forward and backward divided differences
        deriv[0] = 2 * deriv[1] - deriv[2]
        deriv[n - 1] = 2 * deriv[n - 2] - deriv[n - 3]

        return deriv

    def mysecondderivative(self, x, y):
        # Second-order derivative in difference form
        n = len(x)
        deriv2 = np.zeros(n)
        deriv2_interp_on_boundary = np.zeros(n)

        for i in range(1, n - 1):
            h1 = x[i] - x[i - 1]
            h2 = x[i + 1] - x[i]
            h_sum = h1 + h2

            deriv2[i] = 2 * ((y[i + 1] - y[i]) / (h2 * h_sum) - (y[i] - y[i - 1]) / (h1 * h_sum))
            deriv2_interp_on_boundary[i] = deriv2[i]

        h0 = x[1] - x[0]
        h1 = x[2] - x[1]
        h_sum = h0 + h1
        deriv2[0] = 2 * ((y[2] - y[1]) / (h1 * h_sum) - (y[1] - y[0]) / (h0 * h_sum))
        deriv2_interp_on_boundary[0] = 2 * deriv2[1] - deriv2[2]

        hn_minus_2 = x[n - 2] - x[n - 3]
        hn_minus_1 = x[n - 1] - x[n - 2]
        h_sum = hn_minus_2 + hn_minus_1
        deriv2[n - 1] = 2 * (
                (y[n - 1] - y[n - 2]) / (hn_minus_1 * h_sum) - (y[n - 2] - y[n - 3]) / (hn_minus_2 * h_sum))
        deriv2_interp_on_boundary[n - 1] = 2 * deriv2[n - 2] - deriv2[n - 3]

        return deriv2_interp_on_boundary, deriv2

    def MyMatrixExtend(self, arr, m, RowFlag=True):
        # n = np.size(arr)
        if RowFlag:
            arr_new = np.tile(arr, (m, 1))
        else:
            arr_new = np.tile(arr, (m, 1)).T
        return arr_new

    def myderv_matrix(self, x, matrixY, RowFlag=True):
        # First-order derivative in matrix form
        rownum, colnum = np.size(matrixY, 0), np.size(matrixY, 1)
        dMdX = np.zeros_like(matrixY)
        if RowFlag:
            # matrixX = np.tile(x, (rownum, 1))
            matrixX = self.MyMatrixExtend(x, rownum, RowFlag)
            dMdX[:, 1:-1] = (matrixY[:, 2:] - matrixY[:, 0:-2]) / (matrixX[:, 2:] - matrixX[:, 0:-2])
            # At the start point and the end point, using interpolation
            dMdX[:, 0] = 2 * dMdX[:, 1] - dMdX[:, 2]
            dMdX[:, -1] = 2 * dMdX[:, -2] - dMdX[:, -3]
        else:
            # matrixX = np.tile(x, (rownum, 1)).T
            matrixX = self.MyMatrixExtend(x, colnum, RowFlag)
            dMdX[1:-1, :] = (matrixY[2:, :] - matrixY[0:-2, :]) / (matrixX[2:, :] - matrixX[0:-2, :])
            dMdX[0, :] = 2 * dMdX[1, :] - dMdX[2, :]
            dMdX[-1, :] = 2 * dMdX[-2, :] - dMdX[-3, :]

        return dMdX


class FFAG_Algorithm:
    def __init__(self):
        pass

    def BiSection(self, x_left, x_right, func, paras=None):
        x_mid, y_mid = float("nan"), float("nan")
        for k in range(0, 100):
            x_mid = (x_left + x_right) / 2
            y_left, y_right, y_mid = func(x_left, paras), func(x_right, paras), func(x_mid, paras)
            if y_mid * y_left < 0:
                x_right = x_mid
            else:
                x_left = x_mid
            # print("ymid=",y_mid)
        return x_mid, y_mid

    def nelder_mead(self, obj_func, initial_simplex, ObjFuncPara, max_iter=100, alpha=1.0, gamma=2.0, rho=0.5,
                    eps=1e-5):
        n = initial_simplex.shape[1]  # dimension
        func_values = np.zeros(n + 1)  # target function value
        vertex_flag = np.zeros(n + 1)  # if flag = 0, the corresponding vertex should be updated
        n_index = np.arange(0, n+1)
        # n_indexLocal = FFAG_MPI().DivideVariables(n_index)
        for i in n_index:
            func_values[i] = obj_func(initial_simplex[i], ObjFuncPara)
            vertex_flag[i] = 1

        for j in range(max_iter):
            indices = np.argsort(func_values)  # sort
            best, second_best, worst = indices[0], indices[1], indices[-1]
            # if rank==0:
            print("j=", j, "best=", func_values[best])
            print("j=", j, "worst=", func_values[worst])
            if func_values[best] < 1:
                break
            best_x, second_best_x, worst_x = initial_simplex[best], initial_simplex[second_best], initial_simplex[worst]
            best_FuncValue, second_best_FuncValue, worst_FuncValue = func_values[best], func_values[second_best], \
                func_values[worst]
            center = np.mean(initial_simplex[indices[:-1]], axis=0)
            # reflect
            reflect_x = center + alpha * (center - worst_x)
            reflect_FuncValue = obj_func(reflect_x, ObjFuncPara)
            if reflect_FuncValue < second_best_FuncValue and \
                    reflect_FuncValue >= best_FuncValue:
                initial_simplex[worst] = reflect_x
                # print('reflect')
                vertex_flag[worst] = 0
            # expand
            elif reflect_FuncValue < best_FuncValue:
                expand_x = center + gamma * (reflect_x - center)
                expand_FuncValue = obj_func(expand_x, ObjFuncPara)
                if expand_FuncValue < best_FuncValue:
                    initial_simplex[worst] = expand_x
                    # print('expand')
                    vertex_flag[worst] = 0
                else:
                    initial_simplex[worst] = reflect_x
                    # print('reflect')
                    vertex_flag[worst] = 0
            # contract
            else:
                contract_x = center + rho * (worst_x - center)
                contract_FuncValue = obj_func(contract_x, ObjFuncPara)
                if contract_FuncValue < worst_FuncValue:
                    initial_simplex[worst] = contract_x
                    # print('contract1')
                    vertex_flag[worst] = 0
                else:  # contract
                    initial_simplex[indices[1:]] = center + (initial_simplex[indices[1:]] - center) / 2
                    # print('contract2')
                    vertex_flag[indices[1:]] = 0
            # calculate eps
            center = np.mean(initial_simplex, axis=0)
            range_ = np.max(np.abs(initial_simplex - center))
            # if range_ < eps:
            #     break
            for i in range(n + 1):
                if vertex_flag[i] == 0:
                    func_values[i] = obj_func(initial_simplex[i], ObjFuncPara)

        index_min = np.argmin(func_values)
        return initial_simplex[index_min]


class FFAG_ANN():
    def __init__(self):
        pass

class FFAG_BasicTool:
    def __init__(self):
        pass

    def mod2zero(self, x, y):
        mod_value = x - (x/y).astype(int)*y
        return mod_value


# numba optimized functions
@njit
def find_four_points_non_uniform_vector(X, xi):
    right_point = np.searchsorted(X, xi, side='right')
    left_point = right_point - 1
    right_right_point = right_point + 1
    left_left_point = left_point - 1

    flag_left_point_less_than_0 = left_left_point < 0
    flag_right_point_larger_than_len = right_right_point > len(X) - 1
    idx_array_0, idx_array_1, idx_array_2, idx_array_3, flag = \
        left_left_point, left_point, right_point, right_right_point, \
        np.zeros_like(xi)

    idx_array_0[flag_left_point_less_than_0] = 0
    idx_array_1[flag_left_point_less_than_0] = 1
    idx_array_2[flag_left_point_less_than_0] = 2
    idx_array_3[flag_left_point_less_than_0] = 3
    flag[flag_left_point_less_than_0] = 1

    idx_array_0[flag_right_point_larger_than_len] = len(X) - 4
    idx_array_1[flag_right_point_larger_than_len] = len(X) - 3
    idx_array_2[flag_right_point_larger_than_len] = len(X) - 2
    idx_array_3[flag_right_point_larger_than_len] = len(X) - 1
    flag[flag_right_point_larger_than_len] = 1

    flag_left_point_equal_0 = left_point == 0
    flag_left_point_equal_minus2 = left_point == len(X) - 2
    flag[flag_left_point_equal_0] = 0
    flag[flag_left_point_equal_minus2] = 0

    idx_array = np.vstack((idx_array_0, idx_array_1, idx_array_2, idx_array_3))

    return idx_array, flag


@njit
def Interp_DoSum(xs, ys, zs, Xi, Yi):

    Lx = np.zeros((4, len(Xi)))
    Ly = np.zeros((4, len(Yi)))
    Lxy = np.zeros((4, 4, len(Xi)))

    Lx[0, :] = (Xi - xs[1, :]) * (Xi - xs[2, :]) * (Xi - xs[3, :]) / (xs[0, :] - xs[1, :]) / (
            xs[0, :] - xs[2, :]) / (xs[0, :] - xs[3, :])
    Lx[1, :] = (Xi - xs[0, :]) * (Xi - xs[2, :]) * (Xi - xs[3, :]) / (xs[1, :] - xs[0, :]) / (
            xs[1, :] - xs[2, :]) / (xs[1, :] - xs[3, :])
    Lx[2, :] = (Xi - xs[0, :]) * (Xi - xs[1, :]) * (Xi - xs[3, :]) / (xs[2, :] - xs[0, :]) / (
            xs[2, :] - xs[1, :]) / (xs[2, :] - xs[3, :])
    Lx[3, :] = (Xi - xs[0, :]) * (Xi - xs[1, :]) * (Xi - xs[2, :]) / (xs[3, :] - xs[0, :]) / (
            xs[3, :] - xs[1, :]) / (xs[3, :] - xs[2, :])

    Ly[0, :] = (Yi - ys[1, :]) * (Yi - ys[2, :]) * (Yi - ys[3, :]) / (ys[0, :] - ys[1, :]) / (
            ys[0, :] - ys[2, :]) / (ys[0, :] - ys[3, :])
    Ly[1, :] = (Yi - ys[0, :]) * (Yi - ys[2, :]) * (Yi - ys[3, :]) / (ys[1, :] - ys[0, :]) / (
            ys[1, :] - ys[2, :]) / (ys[1, :] - ys[3, :])
    Ly[2, :] = (Yi - ys[0, :]) * (Yi - ys[1, :]) * (Yi - ys[3, :]) / (ys[2, :] - ys[0, :]) / (
            ys[2, :] - ys[1, :]) / (ys[2, :] - ys[3, :])
    Ly[3, :] = (Yi - ys[0, :]) * (Yi - ys[1, :]) * (Yi - ys[2, :]) / (ys[3, :] - ys[0, :]) / (
            ys[3, :] - ys[1, :]) / (ys[3, :] - ys[2, :])

    for i in range(4):
        for j in range(4):
            Lxy[i, j, :] = Lx[i, :] * Ly[j, :]

    return Lxy * zs


@njit
def Interp_DoSum_Bilinear(xs, ys, zs, Xi, Yi):

    Lx = np.zeros((2, len(Xi)))
    Ly = np.zeros((2, len(Yi)))
    Lxy = np.zeros((2, 2, len(Xi)))

    Lx[0, :] = (Xi - xs[1, :]) / (xs[0, :] - xs[1, :])
    Lx[1, :] = (Xi - xs[0, :]) / (xs[1, :] - xs[0, :])

    Ly[0, :] = (Yi - ys[1, :]) / (ys[0, :] - ys[1, :])
    Ly[1, :] = (Yi - ys[0, :]) / (ys[1, :] - ys[0, :])

    for i in range(2):
        for j in range(2):
            Lxy[i, j, :] = Lx[i, :] * Ly[j, :]

    return Lxy * zs

@njit
def Interp_DoSum_new(xs, ys, zs, Xi, Yi):
    Lx = np.zeros((4, len(Xi)))
    Ly = np.zeros((4, len(Yi)))
    Lxy = np.zeros((4, 4, len(Xi)))

    # 预计算 xs 的差值
    xs_01 = xs[0, :] - xs[1, :]
    xs_02 = xs[0, :] - xs[2, :]
    xs_03 = xs[0, :] - xs[3, :]
    xs_12 = xs[1, :] - xs[2, :]
    xs_13 = xs[1, :] - xs[3, :]
    xs_23 = xs[2, :] - xs[3, :]

    # 预计算 Xi 与 xs 的差值
    X_i0 = Xi - xs[0, :]
    X_i1 = Xi - xs[1, :]
    X_i2 = Xi - xs[2, :]
    X_i3 = Xi - xs[3, :]

    # 计算 Lx
    Lx[0, :] = X_i1 * X_i2 * X_i3 / (xs_01 * xs_02 * xs_03)
    Lx[1, :] = -X_i0 * X_i2 * X_i3 / (xs_01 * xs_12 * xs_13)  # 注意此处为负数
    Lx[2, :] = X_i0 * X_i1 * X_i3 / (xs_02 * xs_12 * xs_23)
    Lx[3, :] = -X_i0 * X_i1 * X_i2 / (xs_03 * xs_13 * xs_23)  # 注意此处为负数

    # 预计算 ys 的差值
    ys_01 = ys[0, :] - ys[1, :]
    ys_02 = ys[0, :] - ys[2, :]
    ys_03 = ys[0, :] - ys[3, :]
    ys_12 = ys[1, :] - ys[2, :]
    ys_13 = ys[1, :] - ys[3, :]
    ys_23 = ys[2, :] - ys[3, :]

    # 预计算 Yi 与 ys 的差值
    Y_i0 = Yi - ys[0, :]
    Y_i1 = Yi - ys[1, :]
    Y_i2 = Yi - ys[2, :]
    Y_i3 = Yi - ys[3, :]

    # 计算 Ly
    Ly[0, :] = Y_i1 * Y_i2 * Y_i3 / (ys_01 * ys_02 * ys_03)
    Ly[1, :] = -Y_i0 * Y_i2 * Y_i3 / (ys_01 * ys_12 * ys_13)  # 注意此处为负数
    Ly[2, :] = Y_i0 * Y_i1 * Y_i3 / (ys_02 * ys_12 * ys_23)
    Ly[3, :] = -Y_i0 * Y_i1 * Y_i2 / (ys_03 * ys_13 * ys_23)  # 注意此处为负数

    # 计算 Lxy
    for i in range(4):
        for j in range(4):
            Lxy[i, j, :] = Lx[i, :] * Ly[j, :]

    return Lxy * zs


def Lagrange_interp_2D_vect_old(X, Y, Z, Xi, Yi):
    idx_array_x, OutRangeFlag = find_four_points_non_uniform_vector(X, Xi)
    idx_array_y, _ = find_four_points_non_uniform_vector(Y, Yi)

    xs = X[idx_array_x]
    ys = Y[idx_array_y]
    zs = Z[idx_array_x[:, None, :], idx_array_y[None, :, :]]

    interp_coefs = Interp_DoSum(xs, ys, zs, Xi, Yi)
    interp_value = np.sum(interp_coefs, axis=(0, 1))

    return interp_value, OutRangeFlag


@njit
def Lagrange_interp_2D_vect(X, Y, Z, Xi, Yi):
    idx_array_x, OutRangeFlag = find_four_points_non_uniform_vector(X, Xi)
    idx_array_y, _ = find_four_points_non_uniform_vector(Y, Yi)
    # idx_array_x, OutRangeFlag = find_four_points_non_uniform_vector_new(X, Xi)
    # idx_array_y, _ = find_four_points_non_uniform_vector_new(Y, Yi)

    xs = np.empty((idx_array_x.shape[0], idx_array_x.shape[1]), dtype=X.dtype)
    ys = np.empty((idx_array_y.shape[0], idx_array_y.shape[1]), dtype=Y.dtype)
    zs = np.empty((idx_array_x.shape[0], idx_array_y.shape[0], idx_array_y.shape[1]), dtype=Z.dtype)

    for i in range(idx_array_x.shape[0]):
        for j in range(idx_array_y.shape[0]):
            for k in range(idx_array_y.shape[1]):
                xs[i, k] = X[idx_array_x[i, k]]
                ys[j, k] = Y[idx_array_y[j, k]]
                zs[i, j, k] = Z[idx_array_x[i, k], idx_array_y[j, k]]  # 修改这里的索引逻辑

    interp_coefs = Interp_DoSum(xs, ys, zs, Xi, Yi)
    sum_over_axis_0 = np.sum(interp_coefs, axis=0)
    interp_value = np.sum(sum_over_axis_0, axis=0)

    return interp_value, OutRangeFlag


@njit
def Bilinear_interp_2D_vect(X, Y, Z, Xi, Yi):
    """
    优化前的双线性插值函数：用于非均匀二维网格，基于向量化运算。

    参数说明：
    ----------
    X : 1D ndarray, shape = (NX,)
        非均匀网格的 x 坐标，必须升序排列

    Y : 1D ndarray, shape = (NY,)
        非均匀网格的 y 坐标，必须升序排列

    Z : 2D ndarray, shape = (NX, NY)
        网格函数值，Z[i, j] = f(X[i], Y[j])

    Xi : 1D ndarray, shape = (N,)
        插值点的 x 坐标

    Yi : 1D ndarray, shape = (N,)
        插值点的 y 坐标（与 Xi 一一对应）

    返回：
    ------
    values : 1D ndarray, shape = (N,)
        每个插值点的插值结果 f(Xi[k], Yi[k])

    flags : 1D ndarray, shape = (N,)
        越界标志数组；值为 1 表示插值点超出边界范围（会被裁剪）

    ========================================
    适用于带坐标轴的二维矩阵结构，如：
    ========================================
        A[0, 1:] 表示 Y 轴坐标（如极坐标的 fi）
        A[1:, 0] 表示 X 轴坐标（如极坐标的 r）
        A[1:, 1:] 为数据区，对应 f(r, fi)

    示例：提取网格并插值
    --------------------
    A = ...  # shape = (NX+1, NY+1)
    X = A[1:, 0]        # r 坐标
    Y = A[0, 1:]        # fi 坐标
    Z = A[1:, 1:]       # 数据区

    # 插值目标点
    Xi = np.array([1.5, 2.3])
    Yi = np.array([0.8, 2.1])

    # 插值
    values, flags = Bilinear_interp_2D_vect(X, Y, Z, Xi, Yi)
    """

    idx_array_x, flag_x = find_two_points_non_uniform_vect_njit(X, Xi)
    idx_array_y, flag_y = find_two_points_non_uniform_vect_njit(Y, Yi)

    xs = np.empty((idx_array_x.shape[0], idx_array_x.shape[1]), dtype=X.dtype)
    ys = np.empty((idx_array_y.shape[0], idx_array_y.shape[1]), dtype=Y.dtype)
    zs = np.empty(
        (idx_array_x.shape[0], idx_array_y.shape[0], idx_array_y.shape[1]),
        dtype=Z.dtype,
    )

    for i in range(idx_array_x.shape[0]):
        for j in range(idx_array_y.shape[0]):
            for k in range(idx_array_y.shape[1]):
                xs[i, k] = X[idx_array_x[i, k]]
                ys[j, k] = Y[idx_array_y[j, k]]
                zs[i, j, k] = Z[idx_array_x[i, k], idx_array_y[j, k]]

    interp_coefs = Interp_DoSum_Bilinear(xs, ys, zs, Xi, Yi)
    sum_over_axis0 = np.sum(interp_coefs, axis=0)
    interp_value   = np.sum(sum_over_axis0, axis=0)
    return interp_value, flag_x


# @njit(parallel=True)
@njit()
def Bilinear_interp_2D_vect_uniform(X0, dX, NX,
                                    Y0, dY, NY,
                                    Z,
                                    Xi, Yi,
                                    values, out_flags):
    """
    双线性插值函数：用于等间距二维网格，支持并行加速。

    参数说明：
    ----------
    X0 : float         # x 网格起点，例如 r[0]
    dX : float         # x 网格步长
    NX : int           # x 网格点数
    Y0 : float         # y 网格起点，例如 fi[0]
    dY : float         # y 网格步长
    NY : int           # y 网格点数

    Z : 2D ndarray, shape = (NX, NY)
        网格上的函数值，Z[i, j] = f(X[i], Y[j])

    Xi : 1D ndarray, shape = (N,)
        插值点的 x 坐标

    Yi : 1D ndarray, shape = (N,)
        插值点的 y 坐标

    values : 1D ndarray, shape = (N,)
        输出插值值（预先分配）

    out_flags : 1D ndarray, shape = (N,)
        越界标志数组；1 表示该点超出网格范围

    返回：
    -------
    values : 插值值（写入 values 数组）
    out_flags : 越界标志（写入 out_flags 数组）

    ========================================
    适用于带坐标轴的二维矩阵结构，如：
    ========================================
        A[0, 1:] 表示 Y 轴坐标（如极坐标的 fi）
        A[1:, 0] 表示 X 轴坐标（如极坐标的 r）
        A[1:, 1:] 为数据区，对应 f(r, fi)

    示例：提取网格并插值
    --------------------
    A = ...  # shape = (NX+1, NY+1)
    X = A[1:, 0]        # r 坐标
    Y = A[0, 1:]        # fi 坐标
    Z = A[1:, 1:]       # 数据区

    # 等间距网格判断 & 提取参数
    X0, dX = X[0], X[1] - X[0]
    Y0, dY = Y[0], Y[1] - Y[0]
    NX, NY = len(X), len(Y)

    # 插值目标点
    Xi = np.array([1.5, 2.3])
    Yi = np.array([0.8, 2.1])
    values = np.empty_like(Xi)
    out_flags = np.zeros_like(Xi, dtype=np.int32)

    values, out_flags = Bilinear_interp_2D_vect_uniform(
        X0, dX, NX, Y0, dY, NY, Z, Xi, Yi, values, out_flags
    )
    """
    N = Xi.shape[0]

    for k in range(N):
        x = Xi[k]
        y = Yi[k]

        i = int(np.floor((x - X0) / dX))
        j = int(np.floor((y - Y0) / dY))

        if i < 0:
            i = 0
            out_flags[k] = 1
        elif i >= NX - 1:
            i = NX - 2
            out_flags[k] = 1

        if j < 0:
            j = 0
            out_flags[k] = 1
        elif j >= NY - 1:
            j = NY - 2
            out_flags[k] = 1

        xx0 = X0 + i * dX
        yy0 = Y0 + j * dY

        tx = (x - xx0) / dX
        ty = (y - yy0) / dY

        z00 = Z[i,     j    ]
        z01 = Z[i,     j + 1]
        z10 = Z[i + 1, j    ]
        z11 = Z[i + 1, j + 1]

        val = (1 - tx) * (1 - ty) * z00 + \
              (1 - tx) * ty       * z01 + \
              tx       * (1 - ty) * z10 + \
              tx       * ty       * z11

        values[k] = val

    return values[:N], out_flags[:N]


@njit()
def Bilinear_interp_2D_vect_uniform_3fields(
    X0, dX, NX,
    Y0, dY, NY,
    Er_map, Ezcoef_map, Ephi_map,
    Xi, Yi, Zi,
    rf_amp,
    Er_values, Ez_values, Ephi_values,
    out_flags
):
    """
    对等间距二维网格做三张图的联合双线性插值，并直接完成：
        Er   -> rf_amp * Er
        Ez   -> rf_amp * (Ezcoef * z)
        Ephi -> rf_amp * Ephi

    参数
    ----
    X0, dX, NX : x=r 网格定义
    Y0, dY, NY : y=phi 网格定义

    Er_map, Ezcoef_map, Ephi_map : 2D ndarray, shape=(NX, NY)

    Xi, Yi : 插值点 (r, phi)
    Zi     : 插值点 z，用于 Ez = z * Ezcoef
    rf_amp : 总体幅值系数

    Er_values, Ez_values, Ephi_values : 输出数组，预先分配
    out_flags : 越界标记，预先分配；1 表示该点越界

    返回
    ----
    Er_values, Ez_values, Ephi_values, out_flags
    """
    N = Xi.shape[0]

    for k in range(N):
        out_flags[k] = 0

        x = Xi[k]
        y = Yi[k]
        z = Zi[k]

        i = int(np.floor((x - X0) / dX))
        j = int(np.floor((y - Y0) / dY))

        if i < 0:
            i = 0
            out_flags[k] = 1
        elif i >= NX - 1:
            i = NX - 2
            out_flags[k] = 1

        if j < 0:
            j = 0
            out_flags[k] = 1
        elif j >= NY - 1:
            j = NY - 2
            out_flags[k] = 1

        xx0 = X0 + i * dX
        yy0 = Y0 + j * dY

        tx = (x - xx0) / dX
        ty = (y - yy0) / dY

        w00 = (1.0 - tx) * (1.0 - ty)
        w01 = (1.0 - tx) * ty
        w10 = tx * (1.0 - ty)
        w11 = tx * ty

        # ---- Er ----
        er00 = Er_map[i,     j    ]
        er01 = Er_map[i,     j + 1]
        er10 = Er_map[i + 1, j    ]
        er11 = Er_map[i + 1, j + 1]

        er_val = w00 * er00 + w01 * er01 + w10 * er10 + w11 * er11

        # ---- Ezcoef ----
        ez00 = Ezcoef_map[i,     j    ]
        ez01 = Ezcoef_map[i,     j + 1]
        ez10 = Ezcoef_map[i + 1, j    ]
        ez11 = Ezcoef_map[i + 1, j + 1]

        ezcoef_val = w00 * ez00 + w01 * ez01 + w10 * ez10 + w11 * ez11

        # ---- Ephi ----
        ef00 = Ephi_map[i,     j    ]
        ef01 = Ephi_map[i,     j + 1]
        ef10 = Ephi_map[i + 1, j    ]
        ef11 = Ephi_map[i + 1, j + 1]

        ephi_val = w00 * ef00 + w01 * ef01 + w10 * ef10 + w11 * ef11

        # ---- 直接完成物理缩放 ----
        Er_values[k]   = rf_amp * er_val
        Ez_values[k]   = rf_amp * ezcoef_val * z
        Ephi_values[k] = rf_amp * ephi_val

    return Er_values[:N], Ez_values[:N], Ephi_values[:N], out_flags[:N]


@njit
def interpolate_spiral_emap_unique_gap_kernel(
    r, fi, z,
    gap_azimuths,
    rf_amp,
    r0, dr, nr,
    phi0, dphi, nphi,
    Er_map, Ezcoef_map, Ephi_map,
    Er_out, Ez_out, Ephi_out,
    HitCount,
    tmp_phi, tmp_flag
):
    """
    每个点只命中一个 gap 的纯空间版本：

    1. 不考虑 RF 时间相位
    2. 整圈只保留真实存在的物理 gap
    3. 假设不同 gap 的窗口不重叠
    4. 每个点最多命中一个 gap；一旦命中即停止查找
    5. 直接输出真实 Ez，而不是 Ez/z

    HitCount:
        -1 : 未命中任何 gap
        >=0: 命中的 gap 编号（0,1,2,...）
    """
    npt = r.shape[0]
    ngap = gap_azimuths.shape[0]

    phi_min = phi0
    phi_max = phi0 + dphi * (nphi - 1)

    # --------------------------------------------------
    # 第一步：先为每个点找到唯一命中的 gap，并得到局部 phi
    # --------------------------------------------------
    for i in range(npt):
        Er_out[i] = 0.0
        Ez_out[i] = 0.0
        Ephi_out[i] = 0.0

        HitCount[i] = -1     # 默认未命中

        for igap in range(ngap):
            phi_local, inside = shift_to_nearest_window(
                fi[i] - gap_azimuths[igap], phi_min, phi_max
            )

            if inside:
                tmp_phi[i] = phi_local
                HitCount[i] = igap
                break

    # --------------------------------------------------
    # 第二步：对所有已命中的点只做一次联合插值
    # --------------------------------------------------
    Bilinear_interp_2D_vect_uniform_3fields(
        r0, dr, nr,
        phi0, dphi, nphi,
        Er_map, Ezcoef_map, Ephi_map,
        r, tmp_phi, z,
        rf_amp,
        Er_out, Ez_out, Ephi_out,
        tmp_flag
    )

    # --------------------------------------------------
    # 第三步：若插值失败（例如 r 越界），则回退为未命中状态
    # --------------------------------------------------
    for i in range(npt):
        if HitCount[i] < 0:
            Er_out[i] = 0.0
            Ez_out[i] = 0.0
            Ephi_out[i] = 0.0
            # HitCount[i] = -1


@njit
def shift_to_nearest_window(phi, phi_min, phi_max):
    """
    把 phi 平移到最接近 [phi_min, phi_max] 的那个 2pi 周期副本附近，
    但不强行塞进窗口。

    返回
    ----
    phi_local : float
        平移后的角度
    inside : bool
        是否真的落在 [phi_min, phi_max] 内
    """
    twopi = 2.0 * np.pi
    phi_center = 0.5 * (phi_min + phi_max)
    k = np.round((phi_center - phi) / twopi)
    phi_local = phi + k * twopi
    inside = (phi_local >= phi_min) and (phi_local <= phi_max)
    return phi_local, inside



@njit(parallel=True, fastmath=True)
def Bmap_add_all_orders(
        X0, dX, NX,
        Y0, dY, NY,
        Bz_stack, Bfi_stack, Br_stack,      # shape (max_order+1, NX, NY)
        Xi, Yi, Zi,                        # 粒子坐标 (N,)
        max_order,
        out_Bz, out_Bfi, out_Br            # shape (N,) —— 累加到这里
):
    N = Xi.shape[0]

    for k in prange(N):
        # -------- 网格索引 --------
        x = Xi[k];  y = Yi[k]
        i = int((x - X0) / dX)
        j = int((y - Y0) / dY)
        if i < 0: i = 0
        elif i >= NX-1: i = NX-2
        if j < 0: j = 0
        elif j >= NY-1: j = NY-2

        # -------- 权重 --------
        tx = (x - (X0 + i*dX)) / dX
        ty = (y - (Y0 + j*dY)) / dY
        w00 = (1.0-tx)*(1.0-ty); w01 = (1.0-tx)*ty
        w10 = tx*(1.0-ty);       w11 = tx*ty
        i1, j1 = i+1, j+1

        # -------- z 幂递推 --------
        z     = Zi[k]
        z2    = z * z
        z_even = z2          # 对应 order=1 的 z^(2)
        z_odd  = z_even * z  #            z^(3)

        # -------- 遍历所有阶次 --------
        for n in range(1, max_order+1):
            bz  = (w00*Bz_stack [n, i , j ] + w01*Bz_stack [n, i , j1] +
                   w10*Bz_stack [n, i1, j ] + w11*Bz_stack [n, i1, j1])
            bfi = (w00*Bfi_stack[n, i , j ] + w01*Bfi_stack[n, i , j1] +
                   w10*Bfi_stack[n, i1, j ] + w11*Bfi_stack[n, i1, j1])
            br  = (w00*Br_stack [n, i , j ] + w01*Br_stack [n, i , j1] +
                   w10*Br_stack[n, i1, j ] + w11*Br_stack[n, i1, j1])

            out_Bz [k] += bz  * z_even
            out_Bfi[k] += bfi * z_odd
            out_Br [k] += br  * z_odd

            if n != max_order:          # 递推到下一阶
                z_even *= z2            # z^(2(n+1))
                z_odd  *= z2            # z^(2(n+1)+1)



# @njit
def rk4_step(func, t, r, h, GlobalParameters, LocalParameters):
    # shape of r = number of particles * number of coordinates
    k1 = h * func(t, r, GlobalParameters, LocalParameters)
    k2 = h * func(t + 0.5 * h, r + 0.5 * k1, GlobalParameters, LocalParameters)
    k3 = h * func(t + 0.5 * h, r + 0.5 * k2, GlobalParameters, LocalParameters)
    k4 = h * func(t + h, r + k3, GlobalParameters, LocalParameters)
    # shape of k1,...,k4 = n*7
    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


# segment cross checking
@njit
def direction_np_njit(a, b, c):
    ab = b - a
    ac = c - a
    return cross2d(ab, ac)


@njit
def on_segment_np_njit(a, b, c):
    return np.all(np.logical_and(np.minimum(a, b) <= c, c <= np.maximum(a, b)))


@njit
def segments_intersect_np_njit(arr1, arr2):
    a = arr1[:2]
    b = arr1[2:]
    c = arr2[:2]
    d = arr2[2:]

    LengthCD = np.sqrt((d[0] - c[0]) ** 2 + (d[1] - c[1]) ** 2)
    LengthAB = np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    d1 = direction_np_njit(c, d, a)
    d2 = direction_np_njit(c, d, b)
    d3 = direction_np_njit(a, b, c)
    d4 = direction_np_njit(a, b, d)

    L_A2CD = d1 / LengthCD
    L_B2CD = d2 / LengthCD
    L_C2AB = d3 / LengthAB
    L_D2AB = d4 / LengthAB

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True, L_A2CD, L_B2CD, L_C2AB, L_D2AB

    if d1 == 0 and on_segment_np_njit(c, d, a):
        return True, L_A2CD, L_B2CD, L_C2AB, L_D2AB
    if d2 == 0 and on_segment_np_njit(c, d, b):
        return True, L_A2CD, L_B2CD, L_C2AB, L_D2AB
    if d3 == 0 and on_segment_np_njit(a, b, c):
        return True, L_A2CD, L_B2CD, L_C2AB, L_D2AB
    if d4 == 0 and on_segment_np_njit(a, b, d):
        return True, L_A2CD, L_B2CD, L_C2AB, L_D2AB

    return False, L_A2CD, L_B2CD, L_C2AB, L_D2AB


@njit
def check_intersections_njit(segment_group_a, segment_group_b):
    na = len(segment_group_a)
    nb = len(segment_group_b)
    intersection_matrix = np.zeros((na, nb), dtype=types.boolean)
    Dist_matrix_A2CD = np.zeros((na, nb), dtype=types.float64)
    Dist_matrix_B2CD = np.zeros((na, nb), dtype=types.float64)
    Dist_matrix_C2AB = np.zeros((na, nb), dtype=types.float64)
    Dist_matrix_D2AB = np.zeros((na, nb), dtype=types.float64)
    for i, seg_a in enumerate(segment_group_a):
        for j, seg_b in enumerate(segment_group_b):
            (intersection_matrix[i, j], Dist_matrix_A2CD[i, j],
             Dist_matrix_B2CD[i, j], Dist_matrix_C2AB[i, j],
             Dist_matrix_D2AB[i, j]) = segments_intersect_np_njit(seg_a, seg_b)
    return intersection_matrix, Dist_matrix_A2CD, Dist_matrix_B2CD, Dist_matrix_C2AB, Dist_matrix_D2AB

@njit
def CheckIntersect_njit_ParticleCoord(r_PreStep, r_PostStep, segment_group_b):
    r_pre, fi_pre = r_PreStep[:, 0], r_PreStep[:, 4]
    r_post, fi_post = r_PostStep[:, 0], r_PostStep[:, 4]

    x_pre, y_pre = r_pre * np.cos(fi_pre), r_pre * np.sin(fi_pre)
    x_post, y_post = r_post * np.cos(fi_post), r_post * np.sin(fi_post)

    segment_group_a = np.column_stack((x_pre, y_pre, x_post, y_post))

    na = len(segment_group_a)
    nb = len(segment_group_b)
    nn = r_PreStep.shape[1]
    intersection_matrix = np.zeros((na, nb), dtype=types.boolean)
    Dist_matrix_A2CD = np.zeros((na, nb), dtype=types.float64)
    Dist_matrix_B2CD = np.zeros((na, nb), dtype=types.float64)
    Dist_matrix_C2AB = np.zeros((na, nb), dtype=types.float64)
    Dist_matrix_D2AB = np.zeros((na, nb), dtype=types.float64)

    r_CentralStep = np.zeros((na, nb, nn))

    for i, seg_a in enumerate(segment_group_a):
        for j, seg_b in enumerate(segment_group_b):
            (intersection_matrix[i, j], Dist_matrix_A2CD[i, j],
             Dist_matrix_B2CD[i, j], Dist_matrix_C2AB[i, j],
             Dist_matrix_D2AB[i, j]) = segments_intersect_np_njit(seg_a, seg_b)
            if intersection_matrix[i, j]:
                AoOverBo = (np.abs(Dist_matrix_A2CD[i, j]) / (
                        np.abs(Dist_matrix_A2CD[i, j]) + np.abs(Dist_matrix_B2CD[i, j])))
                r_CentralStep[i, j, :] = r_PreStep[i, :] + AoOverBo * (r_PostStep[i, :] - r_PreStep[i, :])

    return intersection_matrix, r_CentralStep, Dist_matrix_A2CD, Dist_matrix_B2CD, Dist_matrix_C2AB, Dist_matrix_D2AB

@njit
def stack_to_matrix(stack_matrix, stack_ids, input_matrix, input_ids):
    # stack_matrix: 已有的堆叠矩阵 (n, 10, t)
    # input_matrix: 要堆叠的粒子坐标 (n, 10)
    # input_ids: 要堆叠的粒子id号 (m, )
    # stack_ids: 每个粒子的堆叠次数 (n, )

    n = stack_matrix.shape[0]  # 粒子总数
    for particle_id in input_ids:
        # 确定当前粒子的堆叠层数
        layer_id = stack_ids[particle_id]  # 使用 stack_ids 跟踪堆叠次数

        # 如果层数超出当前矩阵的层数，扩展矩阵
        if layer_id >= stack_matrix.shape[2]:
            # 扩展一个层次
            stack_matrix = np.append(stack_matrix, -np.ones((n, 10, stack_matrix.shape[2])), axis=2)

        # 堆叠粒子
        stack_matrix[particle_id, :, layer_id] = input_matrix[particle_id, :]

        # 更新该粒子的堆叠次数
        stack_ids[particle_id] += 1

    return stack_matrix, stack_ids

@njit
def v2Ek_fast(v):
    c = 2.99792458e8
    q = 1.60217662e-19
    E0_J = 938.2723e6 * q
    beta = v / c
    gamma = 1 / np.sqrt(1 - beta ** 2)
    Ek_J = (gamma - 1) * E0_J
    Etotal_J = E0_J + Ek_J
    Ek_MeV = Ek_J / q / 1e6
    return Ek_MeV, Etotal_J

@njit
def v2Ek_J_fast(v):
    c = 2.99792458e8
    q = 1.60217662e-19
    E0_J = 938.2723e6 * q
    Etotal_J = E0_J * c / np.sqrt(c**2 - v ** 2)
    return Etotal_J


@njit
def stack_to_matrix(stack_matrix, stack_ids, input_matrix, input_ids):
    # stack_matrix: 已有的堆叠矩阵 (n, 10, t)
    # input_matrix: 要堆叠的粒子坐标 (n, 10)
    # input_ids: 要堆叠的粒子id号 (m, )
    # stack_ids: 每个粒子的堆叠次数 (n, )

    n = stack_matrix.shape[0]  # 粒子总数
    m = stack_matrix.shape[1]  # dims
    for particle_id in input_ids:
        # 确定当前粒子的堆叠层数
        layer_id = stack_ids[particle_id]  # 使用 stack_ids 跟踪堆叠次数

        # 如果层数超出当前矩阵的层数，扩展矩阵
        if layer_id >= stack_matrix.shape[2]:
            # 扩展一个层次
            stack_matrix = np.append(stack_matrix, -np.ones((n, m, stack_matrix.shape[2])), axis=2)

        # 堆叠粒子
        stack_matrix[particle_id, :, layer_id] = input_matrix[particle_id, :]

        # 更新该粒子的堆叠次数
        stack_ids[particle_id] += 1

    return stack_matrix, stack_ids


# @njit
def linear_interpolation_njit(X, Y, xi):

    idx_array, _ = find_two_points_non_uniform_vect_njit(X, xi)
    x = X[idx_array]
    y = Y[idx_array]
    x0, x1, y0, y1 = x[0], x[1], y[0], y[1]
    return y0 + (xi - x0) / (x1 - x0) * (y1 - y0)


@njit
def linear_interpolation_uniform_njit(X0, dX, Y, xi):
    """
    一维线性插值函数：等间距网格版本

    参数
    ----------
    X0 : float
        网格起点 X[0]
    dX : float
        网格步长
    Y : 1D ndarray, shape=(N,)
        网格函数值，Y[i] = f(X0 + i*dX)
    xi : float
        目标点坐标

    返回
    ----------
    yi : float
        在 xi 处的插值结果
    out_flag : int
        越界标志 (0=正常, 1=超出网格)
    """

    N = Y.shape[0]

    # 计算区间索引
    i = int(np.floor((xi - X0) / dX))
    out_flag = 0

    # 边界检查
    if i < 0:
        i = 0
        out_flag = 1
    elif i >= N - 1:
        i = N - 2
        out_flag = 1

    # 区间端点
    x0 = X0 + i * dX
    x1 = x0 + dX
    y0, y1 = Y[i], Y[i + 1]

    # 线性插值
    yi = y0 + (xi - x0) / (x1 - x0) * (y1 - y0)
    return yi, out_flag


@njit
def linear_interpolation_uniform_njit_two_vars(X0, dX, Y, Z, xi):
    """
    一维线性插值函数：等间距网格版本

    参数
    ----------
    X0 : float
        网格起点 X[0]
    dX : float
        网格步长
    Y, Z : 1D ndarray, shape=(N,)
        网格函数值，Y[i] = f(X0 + i*dX)
    xi : float
        目标点坐标

    返回
    ----------
    yi : float
        在 xi 处的插值结果
    out_flag : int
        越界标志 (0=正常, 1=超出网格)
    """

    N = Y.shape[0]

    # 计算区间索引
    i = int(np.floor((xi - X0) / dX))
    out_flag = 0

    # 边界检查
    if i < 0:
        i = 0
        out_flag = 1
    elif i >= N - 1:
        i = N - 2
        out_flag = 1

    # 区间端点
    x0 = X0 + i * dX
    x1 = x0 + dX
    y0, y1 = Y[i], Y[i + 1]
    z0, z1 = Z[i], Z[i + 1]

    # 线性插值
    yi = y0 + (xi - x0) / (x1 - x0) * (y1 - y0)
    zi = z0 + (xi - x0) / (x1 - x0) * (z1 - z0)
    return yi, zi, out_flag


@njit
def find_two_points_non_uniform_vect_njit(X, xi):

    right_point = np.searchsorted(X, xi, side='right')
    left_point = right_point - 1

    # Check if xi is out of the range of X
    idx_array_0, idx_array_1, flag = (
        left_point, right_point, np.zeros_like(xi))

    # if left_point < 0
    flag_left_point_less_0 = left_point < 0
    idx_array_0[flag_left_point_less_0] = 0
    idx_array_1[flag_left_point_less_0] = 1
    flag[flag_left_point_less_0] = 1

    # if right_right_point > len(X) - 1
    flag_right_point_larger_len = right_point > len(X) - 1
    idx_array_0[flag_right_point_larger_len] = len(X) - 2
    idx_array_1[flag_right_point_larger_len] = len(X) - 1
    flag[flag_right_point_larger_len] = 1

    idx_array = np.vstack((idx_array_0, idx_array_1))

    return idx_array, flag


@njit()
def my_3dInterp_vect(x, y, z, values, xi, yi, zi):

    x_idx, _ = find_two_points_non_uniform_vect_njit(x, xi)
    y_idx, _ = find_two_points_non_uniform_vect_njit(y, yi)
    z_idx, _ = find_two_points_non_uniform_vect_njit(z, zi)

    interp_values = xi * 0.0

    for idxs in range(len(xi)):

        xi_current, yi_current, zi_current = xi[idxs], yi[idxs], zi[idxs]

        x_left, x_right = x[x_idx[0, idxs]], x[x_idx[1, idxs]]
        y_left, y_right = y[y_idx[0, idxs]], y[y_idx[1, idxs]]
        z_left, z_right = z[z_idx[0, idxs]], z[z_idx[1, idxs]]

        x0_00 = values[x_idx[0, idxs], y_idx[0, idxs], z_idx[0, idxs]]
        x0_01 = values[x_idx[0, idxs], y_idx[0, idxs], z_idx[1, idxs]]
        x0_10 = values[x_idx[0, idxs], y_idx[1, idxs], z_idx[0, idxs]]
        x0_11 = values[x_idx[0, idxs], y_idx[1, idxs], z_idx[1, idxs]]

        x1_00 = values[x_idx[1, idxs], y_idx[0, idxs], z_idx[0, idxs]]
        x1_01 = values[x_idx[1, idxs], y_idx[0, idxs], z_idx[1, idxs]]
        x1_10 = values[x_idx[1, idxs], y_idx[1, idxs], z_idx[0, idxs]]
        x1_11 = values[x_idx[1, idxs], y_idx[1, idxs], z_idx[1, idxs]]

        x0_z0_interp = (yi_current - y_left) / (y_right - y_left) * x0_10 + (y_right - yi_current) / (y_right - y_left) * x0_00
        x0_z1_interp = (yi_current - y_left) / (y_right - y_left) * x0_11 + (y_right - yi_current) / (y_right - y_left) * x0_01
        x0_interp = (zi_current - z_left) / (z_right - z_left) * x0_z1_interp + (z_right - zi_current) / (z_right - z_left) * x0_z0_interp

        x1_z0_interp = (yi_current - y_left) / (y_right - y_left) * x1_10 + (y_right - yi_current) / (y_right - y_left) * x1_00
        x1_z1_interp = (yi_current - y_left) / (y_right - y_left) * x1_11 + (y_right - yi_current) / (y_right - y_left) * x1_01
        x1_interp = (zi_current - z_left) / (z_right - z_left) * x1_z1_interp + (z_right - zi_current) / (z_right - z_left) * x1_z0_interp

        interp_values[idxs] = (xi_current - x_left) / (x_right - x_left) * x1_interp + (x_right - xi_current) / (x_right - x_left) * x0_interp

    return interp_values


@njit()
def fast_mod_njit(a, b):
    a -= b * np.floor(a / b)
    return a

@njit(parallel=True)
def fast_mod_parallel(a, b, result):
    for i in prange(a.size):
        result[i] = a[i] - b * np.floor(a[i] / b)
    return result

# @njit(parallel=True)
# def fast_mod_parallel(a, b):
#     result = np.zeros_like(a)
#     for i in prange(a.size):
#         result[i] = a[i] - b * np.floor(a[i] / b)
#     return result

# @profile
@njit(parallel=True)
def detect_azimuth_crossing(pre_fi_raw, post_fi_raw, pre_fi_mod, post_fi_mod, crossing_mask, ratio_merge, survive_flag, target_angle):
    """
    判断粒子是否在本步中穿越指定的目标方位角。

    参数:
        pre_fi_raw: np.ndarray，前一步的方位角
        post_fi_raw: np.ndarray，当前步的方位角
        survive_flag: np.ndarray，粒子是否存活 (1 表示存活)
        target_angle: float，目标方位角

    返回:
        crossing_indices: np.ndarray[int]，穿越目标方位角的粒子索引
        ratio_merge: np.ndarray[float]，对应每个穿越粒子的插值比例，未穿越则为 NaN
    """
    # -------------------- 1) 提取数据 --------------------

    target_angle = target_angle % (2 * np.pi)

    # -------------------- 2) 归一化方位角到 [0, 2π) --------------------
    # pre_fi = pre_fi_raw % (2 * np.pi)
    # post_fi = post_fi_raw % (2 * np.pi)
    # pre_fi = fast_mod_njit(pre_fi_raw, 2 * np.pi)
    # post_fi = fast_mod_njit(post_fi_raw, 2 * np.pi)
    fast_mod_parallel(pre_fi_raw, (2 * np.pi), pre_fi_mod)
    fast_mod_parallel(post_fi_raw, (2 * np.pi), post_fi_mod)

    N = pre_fi_raw.size
    # crossing_mask = np.zeros(N, dtype=np.bool_)
    # ratio_merge = np.zeros(N, dtype=np.float64)

    for i in prange(N):
        crossing_mask[i] = False
        ratio_merge[i] = 0.0

        if survive_flag[i] != 1:
            continue

        pre = pre_fi_mod[i]
        post = post_fi_mod[i]

        if post >= pre:
            # 常规情况，不跨周期
            if pre <= target_angle <= post:
                crossing_mask[i] = True
                span = post - pre
                if span > 1e-12:
                    ratio_merge[i] = (target_angle - pre) / span
        else:
            # 跨周期情况, 若目标角 ∈ [pre_fi, 2π) 或 [0, post_fi], 则表示穿越
            if (target_angle >= pre and target_angle < 2 * np.pi) or \
                    (target_angle >= 0 and target_angle <= post):
                crossing_mask[i] = True
                post_unwrapped = post + 2 * np.pi
                span = post_unwrapped - pre
                if span > 1e-12:
                    target_unwrapped = target_angle
                    if target_angle <= post:
                        target_unwrapped += 2 * np.pi
                    ratio_merge[i] = (target_unwrapped - pre) / span

    # 返回穿越的粒子索引
    return crossing_mask


# @profile
@njit(parallel=True)
def detect_azimuth_crossing(pre_fi_raw, post_fi_raw, pre_fi_mod, post_fi_mod, crossing_mask, ratio_merge, survive_flag, target_angle):
    """
    判断粒子是否在本步中穿越指定的目标方位角。

    参数:
        pre_fi_raw: np.ndarray，前一步的方位角
        post_fi_raw: np.ndarray，当前步的方位角
        survive_flag: np.ndarray，粒子是否存活 (1 表示存活)
        target_angle: float，目标方位角

    返回:
        crossing_indices: np.ndarray[int]，穿越目标方位角的粒子索引
        ratio_merge: np.ndarray[float]，对应每个穿越粒子的插值比例，未穿越则为 NaN
    """
    # -------------------- 1) 提取数据 --------------------

    target_angle = target_angle % (2 * np.pi)

    # -------------------- 2) 归一化方位角到 [0, 2π) --------------------
    # pre_fi = pre_fi_raw % (2 * np.pi)
    # post_fi = post_fi_raw % (2 * np.pi)
    # pre_fi = fast_mod_njit(pre_fi_raw, 2 * np.pi)
    # post_fi = fast_mod_njit(post_fi_raw, 2 * np.pi)
    fast_mod_parallel(pre_fi_raw, (2 * np.pi), pre_fi_mod)
    fast_mod_parallel(post_fi_raw, (2 * np.pi), post_fi_mod)

    N = pre_fi_raw.size
    # crossing_mask = np.zeros(N, dtype=np.bool_)
    # ratio_merge = np.zeros(N, dtype=np.float64)

    for i in prange(N):
        crossing_mask[i] = False
        ratio_merge[i] = 0.0

        if survive_flag[i] != 1:
            continue

        pre = pre_fi_mod[i]
        post = post_fi_mod[i]

        if post >= pre:
            # 常规情况，不跨周期
            if pre <= target_angle <= post:
                crossing_mask[i] = True
                span = post - pre
                if span > 1e-12:
                    ratio_merge[i] = (target_angle - pre) / span
        else:
            # 跨周期情况, 若目标角 ∈ [pre_fi, 2π) 或 [0, post_fi], 则表示穿越
            if (target_angle >= pre and target_angle < 2 * np.pi) or \
                    (target_angle >= 0 and target_angle <= post):
                crossing_mask[i] = True
                post_unwrapped = post + 2 * np.pi
                span = post_unwrapped - pre
                if span > 1e-12:
                    target_unwrapped = target_angle
                    if target_angle <= post:
                        target_unwrapped += 2 * np.pi
                    ratio_merge[i] = (target_unwrapped - pre) / span

    # 返回穿越的粒子索引
    return crossing_mask

@njit(parallel=True)
def detect_azimuth_crossing_portion(pre_fi_raw, post_fi_raw, pre_fi_mod, post_fi_mod, crossing_mask, ratio_merge, survive_flag, target_angle, LIDs_record):
    """
    判断粒子是否在本步中穿越指定的目标方位角。

    参数:
        pre_fi_raw: np.ndarray，前一步的方位角
        post_fi_raw: np.ndarray，当前步的方位角
        survive_flag: np.ndarray，粒子是否存活 (1 表示存活)
        target_angle: float，目标方位角

    返回:
        crossing_indices: np.ndarray[int]，穿越目标方位角的粒子索引
        ratio_merge: np.ndarray[float]，对应每个穿越粒子的插值比例，未穿越则为 NaN
    """
    # -------------------- 1) 提取数据 --------------------

    target_angle = target_angle % (2 * np.pi)

    # -------------------- 2) 归一化方位角到 [0, 2π) --------------------
    fast_mod_parallel(pre_fi_raw, (2 * np.pi), pre_fi_mod)
    fast_mod_parallel(post_fi_raw, (2 * np.pi), post_fi_mod)

    N = LIDs_record.size

    for i in prange(N):
        crossing_mask[LIDs_record[i]] = False
        ratio_merge[LIDs_record[i]] = 0.0

        if survive_flag[LIDs_record[i]] != 1:
            continue

        pre = pre_fi_mod[LIDs_record[i]]
        post = post_fi_mod[LIDs_record[i]]

        if post >= pre:
            # 常规情况，不跨周期
            if pre <= target_angle <= post:
                crossing_mask[LIDs_record[i]] = True
                span = post - pre
                if span > 1e-12:
                    ratio_merge[i] = (target_angle - pre) / span
        else:
            # 跨周期情况, 若目标角 ∈ [pre_fi, 2π) 或 [0, post_fi], 则表示穿越
            if (target_angle >= pre and target_angle < 2 * np.pi) or \
                    (target_angle >= 0 and target_angle <= post):
                crossing_mask[LIDs_record[i]] = True
                post_unwrapped = post + 2 * np.pi
                span = post_unwrapped - pre
                if span > 1e-12:
                    target_unwrapped = target_angle
                    if target_angle <= post:
                        target_unwrapped += 2 * np.pi
                    ratio_merge[LIDs_record[i]] = (target_unwrapped - pre) / span

    # 返回穿越的粒子索引
    return crossing_mask

# @njit
# @profile
# def detect_azimuth_crossing(pre_fi_raw, post_fi_raw, survive_flag, target_angle):
#     """
#     判断粒子是否在本步中穿越指定的目标方位角。
#
#     参数:
#         pre_fi_raw: np.ndarray，前一步的方位角
#         post_fi_raw: np.ndarray，当前步的方位角
#         survive_flag: np.ndarray，粒子是否存活 (1 表示存活)
#         target_angle: float，目标方位角
#
#     返回:
#         crossing_indices: np.ndarray[int]，穿越目标方位角的粒子索引
#         ratio_merge: np.ndarray[float]，对应每个穿越粒子的插值比例，未穿越则为 NaN
#     """
#     # -------------------- 1) 提取数据 --------------------
#
#     target_angle = target_angle % (2 * np.pi)
#
#     # -------------------- 2) 归一化方位角到 [0, 2π) --------------------
#     # pre_fi = pre_fi_raw % (2 * np.pi)
#     # post_fi = post_fi_raw % (2 * np.pi)
#     pre_fi = fast_mod_njit(pre_fi_raw, (2 * np.pi))
#     post_fi = fast_mod_njit(post_fi_raw, (2 * np.pi))
#
#     # 准备输出： ratio_merge 默认 NaN
#     ratio_merge = np.full(pre_fi.shape, np.nan, dtype=np.float64)
#     is_alive = (survive_flag == 1)
#     tgt_ge_pre = (target_angle >= pre_fi)
#     # tgt_lt_pre = ~tgt_ge_pre  # 替代 target_angle < pre_fi
#     tgt_lt_post = (target_angle <= post_fi)
#
#     # -------------------- 3) 情形 A: post_fi >= pre_fi --------------------
#     cond_A = (post_fi >= pre_fi)
#     # A1: 判断穿越 (pre_fi <= target_angle <= post_fi)
#     # crossing_mask_A = (
#     #         cond_A &
#     #         (pre_fi <= target_angle) &
#     #         (target_angle <= post_fi) &
#     #         (survive_flag == 1)
#     # )
#     crossing_mask_A = (
#             cond_A &
#             tgt_ge_pre &
#             tgt_lt_post &
#             is_alive
#     )
#     # A2: 计算插值比例 ratio = (target_angle - pre_fi) / (post_fi - pre_fi)
#     span_A = post_fi - pre_fi
#     valid_A = (span_A != 0)
#     ratio_A = np.full_like(span_A, np.nan, dtype=np.float64)
#     mask_A_final = crossing_mask_A & valid_A
#     ratio_A[mask_A_final] = (target_angle - pre_fi[mask_A_final]) / span_A[mask_A_final]
#
#     # -------------------- 4) 情形 B: post_fi < pre_fi (跨 2π) --------------------
#     # cond_B = (post_fi < pre_fi)
#     cond_B = ~ cond_A
#     # B1: 判断穿越
#     #     若目标角 ∈ [pre_fi, 2π) 或 [0, post_fi], 则表示穿越
#     # crossing_mask_B = (
#     #         cond_B &
#     #         (survive_flag == 1) &
#     #         (
#     #                 ((target_angle >= pre_fi) & (target_angle < 2 * np.pi))  # [pre_fi, 2π)
#     #                 |
#     #                 ((target_angle >= 0) & (target_angle <= post_fi))  # [0, post_fi]
#     #         )
#     # )
#     crossing_mask_B = (
#             cond_B &
#             is_alive &
#             (
#                     (tgt_ge_pre & (target_angle < 2 * np.pi))  # [pre_fi, 2π)
#                     |
#                     ((target_angle >= 0) & tgt_lt_post)  # [0, post_fi]
#             )
#     )
#     # B2: 计算插值比例
#     #     将 post_fi 视为 post_fi+2π, 若目标角 < post_fi 则也视为 target_angle+2π
#     post_fi_eff = np.where(cond_B, post_fi + 2 * np.pi, post_fi)
#     # target_angle_eff = np.where(
#     #     cond_B & (target_angle <= post_fi),
#     #     target_angle + 2 * np.pi,
#     #     target_angle
#     # )
#     target_angle_eff = np.where(
#         cond_B & tgt_lt_post,
#         target_angle + 2 * np.pi,
#         target_angle
#     )
#     span_B = post_fi_eff - pre_fi  # 跨度
#     ratio_B = np.full_like(span_B, np.nan, dtype=np.float64)
#     valid_B = (span_B != 0)
#     mask_B_final = crossing_mask_B & valid_B
#     ratio_B[mask_B_final] = (target_angle_eff[mask_B_final] - pre_fi[mask_B_final]) / span_B[mask_B_final]
#
#     # -------------------- 5) 合并两次判断 --------------------
#     final_crossing_mask = crossing_mask_A | crossing_mask_B
#     crossing_indices = np.where(final_crossing_mask)[0]
#
#     # 把 ratio_A, ratio_B 分别写入 ratio_merge
#     ratio_merge[crossing_mask_A] = ratio_A[crossing_mask_A]
#     ratio_merge[crossing_mask_B] = ratio_B[crossing_mask_B]
#
#     return crossing_indices, ratio_merge

@njit()
def delete_from_bunch(arr, start, end):
    # 创建新数组：保留前段 + 后段
    new_len = arr.shape[0] - (end - start)
    result = np.empty((new_len, arr.shape[1]), dtype=arr.dtype)

    # 拷贝前段
    result[0:start, :] = arr[0:start, :]

    # 拷贝后段
    result[start:, :] = arr[end:, :]

    return result


@njit(parallel=True)
def boris_push_cartesian_njit(x, y, z,
                              vx, vy, vz,
                              Ex, Ey, Ez,
                              Bx, By, Bz,
                              Inj_flag, Survive_flag,
                              r,dt, q, m, c,
                              x_out, y_out, z_out,
                              vx_out, vy_out, vz_out,
                              Aperture_enable, Aperture_m,
                              SurvivedNum_Local, LostNum_Local):
    """
    相对论 Boris 推进器（并行版本，带掩码），结果写入预分配数组。
    """

    N = x.size
    for i in prange(N):
        if Inj_flag[i] != 1 or Survive_flag[i] != 1:
            # 跳过未注入或未存活粒子，位置/速度不变
            x_out[i] = x[i]
            y_out[i] = y[i]
            z_out[i] = z[i]
            vx_out[i] = vx[i]
            vy_out[i] = vy[i]
            vz_out[i] = vz[i]
            continue

        if Aperture_enable:
            if r[i] > Aperture_m[1] or r[i] < Aperture_m[0] or z[i] > Aperture_m[3] or z[i] < Aperture_m[2]:
                # 束损条件
                Survive_flag[i] = 0
                SurvivedNum_Local -= 1
                LostNum_Local += 1

        # === Step 0 ===
        v2 = vx[i]**2 + vy[i]**2 + vz[i]**2
        gamma = 1.0 / np.sqrt(1 - v2 / (c * c))

        px = gamma * m * vx[i]
        py = gamma * m * vy[i]
        pz = gamma * m * vz[i]

        # === Step 1 ===
        px += 0.5 * dt * q * Ex[i]
        py += 0.5 * dt * q * Ey[i]
        pz += 0.5 * dt * q * Ez[i]

        p2 = px**2 + py**2 + pz**2
        gamma_minus = np.sqrt(1 + p2 / (m * m * c * c))
        coeff_b = 0.5 * dt * q / (gamma_minus * m)

        tx = coeff_b * Bx[i]
        ty = coeff_b * By[i]
        tz = coeff_b * Bz[i]

        # p'
        pxp = px + (py * tz - pz * ty)
        pyp = py + (pz * tx - px * tz)
        pzp = pz + (px * ty - py * tx)

        t2 = tx**2 + ty**2 + tz**2
        s = 2.0 / (1.0 + t2)
        sx = s * tx
        sy = s * ty
        sz = s * tz

        # p+
        px += (pyp * sz - pzp * sy)
        py += (pzp * sx - pxp * sz)
        pz += (pxp * sy - pyp * sx)

        # === Step 3 ===
        px += 0.5 * dt * q * Ex[i]
        py += 0.5 * dt * q * Ey[i]
        pz += 0.5 * dt * q * Ez[i]

        p2_new = px**2 + py**2 + pz**2
        gamma_new = np.sqrt(1 + p2_new / (m * m * c * c))

        vx_new = px / (gamma_new * m)
        vy_new = py / (gamma_new * m)
        vz_new = pz / (gamma_new * m)

        # === Step 4 ===
        x_out[i] = x[i] + vx_new * dt
        y_out[i] = y[i] + vy_new * dt
        z_out[i] = z[i] + vz_new * dt
        vx_out[i] = vx_new
        vy_out[i] = vy_new
        vz_out[i] = vz_new


@njit(parallel=True)
def cylindrical_to_cartesian(r, rdot, z, zdot, fi, fidot,
                             E_r, E_z, E_fi, Esc_r, Esc_z, Esc_fi,
                             Br, Bz0, Bfi, RF_Phase,
                             Malloc_x_cart, Malloc_y_cart, Malloc_z_cart,
                             Malloc_vx_cart, Malloc_vy_cart, Malloc_vz_cart,
                             Malloc_Ex_cart, Malloc_Ey_cart, Malloc_Ez_cart,
                             Malloc_Bx_cart, Malloc_By_cart, Malloc_Bz_cart,
                             Malloc_RF_shift, rf_shift, apply_RF=True):

    N = r.shape[0]

    for i in prange(N):
        sinFi = np.sin(fi[i])
        cosFi = np.cos(fi[i])
        sin_RF = np.sin(RF_Phase[i]+rf_shift[Malloc_RF_shift[i]]) if apply_RF else 0.0
        # sin_RF = np.sin(np.pi/6) if apply_RF else 0.0

        # 电场处理
        Ez = E_z[i] * sin_RF
        Er = E_r[i] * sin_RF
        Efi = E_fi[i] * sin_RF

        Ez += Esc_z[i]
        Er += Esc_r[i]
        Efi += Esc_fi[i]

        # 坐标
        Malloc_x_cart[i] = r[i] * cosFi
        Malloc_y_cart[i] = r[i] * sinFi
        Malloc_z_cart[i] = z[i]

        # 速度
        Malloc_vx_cart[i] = rdot[i] * cosFi - r[i] * fidot[i] * sinFi
        Malloc_vy_cart[i] = rdot[i] * sinFi + r[i] * fidot[i] * cosFi
        Malloc_vz_cart[i] = zdot[i]

        # 电场
        Malloc_Ex_cart[i] = (Er * cosFi - Efi * sinFi)
        Malloc_Ey_cart[i] = (Er * sinFi + Efi * cosFi)
        Malloc_Ez_cart[i] = Ez

        # 磁场（取反）
        Malloc_Bx_cart[i] = -(Br[i] * cosFi - Bfi[i] * sinFi)
        Malloc_By_cart[i] = -(Br[i] * sinFi + Bfi[i] * cosFi)
        Malloc_Bz_cart[i] = -Bz0[i]


@njit(parallel=True)
# @profile
def convert_cartesian_to_cylindrical(x_new, y_new, z_new,
                                     vx_new, vy_new, vz_new,
                                     x_old, y_old, fi_old,
                                     mat_in, dRF_phase):
    """
    将推进后结果从笛卡尔坐标系转换回柱坐标，并写入 mat_out。

    参数：
    - x/y/z_new: 推进后的粒子坐标
    - vx/vy/vz_new: 推进后的粒子速度
    - x_old/y_old: 推进前粒子 x/y，用于计算 Δfi
    - fi_old: 推进前粒子的 φ，用于保持连续性
    - mat_out: 要写入的数组，shape=(N, >=6)，对应 r, vr, z, vz, φ, vφ
    """

    N = x_new.shape[0]
    for i in prange(N):
        r_new = np.sqrt(x_new[i]*x_new[i] + y_new[i]*y_new[i])
        fi_before = np.arctan2(y_old[i], x_old[i])
        fi_after  = np.arctan2(y_new[i], x_new[i])
        delta_fi = fi_after - fi_before

        # 修正跨 2π 跳变
        if delta_fi > np.pi:
            delta_fi -= 2 * np.pi
        elif delta_fi < -np.pi:
            delta_fi += 2 * np.pi

        fi_new = fi_old[i] + delta_fi

        # 速度转换
        vr  = vx_new[i] * np.cos(fi_new) + vy_new[i] * np.sin(fi_new)
        vfi = -vx_new[i] * np.sin(fi_new) + vy_new[i] * np.cos(fi_new)

        # 写入 mat_out
        mat_in[i, 0] = r_new
        mat_in[i, 1] = vr
        mat_in[i, 2] = z_new[i]
        mat_in[i, 3] = vz_new[i]
        mat_in[i, 4] = fi_new
        mat_in[i, 5] = vfi / r_new if r_new > 1e-12 else 0.0  # 避免除以零
        mat_in[i, 9] += dRF_phase


# ---------------------------------------
# boris_push_cylindrical_njit工具函数
# ---------------------------------------
@njit
def cross3(a, b):
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])

@njit
def gamma_from_v(v_vec, c):
    v2 = v_vec[0]**2 + v_vec[1]**2 + v_vec[2]**2
    return 1.0 / np.sqrt(1.0 - v2 / c**2)

@njit
def gamma_from_p(p_vec, m, c):
    p2 = np.dot(p_vec, p_vec)
    return np.sqrt(1 + p2 / (m**2 * c**2))

# @profile
@njit(parallel=True)
def boris_push_cylindrical_speed_njit(
    r, z, fi,
    rdot, zdot, fidot,
    E_r, E_z, E_fi, Esc_r, Esc_z, Esc_fi,
    B_r, B_z, B_fi, RF_Phase,
    Inj_flag, Survive_flag,
    dt, q, m, c, mat_in,
    r_new, z_new, fi_new,
    rdot_new, zdot_new, fidot_new, apply_RF=False, apply_space_charge=False):

    Np = r.shape[0]
    for i in prange(Np):

        if Inj_flag[i] != 1 or Survive_flag[i] != 1:
            # 跳过未注入或未存活粒子，位置/速度不变
            continue

        sin_RF = np.sin(RF_Phase[i]) if apply_RF else 0.0

        # 电场处理
        Eztot = E_z[i] * sin_RF
        Ertot = E_r[i] * sin_RF
        Efitot = E_fi[i] * sin_RF

        if apply_space_charge:
            Eztot += Esc_z[i]
            Ertot += Esc_r[i]
            Efitot += Esc_fi[i]

        cos_fi = np.cos(fi[i])
        sin_fi = np.sin(fi[i])
        ar = np.array([cos_fi, sin_fi, 0.0])
        afi = np.array([-sin_fi, cos_fi, 0.0])
        az = np.array([0.0, 0.0, 1.0])

        # v 2 p
        v_vec = rdot[i] * ar + r[i] * fidot[i] * afi + zdot[i] * az
        gamma = gamma_from_v(v_vec, c)
        p_vec = gamma * m * v_vec

        E_vec = Ertot*ar + Efitot*afi + Eztot*az
        B_vec = B_r[i]*ar + B_fi[i]*afi + B_z[i]*az

        # Boris step: Half E
        p_vec += 0.5 * dt * q * E_vec

        # Boris step: B rotation
        gamma = gamma_from_p(p_vec, m, c)
        t = (q * dt / (2 * gamma * m)) * B_vec
        t_mag2 = np.dot(t, t)
        s = 2 * t / (1 + t_mag2)

        p_minus = p_vec
        p_prime = p_minus + cross3(p_minus, t)
        p_plus  = p_minus + cross3(p_prime, s)

        # Boris step: Half E again
        p_vec = p_plus + 0.5 * dt * q * E_vec

        # p 2 v
        gamma = gamma_from_p(p_vec, m, c)
        v_vec = p_vec / (gamma * m)

        # 更新位置
        r_vec = r[i] * ar + dt * v_vec
        x, y = r_vec[0], r_vec[1]
        r_new[i] = np.sqrt(x*x + y*y)
        fi_new[i] = np.arctan2(y, x)
        z_new[i] = z[i] + dt * v_vec[2]

        # 新基底
        cos_fi_new = np.cos(fi_new[i])
        sin_fi_new = np.sin(fi_new[i])
        ar_new = np.array([cos_fi_new, sin_fi_new, 0.0])
        afi_new = np.array([-sin_fi_new, cos_fi_new, 0.0])

        rdot_new[i] = np.dot(v_vec, ar_new)
        fidot_new[i] = np.dot(v_vec, afi_new) / r_new[i]
        zdot_new[i] = np.dot(v_vec, az)

        # 写入 mat_out
        mat_in[i, 0] = r_new[i]
        mat_in[i, 1] = rdot_new[i]
        mat_in[i, 2] = z_new[i]
        mat_in[i, 3] = zdot_new[i]
        mat_in[i, 4] = fi_new[i]
        mat_in[i, 5] = fidot_new[i]


# ---------------------------------------
# FFAG_dump工具函数
# ---------------------------------------
@njit(parallel=True)
def UpdataPreStepNjit(PreStep_Fi, PreStepMat, LocalBunch, Malloc_Step_SurviveFlag):
    """
    并行更新每个粒子的 PreStepMat 和 PreStep_Fi。

    字段索引已固定：
        r         = 0
        vr        = 1
        z         = 2
        vz        = 3
        fi        = 4
        Ek        = 5
        RF_phase  = 9
    """
    for i in prange(LocalBunch.shape[0]):
        PreStep_Fi[i]     = LocalBunch[i, 4]  # fi
        PreStepMat[i, 0]  = LocalBunch[i, 0]  # r
        PreStepMat[i, 1]  = LocalBunch[i, 1]  # vr
        PreStepMat[i, 2]  = LocalBunch[i, 2]  # z
        PreStepMat[i, 3]  = LocalBunch[i, 3]  # vz
        PreStepMat[i, 4]  = LocalBunch[i, 4]  # fi
        PreStepMat[i, 5]  = LocalBunch[i, 5]  # Ek or vf/r
        PreStepMat[i, 6]  = LocalBunch[i, 9]  # RF_phase
        Malloc_Step_SurviveFlag[i] = LocalBunch[i, 8]


# @profile
@njit(parallel=True)
def RK4_equations_of_motion_njit(
    r, z, fi,
    rdot, zdot, Etotal_J,
    E_r, E_z, E_fi, Esc_r, Esc_z, Esc_fi,
    B_r, B_z, B_fi, RF_Phase,
    Inj_flag, Survive_flag,
    q, c, E0_J,
    dxdt, dRF_phase,
    rf_shift, malloc_RFGap_index,
    apply_RF=False):

    Np = r.shape[0]
    for i in prange(Np):

        if Inj_flag[i] != 1 or Survive_flag[i] != 1:
            # 跳过未注入或未存活粒子，位置/速度不变
            # dxdt[i, 0] = 0.0
            # dxdt[i, 1] = 0.0
            # dxdt[i, 2] = 0.0
            # dxdt[i, 3] = 0.0
            # dxdt[i, 4] = 0.0
            # dxdt[i, 5] = 0.0
            dxdt[i, 9] = dRF_phase
            continue

        # gamma = Etotal_J[i] / E0_J
        # beta = np.sqrt(1 - 1 / (gamma ** 2))
        # v = beta * c
        # fidot = np.sqrt(v ** 2 - rdot[i] ** 2 - zdot[i] ** 2) / r[i]

        inv_gamma2 = (E0_J / Etotal_J[i]) ** 2  # 代替 1 / gamma**2
        beta2 = 1.0 - inv_gamma2  # β²
        v2 = beta2 * (c * c)  # v²
        vperp2 = v2 - rdot[i] ** 2 - zdot[i] ** 2
        fidot = np.sqrt(vperp2) / r[i]

        sin_RF = np.sin(RF_Phase[i]+rf_shift[malloc_RFGap_index[i]]) if apply_RF else 0.0

        # 电场处理
        Ez_tot = E_z[i] * sin_RF
        Er_tot = E_r[i] * sin_RF
        Efi_tot = E_fi[i] * sin_RF

        Ez_tot += Esc_z[i]
        Er_tot += Esc_r[i]
        Efi_tot += Esc_fi[i]

        # Ez_tot += z[i] * 2.0e7

        E_dot_v = rdot[i] * Er_tot + (r[i] * fidot) * Efi_tot + zdot[i] * Ez_tot

        dxdt[i, 0] = rdot[i]
        dxdt[i, 1] = (r[i] * fidot * fidot + q * c ** 2 / Etotal_J[i] * (Er_tot - r[i] * fidot * B_z[i] + zdot[i] * B_fi[i])
                      - q * rdot[i] / Etotal_J[i] * E_dot_v)
        dxdt[i, 2] = zdot[i]
        dxdt[i, 3] = (q * c ** 2 / Etotal_J[i] * (Ez_tot + r[i] * fidot * B_r[i] - rdot[i] * B_fi[i])
                      - q * zdot[i] / Etotal_J[i] * E_dot_v)
        dxdt[i, 4] = fidot
        dxdt[i, 5] = q * E_dot_v

        dxdt[i, 9] = dRF_phase


@njit(parallel=True)
def RK4_Post_Step_njit(mat_in, dxdt):
    """
    并行更新bunch的状态矩阵。
    字段索引已固定：
        r         = 0
        vr        = 1
        z         = 2
        vz        = 3
        fi        = 4
        Ek        = 5
        RF_phase  = 9
    """
    Inj_flag = mat_in[:, 7]
    Survive_flag = mat_in[:, 8]

    Np = Inj_flag.shape[0]
    for i in prange(Np):

        if Inj_flag[i] != 1 or Survive_flag[i] != 1:
            # 跳过未注入或未存活粒子，位置/速度不变
            mat_in[i, 9] += dxdt[i, 9]
            continue

        mat_in[i, 0] += dxdt[i, 0]
        mat_in[i, 1] += dxdt[i, 1]
        mat_in[i, 2] += dxdt[i, 2]
        mat_in[i, 3] += dxdt[i, 3]
        mat_in[i, 4] += dxdt[i, 4]
        mat_in[i, 5] += dxdt[i, 5]
        mat_in[i, 9] += dxdt[i, 9]


