import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class FFAG_statics:
    """计算 FFAG 加速器的发射度相关参数"""

    @staticmethod
    def rms_emittance(x, xp):
        """ 计算 RMS 发射度 """
        x_mean = np.mean(x)
        xp_mean = np.mean(xp)

        sigma_xx = np.mean((x - x_mean) ** 2)
        sigma_xpxp = np.mean((xp - xp_mean) ** 2)
        sigma_xxp = np.mean((x - x_mean) * (xp - xp_mean))

        return np.sqrt(sigma_xx * sigma_xpxp - sigma_xxp**2)

    @staticmethod
    def percentile_emittance(x, xp, percentile=99):
        """ 计算指定百分比 (percentile) 的发射度，例如 99% 发射度 """
        data = np.vstack((x, xp))
        cov_matrix = np.cov(data, ddof=1)

        # 计算总发射度
        total_emittance = np.sqrt(np.linalg.det(cov_matrix))

        # 计算相空间半径
        radii = np.sqrt((x - np.mean(x))**2 + (xp - np.mean(xp))**2)

        # 找到99% 粒子的半径阈值
        cutoff_radius = np.percentile(radii, percentile)

        # 选择粒子
        mask = radii <= cutoff_radius
        x_filtered, xp_filtered = x[mask], xp[mask]

        # 计算 RMS 发射度
        return FFAG_statics.rms_emittance(x_filtered, xp_filtered)

    @staticmethod
    def twiss_emittance(beta, alpha, gamma, x, xp):
        """ 计算 Twiss 发射度 """
        x2_mean = np.mean(x**2)
        xp2_mean = np.mean(xp**2)
        xxp_mean = np.mean(x * xp)

        return gamma * x2_mean + 2 * alpha * xxp_mean + beta * xp2_mean

# 测试代码 (增加可视化)
if __name__ == "__main__":
    np.random.seed(42)
    x_sample = np.random.normal(0, 1e-3, 10000)  # 位置 (m)
    xp_sample = np.random.normal(0, 1e-3, 10000) # 角度 (rad)

    beta, alpha = 1.0, 1.5
    gamma = (1 + alpha**2) / beta  # Twiss 关系

    statics = FFAG_statics()
    rms_emit = statics.rms_emittance(x_sample, xp_sample)
    p99_emit = statics.percentile_emittance(x_sample, xp_sample, percentile=84)
    twiss_emit = statics.twiss_emittance(beta, alpha, gamma, x_sample, xp_sample)

    print(f"RMS Emittance: {rms_emit:.5e}")
    print(f"99% Emittance: {p99_emit:.5e}")
    print(f"Twiss Emittance: {twiss_emit:.5e}")

    # 可视化
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(x_sample, xp_sample, s=5, color='blue', alpha=0.5, label='Particles')

    # 画 4×RMS 发射度椭圆
    ellipse_rms = Ellipse((0, 0), width=4*np.sqrt(rms_emit), height=4*np.sqrt(rms_emit/beta),
                          edgecolor='red', facecolor='none', linestyle='--', linewidth=2, label='4x RMS Emittance')

    # 画 4×99% 发射度椭圆
    ellipse_p99 = Ellipse((0, 0), width=4*np.sqrt(p99_emit), height=4*np.sqrt(p99_emit/beta),
                          edgecolor='green', facecolor='none', linestyle='-', linewidth=2, label='4x 99% Emittance')

    ax.add_patch(ellipse_rms)
    ax.add_patch(ellipse_p99)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("x' (rad)")
    ax.legend()
    ax.grid(True)
    plt.title("Phase Space Distribution with 4x Emittance Ellipses")
    plt.show()
