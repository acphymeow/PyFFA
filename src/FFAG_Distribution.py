import os
import sys
import math
import random
import re
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from FFAG_MathTools import FFAG_interpolation
from FFAG_ParasAndConversion import FFAG_ConversionTools, FFAG_GlobalParameters
from FFAG_Utils import FFAG_GeometryCalc, FFAG_SegmentTools
from FFAG_Bunch import FFAG_ManageBunchAttribute
from FFAG_bucket import FFAG_bucket


class TwissContainer:
    """
	Keeps the twiss parameters alpha, beta and the emittance.
	Calculates the normalized value u**2+(alpha*u + beta*u')**2)/(beta*emittance),
	which is (gamma*u**2+2*alpha*u*u'+beta*u'**2)/(emittance).
	Translates the normalized values u and up to the non-normalized ones.
	"""

    def __init__(self, alpha, beta, emittance):
        self.alpha = alpha
        self.beta = beta
        self.gamma = (1.0 + self.alpha ** 2) / self.beta
        self.emittance = emittance
        self.__initialize()

    def __initialize(self):
        self.u_max = math.sqrt(self.beta * self.emittance)
        self.up_coeff = math.sqrt(self.emittance / self.beta)
        self.gamma = (1.0 + self.alpha ** 2) / self.beta
        self.up_max = math.sqrt(self.gamma * self.emittance)

    def setEmittance(self, emittance):
        self.emittance = emittance
        self.__initialize()

    def getNormalizedH(self, u, up):
        """
		Returns (u**2+(alpha*u + beta*u')**2)/(beta*emittance) 
		"""
        return (u ** 2 + (self.alpha * u + self.beta * up) ** 2) / (self.beta * self.emittance)

    def getU_Max(self):
        """
		Returns the maximal value of u.
		"""
        return self.u_max

    def getUP_Max(self):
        """
		Returns the maximal value of uprime.
		"""
        return self.up_max

    def getU_UP(self, u_norm, up_norm):
        """
		Returns the coordinate and momentum for normlized ones
		u = sqrt(beta*emittance)*u_norm
		up = sqrt(emittance/beta)*(up_norm - alpha*u_norm)
		"""
        u = self.u_max * u_norm
        up = self.up_coeff * (up_norm - self.alpha * u_norm)
        return (u, up)

    def getAlphaBetaGammaEmitt(self):
        return (self.alpha, self.beta, self.gamma, self.emittance)

    def getAlphaBetaEmitt(self):
        return (self.alpha, self.beta, self.emittance)


class KVDist1D:
    """
	Generates the 1D KV-distribution. The input emittance in the TwissContainer
	is a rms emittance. The generated distribution will give the same value. Remember
	that 100% emittance is 2 times bigger for 1D KV distribution.
	"""

    def __init__(self, twiss=TwissContainer(0., 1., 1.)):
        """ Constructor """
        (alpha, beta, emittance) = twiss.getAlphaBetaEmitt()
        self.twiss = TwissContainer(alpha, beta, 2 * emittance)
        self.sign_choices = (-1., 1.)
        self.emit_times = 2.0

    def getCoordinates(self):
        """ Return (u,up) distributed for the 1D KV-distribution. """
        x_norm = math.sin(2 * math.pi * (random.random() - 0.5))
        xp_norm = random.choice(self.sign_choices) * math.sqrt(1.0 - x_norm ** 2)
        return self.twiss.getU_UP(x_norm, xp_norm)

    def getTwissContainers(self):
        """ Returns the twiss container. """
        return (self.twiss,)


class WaterBagDist1D:
    """
	Generates the Water Bag 1D distribution.The input emittance in the TwissContainer
	is a rms emittance. The generated distribution will give the same value. Remember
	that 100% emittance is 4 times bigger for 1D WaterBag distribution. 
	"""

    def __init__(self, twiss=TwissContainer(0., 1., 1.)):
        """ Constructor """
        (alpha, beta, emittance) = twiss.getAlphaBetaEmitt()
        twiss = TwissContainer(alpha, beta, 2 * emittance)
        self.kv_dist = KVDist1D(twiss)
        self.emit_times = 2.0

    def getCoordinates(self):
        """ Return (u,up) distributed for the 1D WaterBag-distribution. """
        (u, up) = self.kv_dist.getCoordinates()
        g = math.sqrt(random.random())
        return (g * u, g * up)

    def getTwissContainers(self):
        """ Returns the twiss container. """
        return self.kv_dist.getTwissContainers()


class KVDist2D:
    """
	Generates the 2D KV-distribution. The input emittance in the TwissContainer
	is a rms emittance. The generated distribution will give the same value. Remember
	that 100% emittance is 4 times bigger for 2D KV distribution.
	"""

    def __init__(self, twissX=TwissContainer(0., 1., 1.), twissY=TwissContainer(0., 1., 1.)):
        """ Constructor """
        (alpha_x, beta_x, emittance_x) = twissX.getAlphaBetaEmitt()
        (alpha_y, beta_y, emittance_y) = twissY.getAlphaBetaEmitt()
        self.twissX = TwissContainer(alpha_x, beta_x, 4 * emittance_x)
        self.twissY = TwissContainer(alpha_y, beta_y, 4 * emittance_y)
        self.emit_times = 4.0

    def getCoordinates(self):
        """ Return (x,xp,y,yp) distributed for the 2D KV-distribution. """
        # x-y plane
        phi = 2 * math.pi * (random.random() - 0.5)
        rho = math.sqrt(random.random())
        x_norm = rho * math.cos(phi)
        y_norm = rho * math.sin(phi)
        # momentum
        p0 = math.sqrt(math.fabs(1. - rho ** 2))
        phi = 2 * math.pi * (random.random() - 0.5)
        xp_norm = p0 * math.cos(phi)
        yp_norm = p0 * math.sin(phi)
        (x, xp) = self.twissX.getU_UP(x_norm, xp_norm)
        (y, yp) = self.twissY.getU_UP(y_norm, yp_norm)
        return (x, xp, y, yp)

    def getTwissContainers(self):
        """ Returns the (twissX,twissY) containers. """
        return (self.twissX, self.twissY)


class WaterBagDist2D:
    """
	Generates the Water Bag 2D distribution. The input emittance in the TwissContainer
	is a rms emittance. The generated distribution will give the same value. Remember
	that 100% emittance is 6 times bigger for 2D WaterBag distribution. 
	"""

    def __init__(self, twissX=TwissContainer(0., 1., 1.), twissY=TwissContainer(0., 1., 1.)):
        """ Constructor """
        (alpha_x, beta_x, emittance_x) = twissX.getAlphaBetaEmitt()
        (alpha_y, beta_y, emittance_y) = twissY.getAlphaBetaEmitt()
        twissX = TwissContainer(alpha_x, beta_x, 6.0 * emittance_x / 4.0)
        twissY = TwissContainer(alpha_y, beta_y, 6.0 * emittance_y / 4.0)
        self.kv_dist = KVDist2D(twissX, twissY)
        self.emit_times = 6.0  # only for plotting

    def getCoordinates(self):
        """ Return (x,xp,y,yp) distributed for the 2D WaterBag-distribution. """
        (x, xp, y, yp) = self.kv_dist.getCoordinates()
        g = math.sqrt(math.sqrt(random.random()))
        return (g * x, g * xp, g * y, g * yp)

    def getTwissContainers(self):
        """ Returns the (twissX,twissY) containers. """
        return self.kv_dist.getTwissContainers()


class KVDist3D:
    """
	Generates the 3D KV-distribution.The input emittance in the TwissContainer
	is a rms emittance. The generated distribution will give the same value. Remember
	that 100% emittance is 6 times bigger for 3D KV distribution.
	"""

    def __init__(self, twissX=TwissContainer(0., 1., 1.), twissY=TwissContainer(0., 1., 1.),
                 twissZ=TwissContainer(0., 1., 1.)):
        """ Constructor """
        (alpha_x, beta_x, emittance_x) = twissX.getAlphaBetaEmitt()
        (alpha_y, beta_y, emittance_y) = twissY.getAlphaBetaEmitt()
        (alpha_z, beta_z, emittance_z) = twissZ.getAlphaBetaEmitt()
        self.twissX = TwissContainer(alpha_x, beta_x, 6 * emittance_x)
        self.twissY = TwissContainer(alpha_y, beta_y, 6 * emittance_y)
        self.twissZ = TwissContainer(alpha_z, beta_z, 6 * emittance_z)
        self.emit_times = 6.0

    def getCoordinates(self):
        """ Return (x,xp,y,yp,z,zp) distributed for the 3D KV-distribution. """
        # x-y-z-zp plane
        n_limit = 1000
        n_count = 0
        pxy2 = 0.
        x_norm = 0.
        y_norm = 0.
        z_norm = 0.
        zp_norm = 0.
        while (1 < 2):
            n_count = n_count + 1
            x_norm = 2 * (random.random() - 0.5)
            y_norm = 2 * (random.random() - 0.5)
            z_norm = 2 * (random.random() - 0.5)
            zp_norm = 2 * (random.random() - 0.5)
            pxy2 = 1.0 - x_norm ** 2 - y_norm ** 2 - z_norm ** 2 - zp_norm ** 2
            if pxy2 > 0.:
                break
            if n_count > n_limit:
                print("KVDist3D generator has a problem with Python random module!")
                print("Stop.")
                sys.exit(1)
        # make xp-yp plane
        pxy = math.sqrt(pxy2)
        phi = 2 * math.pi * (random.random() - 0.5)
        xp_norm = pxy * math.cos(phi)
        yp_norm = pxy * math.sin(phi)
        (x, xp) = self.twissX.getU_UP(x_norm, xp_norm)
        (y, yp) = self.twissY.getU_UP(y_norm, yp_norm)
        (z, zp) = self.twissZ.getU_UP(z_norm, zp_norm)
        return x, xp, y, yp, z, zp

    def getTwissContainers(self):
        """ Returns the (twissX,twissY,wissZ) containers. """
        return self.twissX, self.twissY, self.twissZ


class WaterBagDist3D:
    """
	Generates the Water Bag 3D distribution. The input emittance in the TwissContainer
	is a rms emittance. The generated distribution will give the same value. Remember
	that 100% emittance is 8 times bigger for 3D WaterBag distribution. 
	"""

    def __init__(self, twissX=TwissContainer(0., 1., 1.), twissY=TwissContainer(0., 1., 1.),
                 twissZ=TwissContainer(0., 1., 1.)):
        """ Constructor """
        (alpha_x, beta_x, emittance_x) = twissX.getAlphaBetaEmitt()
        (alpha_y, beta_y, emittance_y) = twissY.getAlphaBetaEmitt()
        (alpha_z, beta_z, emittance_z) = twissZ.getAlphaBetaEmitt()
        twissX = TwissContainer(alpha_x, beta_x, 8.0 * emittance_x / 6.0)
        twissY = TwissContainer(alpha_y, beta_y, 8.0 * emittance_y / 6.0)
        twissZ = TwissContainer(alpha_z, beta_z, 8.0 * emittance_z / 6.0)
        self.kv_dist = KVDist3D(twissX, twissY, twissZ)
        self.emit_times = 8.0 / 6.0

    def getCoordinates(self):
        """ Return (x,xp,y,yp,z,zp) distributed for the 3D WaterBag-distribution. """
        (x, xp, y, yp, z, zp) = self.kv_dist.getCoordinates()
        g = math.pow(random.random(), 1. / 6.)
        return (g * x, g * xp, g * y, g * yp, g * z, g * zp)

    def getTwissContainers(self):
        """ Returns the (twissX,twissY,wissZ) containers. """
        return self.kv_dist.getTwissContainers()


class GaussDist1D:
    """
	Generates the 1D Gauss distribution. exp(-x**2/(2*sigma**2)) The cut_off value is x_cutoff/sigma.
	"""

    def __init__(self, twiss=TwissContainer(0., 1., 1.), cut_off=-1.):
        """ Constructor """
        self.twiss = twiss
        self.cut_off = cut_off
        self.cut_off2 = cut_off * cut_off
        self.emit_times = 1.0

    def getCoordinates(self):
        """ Return (u,up) distributed for the 1D Gauss distribution. """
        x_norm = random.gauss(0., 1.0)
        xp_norm = random.gauss(0., 1.0)
        if (self.cut_off > 0.):
            while ((x_norm ** 2 + xp_norm ** 2) > self.cut_off2):
                x_norm = random.gauss(0., 1.0)
                xp_norm = random.gauss(0., 1.0)
        return self.twiss.getU_UP(x_norm, xp_norm)

    def getTwissContainers(self):
        """ Returns the twiss container. """
        return (self.twiss,)


class GaussDist2D:
    """
	Generates the 2D Gauss distribution. exp(-x**2/(2*sigma**2)) The cut_off value is x_cutoff/sigma.
	"""

    def __init__(self, twissX=TwissContainer(0., 1., 1.), twissY=TwissContainer(0., 1., 1.), cut_off=-1.):
        """ Constructor """
        self.twissX = twissX
        self.twissY = twissY
        self.gaussX = GaussDist1D(twissX, cut_off)
        self.gaussY = GaussDist1D(twissY, cut_off)
        self.cut_off = cut_off
        self.emit_times = 1.0

    def getCoordinates(self):
        """ Return (u,up) distributed for the 2D Gauss distribution. """
        (x, xp) = self.gaussX.getCoordinates()
        (y, yp) = self.gaussY.getCoordinates()
        return (x, xp, y, yp)

    def getTwissContainers(self):
        """ Returns the (twissX,twissY) containers. """
        return (self.twissX, self.twissY)


class GaussDist3D:
    """
	Generates the 3D Gauss distribution. exp(-x**2/(2*sigma**2)) The cut_off value is x_cutoff/sigma.
	"""

    def __init__(self, twissX=TwissContainer(0., 1., 1.), \
                 twissY=TwissContainer(0., 1., 1.), \
                 twissZ=TwissContainer(0., 1., 1.), \
                 cut_off=-1.):
        """ Constructor """
        self.twissX = twissX
        self.twissY = twissY
        self.twissZ = twissZ
        self.gaussX = GaussDist1D(twissX, cut_off)
        self.gaussY = GaussDist1D(twissY, cut_off)
        self.gaussZ = GaussDist1D(twissZ, cut_off)
        self.cut_off = cut_off
        self.emit_times = 1.0

    def getCoordinates(self):
        """ Return (u,up) distributed for the 3D Gauss distribution. """
        (x, xp) = self.gaussX.getCoordinates()
        (y, yp) = self.gaussY.getCoordinates()
        (z, zp) = self.gaussZ.getCoordinates()
        return (x, xp, y, yp, z, zp)

    def getTwissContainers(self):
        """ Returns the (twissX,twissY,twissZ) containers. """
        return (self.twissX, self.twissY, self.twissZ)


# --------------------------------------------------
# Auxilary classes 
# --------------------------------------------------

class TwissAnalysis:
    """
	Calculates the rms twiss parameters for 1D,2D, and 3D distributions by 
	using the set of (x,xp), (x,xp,y,yp), and (x,xp,y,yp,z,zp) points.
	There is a c++ replacement for this class BunchTwissAnalysis in the orbit/BunchDiagnostics dir.
	"""

    def __init__(self, nD):
        self.nD = nD
        self.x2_avg_v = []
        self.xp2_avg_v = []
        self.x_xp_avg_v = []
        self.x_avg_v = []
        self.xp_avg_v = []
        self.xp_max_v = []
        self.x_max_v = []
        self.xp_min_v = []
        self.x_min_v = []
        for i in range(self.nD):
            self.x2_avg_v.append(0.)
            self.xp2_avg_v.append(0.)
            self.x_xp_avg_v.append(0.)
            self.x_avg_v.append(0.)
            self.xp_avg_v.append(0.)
            self.xp_max_v.append(-1.0e+38)
            self.x_max_v.append(-1.0e+38)
            self.xp_min_v.append(1.0e+38)
            self.x_min_v.append(1.0e+38)
        self.Np = 0

    def init(self):
        """
		Initilizes the analysis.
		"""
        self.x2_avg_v = []
        self.xp2_avg_v = []
        self.x_xp_avg_v = []
        self.x_avg_v = []
        self.xp_avg_v = []
        self.xp_max_v = []
        self.x_max_v = []
        self.xp_min_v = []
        self.x_min_v = []
        for i in range(self.nD):
            self.x2_avg_v.append(0.)
            self.xp2_avg_v.append(0.)
            self.x_xp_avg_v.append(0.)
            self.x_avg_v.append(0.)
            self.xp_avg_v.append(0.)
            self.xp_max_v.append(-1.0e+38)
            self.x_max_v.append(-1.0e+38)
            self.xp_min_v.append(1.0e+38)
            self.x_min_v.append(1.0e+38)
        self.Np = 0

    def account(self, arr_v):
        """
        Accounts the data. The arr_v should be a list of 2, 4, or 6 size.
        """
        for i in range(self.nD):
            self.x_avg_v[i] = self.x_avg_v[i] + arr_v[i * 2]
            self.xp_avg_v[i] = self.xp_avg_v[i] + arr_v[i * 2 + 1]
            self.x2_avg_v[i] = self.x2_avg_v[i] + (arr_v[i * 2]) ** 2
            self.xp2_avg_v[i] = self.xp2_avg_v[i] + (arr_v[i * 2 + 1]) ** 2
            self.x_xp_avg_v[i] = self.x_xp_avg_v[i] + arr_v[i * 2 + 1] * arr_v[i * 2]
            x = arr_v[i * 2]
            xp = arr_v[i * 2 + 1]
            if self.x_max_v[i] < x:
                self.x_max_v[i] = x
            if self.xp_max_v[i] < xp:
                self.xp_max_v[i] = xp
            if self.x_min_v[i] > x:
                self.x_min_v[i] = x
            if self.xp_min_v[i] > xp:
                self.xp_min_v[i] = xp

        self.Np += 1

    def getTwiss(self, d):
        """
        Returns the twiss parameters in the array
        [alpha,beta,gamma, emittance] for the dimension d.
        """
        if d < 0 or d >= self.nD:
            print("Dimension n=" + str(d) + " does not exist!")
            sys.exit(1)
        if self.Np == 0:
            return 0., 0., 0.
        x_avg = self.x_avg_v[d]
        xp_avg = self.xp_avg_v[d]
        x2_avg = self.x2_avg_v[d]
        xp2_avg = self.xp2_avg_v[d]
        x_xp_avg = self.x_xp_avg_v[d]
        x_avg = x_avg / self.Np
        xp_avg = xp_avg / self.Np
        x2_avg = x2_avg / self.Np - x_avg * x_avg
        x_xp_avg = x_xp_avg / self.Np - x_avg * xp_avg
        xp2_avg = xp2_avg / self.Np - xp_avg * xp_avg

        emitt_rms = math.sqrt(x2_avg * xp2_avg - x_xp_avg * x_xp_avg)
        beta = x2_avg / emitt_rms
        alpha = - x_xp_avg / emitt_rms
        gamma = xp2_avg / emitt_rms
        return alpha, beta, gamma, emitt_rms

    def getAvgU_UP(self, d):
        """
        Returns the (u_avg,up_avg) parameters in the array
        [u,up] for the dimension d.
        """
        if d < 0 or d >= self.nD:
            print("Dimension n=" + str(d) + " does not exist!")
            sys.exit(1)
        if self.Np == 0: return 0., 0.
        x_avg = self.x_avg_v[d] / self.Np
        xp_avg = self.xp_avg_v[d] / self.Np
        return x_avg, xp_avg

    def getRmsU_UP(self, d):
        """
        Returns the (rms u,rms up) parameters in the array
        [u,up] for the dimension d.
        """
        if d < 0 or d >= self.nD:
            print("Dimension n=" + str(d) + " does not exist!")
            sys.exit(1)
        if self.Np == 0 or self.Np == 1: return (0., 0.)
        x2_rms = math.sqrt(math.fabs((self.x2_avg_v[d] - (self.x_avg_v[d]) ** 2 / self.Np) / (self.Np - 1)))
        xp2_rms = math.sqrt(math.fabs((self.xp2_avg_v[d] - (self.xp_avg_v[d]) ** 2 / self.Np) / (self.Np - 1)))
        return (x2_rms, xp2_rms)

    def getMaxU_UP(self, d):
        """
		Returns the (u_max,up_max) parameters in the array 
		[u,up] for the dimension d.
		"""
        if (d < 0 or d >= self.nD):
            print("Dimension n=" + str(d) + " does not exist!")
            sys.exit(1)
        if (self.Np == 0): return (0., 0.)
        return (self.x_max_v[d], self.xp_max_v[d])

    def getMinU_UP(self, d):
        """
		Returns the (u_min,up_min) parameters in the array 
		[u,up] for the dimension d.
		"""
        if d < 0 or d >= self.nD:
            print("Dimension n=" + str(d) + " does not exist!")
            sys.exit(1)
        if self.Np == 0: return 0., 0.
        return self.x_min_v[d], self.xp_min_v[d]


class HollowWaterBagDist1D:
    """
    Generates a hollow Water Bag 1D distribution. The input emittance in the TwissContainer
    is a rms emittance. The generated distribution will give the same value. Remember
    that 100% emittance is 2 times bigger for 1D WaterBag distribution.
    This class generates a hollow distribution by limiting the scaling factor g between 0.9 and 1.
    """

    def __init__(self, twiss=TwissContainer(0., 1., 1.)):
        """ Constructor """
        (alpha, beta, emittance) = twiss.getAlphaBetaEmitt()
        twiss = TwissContainer(alpha, beta, emittance/2)  # max emit at the boundary
        self.kv_dist = KVDist1D(twiss)

    def getCoordinates(self):
        """ Return (u,up) distributed for the 1D hollow WaterBag-distribution. """
        (u, up) = self.kv_dist.getCoordinates()
        g = random.uniform(0.4, 1.0)  # 缩放因子 g 在 0.9 到 1.0 之间
        return (g * u, g * up)

    def getTwissContainers(self):
        """ Returns the twiss container. """
        return self.kv_dist.getTwissContainers()


class IndependentHollowWaterBagDist2D:
    """
    Generates a 2D distribution where x-xp and y-yp are independently distributed according
    to hollow 1D WaterBag distributions. Additionally, the (x, y) points are filtered to lie within
    the maximal ellipse in the x-y real plane.
    """

    def __init__(self,
                 twissX=TwissContainer(0., 1., 1.),
                 twissY=TwissContainer(0., 1., 1.)):
        """ Constructor """
        self.hollow_x = HollowWaterBagDist1D(twissX)
        self.hollow_y = HollowWaterBagDist1D(twissY)

    def getCoordinates(self):
        """
        Return (x, xp, y, yp) distributed for two independent 1D hollow WaterBag-distributions
        in x-xp and y-yp planes, with additional filtering to ensure points are inside the x-y ellipse.
        """
        while True:
            # Generate x-xp and y-yp planes independently
            x, xp = self.hollow_x.getCoordinates()
            y, yp = self.hollow_y.getCoordinates()

            # Maximum displacement for x and y
            u_max_x = self.hollow_x.kv_dist.twiss.getU_Max()
            u_max_y = self.hollow_y.kv_dist.twiss.getU_Max()

            # Check if the point (x, y) lies inside the x-y ellipse
            # equation: (x^2 / a^2) + (y^2 / b^2) <= 1
            if 0.3 <= (x ** 2 / u_max_x ** 2) + (y ** 2 / u_max_y ** 2) <= 1:
                return x, xp, y, yp

    def getTwissContainers(self):
        """ Returns the (twissX, twissY) containers. """
        return self.hollow_x.getTwissContainers(), self.hollow_y.getTwissContainers()

class PlotDistribution():

    def __init__(self, ):
        pass

    def plot_kv_1d(self, ):
        kv_1d = KVDist1D(TwissContainer(0., 1., 1.))
        x_vals = []
        xp_vals = []
        y_vals = []
        yp_vals = []

        for _ in range(10000):
            x, xp = kv_1d.getCoordinates()
            x_vals.append(x)
            xp_vals.append(xp)

        for _ in range(10000):
            y, yp = kv_1d.getCoordinates()
            y_vals.append(y)
            yp_vals.append(yp)

        plt.figure()
        plt.scatter(x_vals, xp_vals, s=1)
        plt.title('1D KV Distribution')
        plt.xlabel('x')
        plt.ylabel('xp')

        plt.figure()
        plt.scatter(x_vals, y_vals, s=1)
        plt.title('1D KV Distribution')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.show()


    def plot_kv_2d(self, ):
        kv_2d = KVDist2D(TwissContainer(0., 1., 1.), TwissContainer(0., 1., 1.))
        x_vals = []
        xp_vals = []
        y_vals = []
        yp_vals = []

        for _ in range(10000):
            x, xp, y, yp = kv_2d.getCoordinates()
            x_vals.append(x)
            xp_vals.append(xp)
            y_vals.append(y)
            yp_vals.append(yp)

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.scatter(x_vals, xp_vals, s=1)
        plt.title('2D KV Distribution (x-xp)')
        plt.xlabel('x')
        plt.ylabel('xp')

        plt.subplot(1, 3, 2)
        plt.scatter(y_vals, yp_vals, s=1)
        plt.title('2D KV Distribution (y-yp)')
        plt.xlabel('y')
        plt.ylabel('yp')

        plt.subplot(1, 3, 3)
        plt.scatter(x_vals, y_vals, s=1)
        plt.title('2D KV Distribution (x-y)')
        plt.xlabel('y')
        plt.ylabel('yp')

        plt.show()


    def plot_kv_3d(self, ):
        kv_3d = KVDist3D(TwissContainer(0., 1., 1.), TwissContainer(0., 1., 1.), TwissContainer(0., 1., 1.))
        x_vals = []
        xp_vals = []
        y_vals = []
        yp_vals = []
        z_vals = []
        zp_vals = []

        for _ in range(10000):
            x, xp, y, yp, z, zp = kv_3d.getCoordinates()
            x_vals.append(x)
            xp_vals.append(xp)
            y_vals.append(y)
            yp_vals.append(yp)
            z_vals.append(z)
            zp_vals.append(zp)

        plt.figure(figsize=(12, 4))

        # x-y 投影
        plt.subplot(1, 3, 1)
        plt.scatter(x_vals, y_vals, s=1)
        plt.title('3D KV Distribution Projection (x-y)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)

        # x-z 投影
        plt.subplot(1, 3, 2)
        plt.scatter(x_vals, z_vals, s=1)
        plt.title('3D KV Distribution Projection (x-z)')
        plt.xlabel('x')
        plt.ylabel('z')
        plt.grid(True)

        # y-z 投影
        plt.subplot(1, 3, 3)
        plt.scatter(y_vals, z_vals, s=1)
        plt.title('3D KV Distribution Projection (y-z)')
        plt.xlabel('y')
        plt.ylabel('z')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_ellipse(self, epsilon, alpha, beta, num_points=1000):
        # 计算 gamma
        gamma = (1 + alpha ** 2) / beta

        # 生成 theta 参数
        theta = np.linspace(0, 2 * np.pi, num_points)

        u = np.cos(theta)
        uprime = np.sin(theta)
        # 对于每个theta点，找到对应的R，使其满足beta*y^2+2*alpha*xy+gamma*x^2=epsilon
        FuncValue = beta * uprime ** 2 + 2 * alpha * u * uprime + gamma * u ** 2
        Radius = np.sqrt(4 * epsilon / FuncValue)
        x_ellipse = Radius * np.cos(theta)
        y_ellipse = Radius * np.sin(theta)

        return x_ellipse, y_ellipse

    def calculate_inside_ellipse(self, points_x, points_xp, epsilon, alpha, beta):
        # 计算 gamma
        gamma = (1 + alpha ** 2) / beta

        # 对应椭圆方程 βx'^2 + 2αxx' + γx^2 ≤ ε
        x = points_x
        x_prime = points_xp

        # 椭圆方程的值
        ellipse_values = beta * x_prime ** 2 + 2 * alpha * x * x_prime + gamma * x ** 2

        # 判断散点是否在椭圆内
        inside = ellipse_values <= 4 * epsilon  # 注意这里的发射度因子是 4 * epsilon
        proportion_inside = np.sum(inside) / len(points_x)

        print(f"位于椭圆内部的散点比例: {proportion_inside:.4f}")

        return proportion_inside


    def plot_gauss_1d(self, alpha_val=0.0, beta_val=1.0, emittance_val=1.0, cutoff=-1.0):
        gauss_1d = GaussDist1D(TwissContainer(alpha_val, beta_val, emittance_val), cutoff)
        u_vals = []
        up_vals = []

        for _ in range(20000):
            u, up = gauss_1d.getCoordinates()
            u_vals.append(u)
            up_vals.append(up)

        u_ellipse, up_ellipse = self.plot_ellipse(emittance_val, alpha_val, beta_val)
        print(f"输入发射度: {emittance_val:.2f}, 椭圆面积: {4 * emittance_val:.2f}")
        self.calculate_inside_ellipse(np.array(u_vals), np.array(up_vals), emittance_val, alpha_val, beta_val)

        # 计算分布的协方差矩阵
        x_mean = np.mean(np.array(u_vals))
        px_mean = np.mean(np.array(up_vals))
        x2_mean = np.mean(np.array(u_vals) ** 2)
        px2_mean = np.mean(np.array(up_vals) ** 2)
        xp_mean = np.mean(np.array(u_vals) * np.array(up_vals))

        # 计算RMS发射度
        CalcRMSemit = np.sqrt((x2_mean - x_mean ** 2) * (px2_mean - px_mean ** 2) - (xp_mean - x_mean * px_mean) ** 2)
        print(f"计算RMS发射度: {CalcRMSemit:.6f}")

        plt.figure()
        plt.scatter(u_vals, up_vals, s=1)
        plt.plot(u_ellipse, up_ellipse, color='r', linewidth=2)
        plt.title('1D Gaussian Distribution')
        plt.xlabel('u')
        plt.ylabel('up')
        plt.show()


    def plot_gauss_2d(self, ):
        gauss_2d = GaussDist2D(TwissContainer(0., 1., 1.), TwissContainer(0., 1., 1.))
        x_vals = []
        xp_vals = []
        y_vals = []
        yp_vals = []

        for _ in range(10000):
            x, xp, y, yp = gauss_2d.getCoordinates()
            x_vals.append(x)
            xp_vals.append(xp)
            y_vals.append(y)
            yp_vals.append(yp)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.scatter(x_vals, xp_vals, s=1)
        plt.title('2D Gaussian Distribution (x-xp)')
        plt.xlabel('x')
        plt.ylabel('xp')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.scatter(y_vals, yp_vals, s=1)
        plt.title('2D Gaussian Distribution (y-yp)')
        plt.xlabel('y')
        plt.ylabel('yp')
        plt.grid(True)

        plt.show()


    def plot_gauss_3d(self, ):
        gauss_3d = GaussDist3D(TwissContainer(0., 1., 1.), TwissContainer(0., 1., 1.), TwissContainer(0., 1., 1.))
        x_vals = []
        xp_vals = []
        y_vals = []
        yp_vals = []
        z_vals = []
        zp_vals = []

        for _ in range(10000):
            x, xp, y, yp, z, zp = gauss_3d.getCoordinates()
            x_vals.append(x)
            xp_vals.append(xp)
            y_vals.append(y)
            yp_vals.append(yp)
            z_vals.append(z)
            zp_vals.append(zp)

        plt.figure(figsize=(12, 4))

        # x-y 投影
        plt.subplot(1, 3, 1)
        plt.scatter(x_vals, y_vals, s=1)
        plt.title('3D Gaussian Distribution Projection (x-y)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)

        # x-z 投影
        plt.subplot(1, 3, 2)
        plt.scatter(x_vals, z_vals, s=1)
        plt.title('3D Gaussian Distribution Projection (x-z)')
        plt.xlabel('x')
        plt.ylabel('z')
        plt.grid(True)

        # y-z 投影
        plt.subplot(1, 3, 3)
        plt.scatter(y_vals, z_vals, s=1)
        plt.title('3D Gaussian Distribution Projection (y-z)')
        plt.xlabel('y')
        plt.ylabel('z')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


    def plot_waterbag_1d(self, ):
        waterbag_1d = WaterBagDist1D(TwissContainer(0., 1., 1.))
        u_vals = []
        up_vals = []

        for _ in range(10000):
            u, up = waterbag_1d.getCoordinates()
            u_vals.append(u)
            up_vals.append(up)

        plt.figure()
        plt.scatter(u_vals, up_vals, s=1)
        plt.title('1D WaterBag Distribution')
        plt.xlabel('u')
        plt.ylabel('up')
        plt.grid(True)
        plt.show()


    def plot_waterbag_2d(self, ):
        waterbag_2d = WaterBagDist2D(TwissContainer(0., 1., 1.), TwissContainer(0., 1., 1.))
        x_vals = []
        xp_vals = []
        y_vals = []
        yp_vals = []

        for _ in range(20000):
            x, xp, y, yp = waterbag_2d.getCoordinates()
            x_vals.append(x)
            xp_vals.append(xp)
            y_vals.append(y)
            yp_vals.append(yp)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.scatter(x_vals, xp_vals, s=1)
        plt.title('2D WaterBag Distribution (x-xp)')
        plt.xlabel('x')
        plt.ylabel('xp')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.scatter(y_vals, yp_vals, s=1)
        plt.title('2D WaterBag Distribution (y-yp)')
        plt.xlabel('y')
        plt.ylabel('yp')
        plt.grid(True)

        plt.show()


    def plot_waterbag_3d(self, ):
        waterbag_3d = WaterBagDist3D(TwissContainer(0., 1., 1.), TwissContainer(0., 1., 1.), TwissContainer(0., 1., 1.))
        x_vals = []
        xp_vals = []
        y_vals = []
        yp_vals = []
        z_vals = []
        zp_vals = []

        for _ in range(30000):
            x, xp, y, yp, z, zp = waterbag_3d.getCoordinates()
            x_vals.append(x)
            xp_vals.append(xp)
            y_vals.append(y)
            yp_vals.append(yp)
            z_vals.append(z)
            zp_vals.append(zp)

        plt.figure(figsize=(12, 4))

        # x-y 投影
        plt.subplot(1, 3, 1)
        plt.scatter(x_vals, y_vals, s=1)
        plt.title('3D WaterBag Distribution Projection (x-y)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)

        # x-z 投影
        plt.subplot(1, 3, 2)
        plt.scatter(x_vals, z_vals, s=1)
        plt.title('3D WaterBag Distribution Projection (x-z)')
        plt.xlabel('x')
        plt.ylabel('z')
        plt.grid(True)

        # y-z 投影
        plt.subplot(1, 3, 3)
        plt.scatter(y_vals, z_vals, s=1)
        plt.title('3D WaterBag Distribution Projection (y-z)')
        plt.xlabel('y')
        plt.ylabel('z')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


    def plot_hollow_waterbag_1d(self, ):
        hollow_waterbag_1d = HollowWaterBagDist1D(TwissContainer(1., 1., 1.))
        x_vals = []
        xp_vals = []
        y_vals = []
        yp_vals = []

        for _ in range(10000):
            x, xp = hollow_waterbag_1d.getCoordinates()
            x_vals.append(x)
            xp_vals.append(xp)

        for _ in range(10000):
            y, yp = hollow_waterbag_1d.getCoordinates()
            y_vals.append(y)
            yp_vals.append(yp)

        plt.figure()
        plt.scatter(x_vals, xp_vals, s=1)
        plt.title('1D Hollow WaterBag Distribution')
        plt.xlabel('x')
        plt.ylabel('xp')

        plt.figure()
        plt.scatter(x_vals, y_vals, s=1)
        plt.title('1D Hollow WaterBag Distribution')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def plot_independent_hollow_waterbag_2d(self, ):
        independent_hollow_waterbag_2d = IndependentHollowWaterBagDist2D(
            TwissContainer(0., 1., 1.),
            TwissContainer(0., 1., 1.))
        x_vals = []
        xp_vals = []
        y_vals = []
        yp_vals = []

        for _ in range(10000):
            x, xp, y, yp = independent_hollow_waterbag_2d.getCoordinates()
            x_vals.append(x)
            xp_vals.append(xp)
            y_vals.append(y)
            yp_vals.append(yp)

        plt.figure(figsize=(10, 4))

        # x-xp投影
        plt.subplot(1, 2, 1)
        plt.scatter(x_vals, xp_vals, s=1)
        plt.title('Independent 1D Hollow WaterBag Distribution (x-xp)')
        plt.xlabel('x')
        plt.ylabel('xp')

        # y-yp投影
        plt.subplot(1, 2, 2)
        plt.scatter(y_vals, yp_vals, s=1)
        plt.title('Independent 1D Hollow WaterBag Distribution (y-yp)')
        plt.xlabel('y')
        plt.ylabel('yp')

        # x-y投影（实平面）
        plt.figure()
        plt.scatter(x_vals, y_vals, s=1)
        plt.title('Filtered 2D Hollow WaterBag Distribution (x-y Plane)')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.show()

    def plot_kv_2d_new(self):
        kv_2d = KVDist2D(TwissContainer(1., 2., 3.), TwissContainer(0., 1., 1.))
        x_vals = []
        xp_vals = []
        y_vals = []
        yp_vals = []

        for _ in range(10000):
            x, xp, y, yp = kv_2d.getCoordinates()
            x_vals.append(x)
            xp_vals.append(xp)
            y_vals.append(y)
            yp_vals.append(yp)

        # Twiss parameters and RMS emittance
        alpha_x, beta_x, _, emittance_x = kv_2d.getTwissContainers()[0].getAlphaBetaGammaEmitt()
        alpha_y, beta_y, _, emittance_y = kv_2d.getTwissContainers()[1].getAlphaBetaGammaEmitt()

        # Generate ellipse for x-xp
        ellipse_x, ellipse_xp = self.plot_ellipse(emittance_x/4, alpha_x, beta_x)

        # Generate ellipse for y-yp
        ellipse_y, ellipse_yp = self.plot_ellipse(emittance_y/4, alpha_y, beta_y)

        # Plot x-xp
        plt.figure()
        plt.scatter(x_vals, xp_vals, s=1, label='Particles')
        plt.plot(ellipse_x, ellipse_xp, color='r', linewidth=2, label='4 RMS Ellipse')
        plt.title('2D KV Distribution (x-xp)')
        plt.xlabel('x')
        plt.ylabel('xp')
        plt.legend()
        plt.grid(True)

        # Plot y-yp
        plt.figure()
        plt.scatter(y_vals, yp_vals, s=1, label='Particles')
        plt.plot(ellipse_y, ellipse_yp, color='r', linewidth=2, label='4 RMS Ellipse')
        plt.title('2D KV Distribution (y-yp)')
        plt.xlabel('y')
        plt.ylabel('yp')
        plt.legend()
        plt.grid(True)

        # Plot x-y
        plt.figure()
        plt.scatter(x_vals, y_vals, s=1)
        plt.title('2D KV Distribution (x-y)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)

        plt.show()


class CustomDist3D:
    """
    Generates a 3D distribution with a specified 2D transverse distribution and a 1D longitudinal distribution.
    """

    def __init__(self, dist_transverse, dist_longitudinal):
        """
        Constructor

        Parameters:
        - dist_transverse: an instance of a 2D distribution class for x-x' and y-y' planes
        - dist_longitudinal: an instance of a 1D distribution class for z-z' plane
        """
        self.dist_transverse = dist_transverse
        self.dist_longitudinal = dist_longitudinal

    def getCoordinates(self):
        """
        Return (x, xp, y, yp, z, zp) distributed according to the specified distributions.
        """
        x, xp, y, yp = self.dist_transverse.getCoordinates()
        z, zp = self.dist_longitudinal.getCoordinates()
        return x, xp, y, yp, z, zp

    def getTwissContainers(self):
        """
        Returns the (twissX, twissY, twissZ) containers.
        """
        twiss_transverse = self.dist_transverse.getTwissContainers()
        twiss_longitudinal = self.dist_longitudinal.getTwissContainers()
        return twiss_transverse + (twiss_longitudinal,)


def create_distribution(dist_type, twiss_transverse=None, twiss_longitudinal=None):
    """
    Creates a distribution object based on the specified type.

    Parameters:
    - dist_type: str
        Type of the distribution ('gauss', 'kv', 'waterbag', 'hollow_waterbag')
    - twiss_transverse: tuple of TwissContainer
        Twiss parameters for the transverse distribution (twissX, twissY)
    - twiss_longitudinal: TwissContainer
        Twiss parameters for the longitudinal distribution

    Returns:
    - An instance of a distribution class
    """
    if dist_type == 'gauss':
        if twiss_transverse is not None:
            return GaussDist2D(twiss_transverse[0], twiss_transverse[1])
        else:
            return GaussDist1D(twiss_longitudinal)
    elif dist_type == 'kv':
        if twiss_transverse is not None:
            return KVDist2D(twiss_transverse[0], twiss_transverse[1])
        else:
            return KVDist1D(twiss_longitudinal)
    elif dist_type == 'waterbag':
        if twiss_transverse is not None:
            return WaterBagDist2D(twiss_transverse[0], twiss_transverse[1])
        else:
            return WaterBagDist1D(twiss_longitudinal)
    elif dist_type == 'hollow_waterbag':
        if twiss_transverse is not None:
            return IndependentHollowWaterBagDist2D(twiss_transverse[0], twiss_transverse[1])
        else:
            return HollowWaterBagDist1D(twiss_longitudinal)
    elif dist_type == 'match':
        return GaussDist1D(twiss_longitudinal)

    elif dist_type == 'UniformGauss':
        return GaussDist1D(twiss_longitudinal)

    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")


def GenerateInitDistributionDtInjNew(SEOFileName, InjectionPosition,Ek_c,
                                     Emit, num_points, InjTimeNanoSec,
                                     LongitudeAlpha, LongitudeBeta, LongitudeT,
                                     acc_U0 = 100.0e3,
                                     acc_phi0 = 0.0,
                                     TwissParams=None,
                                     dist_types=('gauss', 'gauss'),
                                     dx_xp_y_yp=(0.0, 0.0, 0.0, 0.0),
                                     Bunch_ID = 0.0,
                                     bucket_obj = None):
    """
    生成用于注入的初始分布。

    参数：
    - SEOFileName: str
        SEO 文件的路径。
    - InjectionPosition: str 或 float
        注入位置数据或注入数据文件的路径。
    - Ek_c: float
        中心动能。
    - Emit: list 或 array
        R, Z, Fi方向的RMS发射度(pi.mm.mrad)。
        和100%发射度的关系：
        1维KV100%发射度--->2*RMS, 2维KV100%发射度--->4*RMS, 3维KV100%发射度--->6*RMS,
        1维水袋100%发射度--->4*RMS, 2维水袋100%发射度--->6*RMS, 3维水袋100%发射度--->8*RMS,
        1维Gauss86%发射度--->4*RMS

    - num_points: int
        生成的点数。
    - TwissParams: tuple 或 None，可选
        手动输入的 Twiss 参数 (BetaZ_inj, BetaR_inj, AlphaZ_inj, AlphaR_inj)。
        如果为 None，函数将从 SEO 文件中加载 Twiss 参数。
    - dist_types: tuple of str, optional
        指定分布类型 (transverse_dist_type, longitudinal_dist_type)。
        可选类型('gauss', 'kv', 'waterbag', 'hollow_waterbag', 'UniformGauss')

    返回：
    - particles: list
        生成的粒子坐标列表，每个元素为 [x, xp, y, yp, z, zp]
    """

    epsilonR, epsilonZ, epsilonFiEk = (
        Emit[0] / 1000.0 / 1000.0,
        Emit[1] / 1000.0 / 1000.0,
        Emit[2])

    # Load SEO data
    SEOiniFileName = os.path.join(SEOFileName, "SEO_ini.txt")
    SEO_R_FileName = os.path.join(SEOFileName, "SEO_r.txt")
    SEO_Pr_FileName = os.path.join(SEOFileName, "SEO_pr.txt")
    SEO_ini, BmapPath, SEOdata = LoadSEOParams(SEOiniFileName, Ek_c)
    SEO_R = np.loadtxt(SEO_R_FileName,skiprows=1)
    SEO_Pr = np.loadtxt(SEO_Pr_FileName,skiprows=1)

    Ek_c = SEO_ini[0]  # MeV

    if isinstance(InjectionPosition, str):
        # 如果 InjectionPosition 是字符串，读取并处理文件
        InjectionCL = np.loadtxt(InjectionPosition, skiprows=1)
        # 给定注入截面 CL 和注入能量 Ek_inj,找到与平衡轨道的交点,得到 r_c 和 fi_c
        fi_c, r_c, pr_c = GetInitRAndFiForInjection(SEO_R, SEO_Pr, InjectionCL, Ek_c)
    elif isinstance(InjectionPosition, float):
        # 如果 InjectionPosition 是 float，直接赋值
        InjectionCL = InjectionPosition
        # 给定注入截面 CL 和注入能量 Ek_inj,找到与平衡轨道的交点,得到 r_c 和 fi_c
        fi_c, r_c, pr_c = GetInitRAndFiForInjection(SEO_R, SEO_Pr, InjectionCL, Ek_c)
    elif isinstance(InjectionPosition, dict):
        # 如果 InjectionPosition 是字典，直接获取 fi, r, pr
        fi_c, r_c, pr_c = InjectionPosition["fi"], InjectionPosition["r"], InjectionPosition["pr"]
    else:
        raise ValueError("Unsupported type for InjectionPosition")

    # 如果提供了 Twiss 参数，则直接使用，否则从 SEO 文件中加载
    if TwissParams is not None:
        BetaZ_inj, BetaR_inj, AlphaZ_inj, AlphaR_inj = TwissParams
    else:
        BetaZ_inj, BetaR_inj, AlphaZ_inj, AlphaR_inj = LoadSEOBetaFunc(SEOFileName, Ek_c, fi_c)

    BetaEk_inj, AlphaEk_inj = LongitudeBeta, LongitudeAlpha
    # BetaZ_inj, BetaR_inj, AlphaZ_inj, AlphaR_inj = 1.0, 1.0, 0.0, 0.0

    # 创建 Twiss 容器, emit in pi*m*rad
    # print(AlphaZ_inj, BetaZ_inj, epsilonZ)

    # 根据用户指定的分布类型创建分布对象
    transverse_dist_type = dist_types[0]
    longitudinal_dist_type = dist_types[1]

    if longitudinal_dist_type == "kv":
        epsilonFiEk *= 2
    # 1维KV100 % 发射度 - -->2 * RMS,
    # 1维水袋100 % 发射度 - -->4 * RMS,
    # 1维Gauss86 % 发射度 - -->4 * RMS
    # 希望给定的分布范围接近实际分布范围，需要将发射度乘以2(2 * RMS--->4 * RMS)
    # BunchPara['LongitudeT'] = 72  # 纵向长度 (ns) waterbag: 100%包络椭圆的轴长。 gauss: 4RMS椭圆的轴长.
    # BunchPara['LongitudeDEk'] = 10  # 纵向能散 (MeV) waterbag: 100%包络椭圆的轴长。 gauss: 4RMS椭圆的轴长
    # 这样72ns和10MeV近似等于100%KV和水袋发射度边界或86%发射度边界

    twiss_R = TwissContainer(AlphaR_inj, BetaR_inj, epsilonR)
    twiss_Z = TwissContainer(AlphaZ_inj, BetaZ_inj, epsilonZ)
    twiss_Fi = TwissContainer(AlphaEk_inj, BetaEk_inj, epsilonFiEk)

    dist_transverse = create_distribution(transverse_dist_type, twiss_transverse=(twiss_R, twiss_Z))
    dist_longitudinal = create_distribution(longitudinal_dist_type, twiss_longitudinal=twiss_Fi)

    # 创建自定义的 3D 分布对象
    custom_dist = CustomDist3D(dist_transverse, dist_longitudinal)

    # 生成粒子分布,中心在(0, 0, 0, 0, 0, 0), 单位m, rad, m, rad, rad, J
    particles7DList = []
    for _ in range(num_points):
        r, pr, z, pz, t_inj, Ek = custom_dist.getCoordinates()
        particles7DList.append([r, pr, z, pz, fi_c, Ek, t_inj])

    # 微分方程中的顺序(r, vr), (z, vz), (fi, dfidt), (t_inj, inj_flag,survive_flag),
    # (rf_phase, Esc_r, Esc_z, Esc_fi)
    # (Bunch_ID, Local_ID, Global_ID)
    particles_7D = np.array(particles7DList)

    if longitudinal_dist_type == "match":
        if bucket_obj is None:
            bucket_obj = FFAG_bucket(Bmapfoldname=BmapPath, Ek0=Ek_c, fi0=acc_phi0, U0=acc_U0, step_t=20e-9, step_N=18000)
        t_ns, Ek_MeV, H_thr = bucket_obj.scatter_filter_by_span(num_points, LongitudeT, show=False)
        particles_7D[:,5] = Ek_MeV - Ek_c
        particles_7D[:,6] = t_ns

    elif longitudinal_dist_type == "UniformGauss":
        sigma_Ek = math.sqrt(epsilonFiEk / BetaEk_inj)
        # 2) 时间（相位）均匀分布, 单位 ns
        #    令 bunch 长度为 LongitudeT
        t_ns = np.random.uniform(
            low=-0.5 * LongitudeT,
            high=+0.5 * LongitudeT,
            size=num_points
        )

        # 3) 能量偏差 (Ek - Ek_c) 高斯分布
        dEk = np.random.normal(
            loc=0.0,
            scale=sigma_Ek,
            size=num_points
        )

        # 4) 写回粒子数组
        particles_7D[:, 5] = dEk  # 注意后面有 +Ek_c
        particles_7D[:, 6] = t_ns

    r_c_paint = r_c + dx_xp_y_yp[0]
    pr_c_paint = pr_c + dx_xp_y_yp[1]
    z_c_paint = dx_xp_y_yp[2]
    pz_c_paint = dx_xp_y_yp[3]
    Ek_c_paint = Ek_c
    T_c_paint = InjTimeNanoSec

    particles_7D[:, 0] += r_c_paint
    particles_7D[:, 1] += pr_c_paint
    particles_7D[:, 2] += z_c_paint
    particles_7D[:, 3] += pz_c_paint
    particles_7D[:, 5] += Ek_c  # Ek_MeV
    particles_7D[:, 6] += InjTimeNanoSec

    particles_InjectFlag = np.zeros((num_points,))
    particles_SurviveFlag = np.zeros((num_points, ))
    particles_RFphase = np.zeros((num_points, ))
    particles_ErSC = np.zeros((num_points, ))
    particles_EzSC = np.zeros((num_points, ))
    particles_EfSC = np.zeros((num_points, ))
    particles_BunchID = np.ones((num_points, )) * Bunch_ID
    particles_LocalID = np.arange(0, num_points)
    particles_GlobalID = np.arange(0, num_points)

    BunchAttribute = FFAG_ManageBunchAttribute()
    BunchAttributeNum = BunchAttribute.get_num_attributes()
    IniBunchDist = np.zeros((num_points, BunchAttributeNum))

    IniBunchDist[:, BunchAttribute.Attribute['r']] = particles_7D[:, 0]
    IniBunchDist[:, BunchAttribute.Attribute['vr']] = particles_7D[:, 1]
    IniBunchDist[:, BunchAttribute.Attribute['z']] = particles_7D[:, 2]
    IniBunchDist[:, BunchAttribute.Attribute['vz']] = particles_7D[:, 3]
    IniBunchDist[:, BunchAttribute.Attribute['fi']] = particles_7D[:, 4]
    IniBunchDist[:, BunchAttribute.Attribute['Ek']] = particles_7D[:, 5]
    IniBunchDist[:, BunchAttribute.Attribute['inj_t']] = particles_7D[:, 6]
    IniBunchDist[:, BunchAttribute.Attribute['Inj_flag']] = particles_InjectFlag
    IniBunchDist[:, BunchAttribute.Attribute['Survive']] = particles_SurviveFlag
    IniBunchDist[:, BunchAttribute.Attribute['RF_phase']] = particles_RFphase
    IniBunchDist[:, BunchAttribute.Attribute['Esc_r']] = particles_ErSC
    IniBunchDist[:, BunchAttribute.Attribute['Esc_z']] = particles_EzSC
    IniBunchDist[:, BunchAttribute.Attribute['Esc_fi']] = particles_EfSC
    IniBunchDist[:, BunchAttribute.Attribute['Bunch_ID']] = particles_BunchID
    IniBunchDist[:, BunchAttribute.Attribute['Local_ID']] = particles_LocalID
    IniBunchDist[:, BunchAttribute.Attribute['Global_ID']] = particles_GlobalID

    return IniBunchDist, custom_dist, (r_c_paint, pr_c_paint, z_c_paint, pz_c_paint, Ek_c_paint, T_c_paint), bucket_obj


def GetInitRAndFiForInjection(SEO_R, SEO_Pr, InjectCL, Ek):

    # 1) 读取不同能量值的SEO Trajectories
    # 2) 给定Ek, 插值得到Ek的SEO Trajectory
    # 3) 读取注入central line
    # 4) 找到SEO Trajectory和central line交点的r0, fi0作为注入点中心

    # test code: test_SEO_InjCL_intersect.py

    EK_Axis = SEO_R[1:, 0]
    R_values = SEO_R[1:, 1:]
    Fi_Axis = SEO_R[0, 1:]
    Pr_values = SEO_Pr[1:, 1:]

    if isinstance(InjectCL, np.ndarray):
        # if InjectCL is a segment:

        Func_FiR_SEO = FFAG_interpolation().My1p5DInterp(EK_Axis, R_values)
        Func_FiPr_SEO = FFAG_interpolation().My1p5DInterp(EK_Axis, Pr_values)

        FiR_interp, _ = Func_FiR_SEO(np.array([Ek, ]))
        FiPr_interp, _ = Func_FiPr_SEO(np.array([Ek, ]))

        x_interp = FiR_interp[0, :] * np.cos(Fi_Axis)
        y_interp = FiR_interp[0, :] * np.sin(Fi_Axis)

        trajectory_xy = np.column_stack((x_interp, y_interp))
        trajectory_xy = np.row_stack((trajectory_xy, trajectory_xy[1, :]))

        SEO_segment = FFAG_SegmentTools().CentralLine2Segments(trajectory_xy)
        CL_segment = FFAG_SegmentTools().CentralLine2Segments(InjectCL)

        # intersection_matrix, r_CentralStep, _, _, _, _ = CheckIntersect_njit_ParticleCoord(r_PreStep, r_PostStep, segment_group_b)

        Flag_reshape, D_A1_B_matrix, D_A2_B_matrix, D_B1_A_matrix, D_B2_A_matrix = (
            FFAG_GeometryCalc().FindIntersectionVect(SEO_segment, CL_segment))

        CrossXindex, CrossYindex = np.where(Flag_reshape)
        if len(CrossXindex) > 1:
            CrossXindex, CrossYindex = np.array([CrossXindex[0]]), np.array([CrossYindex[0]])

        PreStepLength = np.abs(D_A1_B_matrix[CrossXindex, CrossYindex])
        PostStepLength = np.abs(D_A2_B_matrix[CrossXindex, CrossYindex])
        PreStepPoint, PostStepPoint = SEO_segment[CrossXindex, 0:2], SEO_segment[CrossXindex, 2:4]

        StepTotalLength = PreStepLength + PostStepLength

        IntersectionPoint = ((PreStepLength - StepTotalLength) / (0 - StepTotalLength) * PreStepPoint +
                             (PreStepLength - 0) / (StepTotalLength - 0) * PostStepPoint)

        theta_rad, r, _ = FFAG_ConversionTools().xy2rfi_m2p180(
            IntersectionPoint[0, 0], IntersectionPoint[0, 1])

        Pr_cross_point = FiPr_interp[0, CrossXindex][0]

        # plt.figure()
        # plt.plot(trajectory_xy[:, 0], trajectory_xy[:, 1], color='blue')
        # plt.plot(InjectCL[:, 0], InjectCL[:, 1], color='red')
        # plt.scatter(IntersectionPoint[0, 0], IntersectionPoint[0, 1], s=10, color='black')
        # plt.show()
    else:
        # if InjectCL is a value:
        theta_rad = np.array([np.deg2rad(InjectCL),])
        r = FFAG_interpolation().Lagrange_interp_2D_vect(EK_Axis, Fi_Axis, R_values, np.array([Ek,]), theta_rad)[0][0]
        Pr_cross_point = FFAG_interpolation().Lagrange_interp_2D_vect(EK_Axis, Fi_Axis, Pr_values, np.array([Ek,]), theta_rad)[0][0]


    return theta_rad[0], r, Pr_cross_point


def LoadSEOParams(SEOFileName, Ek_c):

    path = os.path.dirname(os.path.dirname(SEOFileName))
    SEOdata = np.loadtxt(SEOFileName, skiprows=2)
    Ek0, r0, pr0, T0_s = SEOdata[:, 1], SEOdata[:, 6], SEOdata[:, 7], 1/SEOdata[:, 4]
    Func_Ek_r = FFAG_interpolation().linear_interpolation(Ek0, r0)
    Func_Ek_pr = FFAG_interpolation().linear_interpolation(Ek0, pr0)
    Func_Ek_T = FFAG_interpolation().linear_interpolation(Ek0, T0_s)
    r_c, pr_c, T_s = Func_Ek_r(Ek_c), Func_Ek_pr(Ek_c), Func_Ek_T(Ek_c)

    return np.array([Ek_c, r_c, pr_c, T_s]), path, SEOdata


def LoadSEOBetaFunc(SEOPATHName, Ek_c, fi_c):

    BetaZFileName = os.path.join(SEOPATHName, "BetaFuncZ.txt")
    BetaRFileName = os.path.join(SEOPATHName, "BetaFuncR.txt")
    AlphaZFileName = os.path.join(SEOPATHName, "AlphaFuncZ.txt")
    AlphaRFileName = os.path.join(SEOPATHName, "AlphaFuncR.txt")

    BetaZdata = np.loadtxt(BetaZFileName, skiprows=1)
    Ek_axis1, fi_axis1, BetaZ_matrix = BetaZdata[1:, 0], BetaZdata[0, 1:], BetaZdata[1:, 1:]

    Ek_c_arr = np.array([Ek_c,])
    fi_c_arr = np.mod(np.array([fi_c,]), np.deg2rad(fi_axis1[-1]-fi_axis1[0]))

    BetaZ_inj = FFAG_interpolation().Lagrange_interp_2D_vect(Ek_axis1, fi_axis1, BetaZ_matrix, Ek_c_arr, fi_c_arr)

    BetaRdata = np.loadtxt(BetaRFileName, skiprows=1)
    Ek_axis2, fi_axis2, BetaR_matrix = BetaRdata[1:, 0], BetaRdata[0, 1:], BetaRdata[1:, 1:]
    BetaR_inj = FFAG_interpolation().Lagrange_interp_2D_vect(Ek_axis2, fi_axis2, BetaR_matrix, Ek_c_arr, fi_c_arr)

    AlphaZdata = np.loadtxt(AlphaZFileName, skiprows=1)
    Ek_axis3, fi_axis3, AlphaZ_matrix = AlphaZdata[1:, 0], AlphaZdata[0, 1:], AlphaZdata[1:, 1:]
    AlphaZ_inj = FFAG_interpolation().Lagrange_interp_2D_vect(Ek_axis3, fi_axis3, AlphaZ_matrix, Ek_c_arr, fi_c_arr)

    AlphaRdata = np.loadtxt(AlphaRFileName, skiprows=1)
    Ek_axis4, fi_axis4, AlphaR_matrix = AlphaRdata[1:, 0], AlphaRdata[0, 1:], AlphaRdata[1:, 1:]
    AlphaR_inj = FFAG_interpolation().Lagrange_interp_2D_vect(Ek_axis4, fi_axis4, AlphaR_matrix, Ek_c_arr, fi_c_arr)

    return BetaZ_inj[0][0], BetaR_inj[0][0], AlphaZ_inj[0][0], AlphaR_inj[0][0]


def generate_ellipse_points(epsilon_pi_m_rad_1rms, alpha, beta, rms_times, num_points=1000):
    # 计算 gamma
    gamma = (1 + alpha ** 2) / beta

    # 生成 theta 参数
    theta = np.linspace(0.000, 2 * np.pi, num_points)

    u = np.cos(theta)
    uprime = np.sin(theta)
    # 对于每个theta点，找到对应的R，使其满足beta*y^2+2*alpha*xy+gamma*x^2=epsilon
    FuncValue = beta * uprime ** 2 + 2 * alpha * u * uprime + gamma * u ** 2
    Radius = np.sqrt(rms_times * epsilon_pi_m_rad_1rms / FuncValue)
    x_ellipse_4rms = Radius * np.cos(theta)
    y_ellipse_4rms = Radius * np.sin(theta)

    return x_ellipse_4rms, y_ellipse_4rms

def generate_ellipse_points_from_emittance(epsilon, alpha, beta, num_points=1000):
    """
    给定任意发射度 epsilon、Twiss 参数 alpha beta，
    生成满足 βp^2 + 2αxp + γx^2 = ε 的椭圆点 (x, p)。

    参数
    ----
    epsilon : float
        几何发射度（m·rad）
    alpha, beta : float
        Twiss 参数
    num_points : int
        采样点数（默认1000）

    返回
    ----
    x_ell, p_ell : ndarray
        椭圆上的 x 与 p 坐标
    """

    gamma = (1 + alpha ** 2) / beta   # Twiss gamma

    theta = np.linspace(0, 2*np.pi, num_points)

    # 单位方向向量
    u = np.cos(theta)
    uprime = np.sin(theta)

    # 方向上的 Courant-Snyder 权重 F(θ)
    F = beta * uprime**2 + 2*alpha*u*uprime + gamma * u**2

    # 对应方向的半径 R(θ)
    R = np.sqrt(epsilon / F)

    # 椭圆点
    x_ell = R * u
    p_ell = R * uprime

    return x_ell, p_ell


def compute_rms_emit(x, xp):
    """
    输入:
        x  : 1D ndarray, 位置 (m)
        xp : 1D ndarray, 角度或共轭动量 (rad)
    返回:
        eps_rms : RMS 发射度 (m·rad)
        stats   : (sigma_x, sigma_xp, cov_x_xp)
    """
    x  = np.asarray(x)
    xp = np.asarray(xp)

    x_mean  = np.mean(x)
    xp_mean = np.mean(xp)

    x2_mean  = np.mean((x  - x_mean)**2)
    xp2_mean = np.mean((xp - xp_mean)**2)
    xxp_mean = np.mean((x - x_mean) * (xp - xp_mean))

    eps_rms = np.sqrt(x2_mean * xp2_mean - xxp_mean**2)

    return eps_rms, (np.sqrt(x2_mean), np.sqrt(xp2_mean), xxp_mean)


def GenerateBunches(BunchPara):
    """
    根据输入参数生成多个粒子分布，并将它们拼接成一个完整的bunch。
    """

    # 提取参数
    SEOFileName = BunchPara["SEO"]
    BmapPATHName = BunchPara["BmapPATHName"]
    InjectEk = BunchPara["InjectEk"]
    InjectionPosition = BunchPara["InjPosition"]
    ParticleNum = BunchPara["ParticleNum"]
    TransverseREmit = BunchPara["TransverseREmit"]
    TransverseZEmit = BunchPara["TransverseZEmit"]
    LongitudeT = BunchPara["LongitudeT"]
    LongitudeDEk = BunchPara["LongitudeDEk"]

    InjTimeNanoSec = BunchPara["InjTimeNanoSec"]
    TransverseDistType = BunchPara["TransverseDistType"]
    LongitudeDistType = BunchPara["LongitudeDistType"]
    PlotFlag = BunchPara["PlotFlag"]
    PlotRMS = BunchPara["PlotRMS"]

    PaintEnable = BunchPara["PaintEnable"]
    PaintMaxNum = BunchPara["PaintMaxNum"]
    PaintTimeInterval = BunchPara["PaintTimeInterval"]
    PaintCurvePath = BunchPara["PaintCurve"]

    EmapPATHName = BunchPara["EmapPATHName"]
    with open(EmapPATHName, "r") as file:
        content = file.read()  # 读取整个文件内容

    # 匹配 gap_azimuth 和 acc_phi0
    match_gap = re.search(r'gap_azimuth = ([\d.eE+-]+)', content)
    match_acc = re.search(r'acc_phi_paint = ([\d.eE+-]+)', content)
    match_V0 = re.search(r'acc_voltage_paint = ([\d.eE+-]+)', content)

    # 转换成 float（如果匹配到）
    gap_azimuth = float(match_gap.group(1)) if match_gap else None
    acc_phi0 = float(match_acc.group(1)) if match_acc else None
    acc_V0 = float(match_V0.group(1)) if match_V0 else None

    if PaintEnable:
        # MaxBunchNum = PaintMaxNum
        PaintCurve = np.loadtxt(PaintCurvePath, skiprows=1)
        MaxBunchNum = int(min(np.floor(np.max(PaintCurve[:, 0])/PaintTimeInterval), PaintMaxNum))
    else:
        MaxBunchNum = 1

    # 初始化数组以存储粒子束
    Bunches = []
    custdist_list = []
    Centers_list = []

    # 循环生成每个子束团的分布
    bucket_obj1 = None
    for i in range(MaxBunchNum):
        # 发射度和粒子数
        Emit = [TransverseREmit, TransverseZEmit, LongitudeT*LongitudeDEk/16]
        num_points = ParticleNum
        InjTime = i * PaintTimeInterval
        LongitudeAlpha_i = 0.0
        LongitudeBeta_i = LongitudeT/LongitudeDEk

        if PaintEnable:
            # 通过涂抹曲线获取偏移量
            Paint_x = np.interp(InjTime, PaintCurve[:, 0], PaintCurve[:, 1])  # 涂抹曲线中的 x
            Paint_px = np.interp(InjTime, PaintCurve[:, 0], PaintCurve[:, 2])  # 涂抹曲线中的 px
            Paint_y = np.interp(InjTime, PaintCurve[:, 0], PaintCurve[:, 3])  # 涂抹曲线中的 y
            Paint_py = np.interp(InjTime, PaintCurve[:, 0], PaintCurve[:, 4])  # 涂抹曲线中的 py
        else:
            Paint_x, Paint_px, Paint_y, Paint_py = 0, 0, 0, 0

        dx_xp_y_yp = (Paint_x, Paint_px, Paint_y, Paint_py)

        # 生成子束团 Ek in MeV
        sub_bunch, custdist, Centers, bucket_obj1 = GenerateInitDistributionDtInjNew(
            SEOFileName, InjectionPosition, InjectEk, Emit,
            num_points, InjTime, LongitudeAlpha_i, LongitudeBeta_i, LongitudeT,
            acc_U0 = acc_V0,
            acc_phi0 = acc_phi0,
            dist_types=(TransverseDistType, LongitudeDistType),
            dx_xp_y_yp = dx_xp_y_yp,
            Bunch_ID=i,
            bucket_obj=bucket_obj1,
        )

        # 将每个子束团加入到Bunches数组中
        Bunches.append(sub_bunch)
        custdist_list.append(custdist)
        Centers_list.append(Centers)
        # print(f"BinchID={i}")

    # 拼接所有子束团，形成一个完整的粒子束
    BunchesCombined = np.vstack(Bunches)
    # r, pr, z, pz, fi, Ek, inj_t, inj_flag, survive_flag, ... ...

    if PlotFlag:
        # 获取 MPI 通信对象和当前线程 rank
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank == 0:
            for i, sub_bunch in enumerate(Bunches):
                # 提取当前子束团的相空间数据
                x_vals = sub_bunch[:, 0]
                xp_vals = sub_bunch[:, 1]
                y_vals = sub_bunch[:, 2]
                yp_vals = sub_bunch[:, 3]
                fi_vals = sub_bunch[:, 4]
                Ek_vals = sub_bunch[:, 5]
                t_vals = sub_bunch[:, 6]

                # 获取 Twiss 参数和发射度
                custdist = custdist_list[i]
                Centers = Centers_list[i]
                TwissR_container = custdist.getTwissContainers()[0]
                TwissZ_container = custdist.getTwissContainers()[1]
                TwissF_container = custdist.getTwissContainers()[2][0]

                emit_times_rz = custdist.dist_transverse.emit_times
                emit_times_f = custdist.dist_longitudinal.emit_times

                transverse_alpha_r = TwissR_container.alpha
                transverse_beta_r = TwissR_container.beta
                epsilon_r_1 = TwissR_container.emittance / emit_times_rz

                transverse_alpha_z = TwissZ_container.alpha
                transverse_beta_z = TwissZ_container.beta
                epsilon_z_1 = TwissZ_container.emittance / emit_times_rz

                longitudinal_alpha_f = TwissF_container.alpha
                longitudinal_beta_f = TwissF_container.beta
                epsilon_f_1 = TwissF_container.emittance / emit_times_f

                r_c_plot, pr_c_plot, y_c_plot, py_c_plot, Ek_c_plot, t_c_plot = Centers[0], Centers[1], Centers[2], Centers[
                    3], Centers[4], Centers[5]

                # 绘制径向相空间
                plt.figure(1)
                plt.scatter(x_vals * 1000, xp_vals * 1000, s=0.5, label="Particles")
                ellipse_x, ellipse_xp = generate_ellipse_points(
                    epsilon_r_1, transverse_alpha_r, transverse_beta_r, PlotRMS[0], num_points=1000
                )
                plt.plot((ellipse_x + r_c_plot) * 1000, (ellipse_xp + pr_c_plot) * 1000, linewidth=2,
                         label=f'{PlotRMS[0]} RMS Ellipse')
                plt.xlabel("r (mm)")
                plt.ylabel("pr (mrad)")
                plt.title(f"Bunch {i} - Radial Phase Space")

                eps_r_rms, (sigma_r, sigma_pr, cov_r_pr) = \
                    compute_rms_emit(x_vals, xp_vals)

                print("=== rz plane RMS emittance ===")
                print(f"RMS ε_r   = {eps_r_rms*1e3*1e3:.6e}  (mm·mrad)")
                print(f"sigma_r   = {sigma_r*1e3:.6e} mm")
                print(f"sigma_pr  = {sigma_pr*1e3:.6e} mrad")


                # 绘制垂直相空间
                plt.figure(2)
                plt.scatter(y_vals * 1000, yp_vals * 1000, s=0.5, label="Particles")
                ellipse_y, ellipse_yp = generate_ellipse_points(
                    epsilon_z_1, transverse_alpha_z, transverse_beta_z, PlotRMS[1], num_points=1000
                )
                plt.plot((ellipse_y+y_c_plot) * 1000, (ellipse_yp+py_c_plot) * 1000, linewidth=2,
                         label=f'{PlotRMS[1]} RMS Ellipse')
                plt.xlabel("z (mm)")
                plt.ylabel("pz (mrad)")
                plt.title(f"Bunch {i} - Vertical Phase Space")

                eps_z_rms, (sigma_z, sigma_pz, cov_z_pz) = \
                    compute_rms_emit(y_vals, yp_vals)

                print("\n=== z direction RMS emittance ===")
                print(f"RMS ε_z   = {eps_z_rms*1e3*1e3:.6e}  (mm·mrad)")
                print(f"sigma_z   = {sigma_z*1e3:.6e} mm")
                print(f"sigma_pz  = {sigma_pz*1e3:.6e} mrad")

                # 绘制纵向相空间
                plt.figure(3)
                plt.scatter(t_vals, Ek_vals, s=0.5, label="Particles")
                ellipse_t, ellipse_Ek = generate_ellipse_points(
                    epsilon_f_1, longitudinal_alpha_f, longitudinal_beta_f, PlotRMS[2], num_points=1000
                )
                # 如果需要绘制椭圆可以取消注释
                plt.plot(ellipse_t + t_c_plot, ellipse_Ek + Ek_c_plot, color='r', linewidth=2,
                         label=f'{PlotRMS[2]} RMS Ellipse')
                plt.xlabel("t(ns)")
                plt.ylabel("Ek(MeV)")
                # plt.legend()
                plt.title(f"Bunch {i} - Longitudinal Phase Space")

            # 显示所有图
            plt.show()

            # 所有线程在这里等待，确保 rank 0 完成绘图后同步退出
        comm.Barrier()
        sys.exit()

    # 返回生成的粒子分布
    return BunchesCombined


if __name__ == "__main__":

    # plot_kv_1d()  # 绘制1D KV分布
    # plot_kv_2d()  # 绘制2D KV分布
    # plot_kv_3d()  # 绘制3D KV分布

    # plot_gauss_1d()  # 绘制1D Gaussian分布
    # plot_gauss_2d()  # 绘制2D Gaussian分布
    # plot_gauss_3d()  # 绘制3D Gaussian分布的三种投影

    # plot_waterbag_1d()  # 绘制1D WaterBag分布
    # plot_waterbag_2d()  # 绘制2D WaterBag分布
    # plot_waterbag_3d()  # 绘制3D WaterBag分布的三种投影

    # plot_hollow_waterbag_1d()  # 绘制1D空心水袋分布
    # plot_hollow_waterbag_2d()  # 绘制2D空心水袋分布

    PlotDistribution().plot_kv_2d_new()

    # SEOPATHName = "./resultsSEO/Bmap-2024-07-02-231113"
    # InjectionPositionPATHName = "./Input_InjBunch/cl2.cl"
    #
    # Partcle_dist, _, _ = GenerateInitDistributionDtInjNew(SEOPATHName, InjectionPositionPATHName, 350.0, (200, 200, 200),
    #                                  10000, dist_types=('hollow_waterbag', 'waterbag'))
    # array_dist = np.array(Partcle_dist)
    #
    # plt.figure()
    # plt.scatter(array_dist[:, 0], array_dist[:, 1], s=2, c='b')
    # plt.figure()
    # plt.scatter(array_dist[:, 2], array_dist[:, 3], s=2, c='b')
    # plt.figure()
    # plt.scatter(array_dist[:, 0], array_dist[:, 2], s=2, c='b')
    # plt.figure()
    # plt.scatter(array_dist[:, 4], array_dist[:, 5], s=2, c='b')
    # plt.show()
    # pass