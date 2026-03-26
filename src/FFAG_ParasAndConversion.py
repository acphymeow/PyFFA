import numpy as np


class FFAG_GlobalParameters:
    def __init__(self):
        self.Bmap = None
        self.Emap = None
        self.Bunch = None
        self.SEOInfo = None
        self.SEO_SaveFold = 'resultsSEO'
        self.units = {}
        self.Attribute = {}
        self.q = 1.60217662e-19
        self.c = 2.99792458e8  # m/s
        self.E0 = 938.2723e6 * self.q  # J
        self.h = 2
        self.f0 = 45.75e6  # Hz
        self.current_phase = 0.0  # V
        self.m0 = self.E0 / self.c ** 2
        self.DefineUnits()
        self.DefineBunchAttributes()
        self.TempVariable = None
        self.dumpInfo = None
        self.ReverseTrack = False
        self.DumpInfo = None
        self.Aperture_m = None
        self.Aperture_enable = False
        self.BHarmonics = None


    def __repr__(self):
        return "E0=%.fMeV, f0=%.2fMHZ, h=%d" % (
            self.E0 / self.q / 1e6, self.f0 / 1e6, self.h)

    def DefineUnits(self):
        self.units = {'m': 1, 'rad': 1, 'T': 1, 's': 1, 'Hz': 1, 'V': 1,
                      'cm': 100, 'mm': 1000, 'deg': 180.0 / np.pi, 'Gs': 10000,
                      'ms': 1e3, 'us': 1e6, 'ns': 1e9, 'kHz': 1e-3,
                      'MHz': 1e-6, 'kV': 1e-3, 'MV': 1e-6}

    def DefineBunchAttributes(self):
        self.Attribute = {'r': 0, 'vr': 1, 'z': 2, 'vz': 3, 'fi': 4, 'Ek': 5, 'inj_t': 6,
                          'ini_flag': 7, 'RF_phase': 8, 'Esc_r': 9, 'Esc_z': 10, 'Esc_fi': 11,
                          'Bunch_ID': 12, 'Local_ID': 13, 'Global_ID': 14}

    def get_attributes_num(self):
        """
        获取当前属性的个数，即Bunch矩阵的列数。
        """
        return len(self.Attribute)

    def get_attribute_names(self):
        """
        获取所有属性名称的列表。
        """
        return list(self.Attribute.keys())

    def AddBMap(self, BFieldMapData):
        self.Bmap = BFieldMapData

    def AddBHarmonics(self, BHarmonicsList):
        self.BHarmonics = BHarmonicsList

    def AddEMap(self, EFieldMapData):
        self.Emap = EFieldMapData

    def AddBunch(self, Bunch):
        self.Bunch = Bunch

    def AddSEOInfo(self, SEOdata):
        self.SEOInfo = SEOdata

    def AddDumpInfo(self, Dumpdata):
        self.DumpInfo = Dumpdata

    def SetReverseTrack(self, ReverseTrackFlag):
        self.ReverseTrack = ReverseTrackFlag

    def AddAperture(self, rmin, rmax, zmin, zmax):
        self.Aperture_m = np.array((rmin, rmax, zmin, zmax))
        self.Aperture_enable = True


class FFAG_ConversionTools:
    def __init__(self):
        self.CommonVariables = FFAG_GlobalParameters()

    def Ek2P(self, Ek_value):
        # convert energy of protons to momentum
        Ek_J = Ek_value * 1e6 * self.CommonVariables.q
        E0_J = self.CommonVariables.E0
        P = 1 / self.CommonVariables.c * np.sqrt((Ek_J + E0_J) ** 2 - E0_J ** 2)
        return P

    def Ek2v(self, Ek_value):
        Ek_J = Ek_value * 1e6 * self.CommonVariables.q
        E0_J = self.CommonVariables.E0
        gamma = (Ek_J + E0_J) / E0_J
        beta = np.sqrt(1 - (1 / gamma) ** 2)
        v = beta * self.CommonVariables.c
        return v

    def EkMeV2EtotJ(self, Ek_value):
        Ek_J = Ek_value * 1e6 * self.CommonVariables.q
        E0_J = self.CommonVariables.E0
        return Ek_J + E0_J

    def EtotJ2EkMeV(self, Etot_J):
        E0_J = self.CommonVariables.E0
        Ek_J = Etot_J - E0_J
        Ek_MeV = Ek_J / 1e6 / self.CommonVariables.q
        return Ek_MeV

    def EkMeV2EkJ(self, Ek_value):
        Ek_J = Ek_value * 1e6 * self.CommonVariables.q
        return Ek_J

    def EkJ2EkMeV(self, Ek_J):
        Ek_MeV = Ek_J / 1e6 / self.CommonVariables.q
        return Ek_MeV

    def v2P(self, v):
        c = self.CommonVariables.c
        m0 = self.CommonVariables.m0
        beta = v / c
        gamma = 1 / np.sqrt(1 - beta ** 2)
        P = gamma * beta * m0 * c
        return P

    def v2Ek(self, v):
        c = self.CommonVariables.c
        E0_J = self.CommonVariables.E0
        beta = v / c
        gamma = 1 / np.sqrt(1 - beta ** 2)
        Ek_J = (gamma - 1) * E0_J
        Ek_MeV = Ek_J / 1e6 / self.CommonVariables.q
        return Ek_MeV

    def v2Ek_J(self, v):
        c = self.CommonVariables.c
        q = self.CommonVariables.q
        E0_J = self.CommonVariables.E0
        beta = v / c
        gamma = 1 / np.sqrt(1 - beta ** 2)
        Ek_J = (gamma - 1) * E0_J
        Etotal_J = E0_J + Ek_J
        return Ek_J, Etotal_J

    def P2Ek(self, P):
        E0_J = self.CommonVariables.E0
        c = self.CommonVariables.c
        Ek_J = np.sqrt((P * c) ** 2 + E0_J ** 2) - E0_J
        return Ek_J

    def P2m(self, P):
        E0_J = self.CommonVariables.E0
        c = self.CommonVariables.c
        E_J = np.sqrt((P * c) ** 2 + E0_J ** 2)
        m = E_J / (c ** 2)
        return m

    def Ek2tp(self, Ek, r, rp, zp):
        Ek_J = Ek * 1e6 * self.CommonVariables.q
        E0_J = self.CommonVariables.E0
        gamma = (Ek_J + E0_J) / E0_J
        beta = np.sqrt(1 - (1 / gamma) ** 2)
        v = beta * self.CommonVariables.c
        tp = np.sqrt(r ** 2 + rp ** 2 + zp ** 2) / v
        return tp

    def tp2Ek_MeV(self, r, rp, zp, tp):
        v = np.sqrt(r ** 2 + rp ** 2 + zp ** 2) / tp
        Ek = self.v2Ek(v)  # MeV
        return Ek

    def tp2Ek(self, Result):
        if np.size(Result, 1) == 7:
            r, rp, z, zp, t, tp, index = Result[:, 0], Result[:, 1], Result[:, 2], \
                Result[:, 3], Result[:, 4], Result[:, 5], Result[:, 6]
            v = np.sqrt(r ** 2 + rp ** 2 + zp ** 2) / tp
            Ek = self.v2Ek(v)
        else:
            r, rp, z, zp, P = Result[:, 0], Result[:, 1], Result[:, 2], \
                Result[:, 3], Result[:, 4]
            Ek = self.P2Ek(P) / 1e6 / self.CommonVariables.q
        return Ek

    def xy2rfi_m2p180(self, x, y, SmoothAzimuth=False):
        # convert (x,y) to (r,fi), fi is from -180deg to 180deg(m2p180,minus to positive 180)
        # test code: test_xy2rfi.py
        r = np.hypot(x, y)
        theta_rad = np.arctan2(y, x)
        if SmoothAzimuth:
            for index in range(len(x) - 1):
                if abs(theta_rad[index + 1] - theta_rad[index]) > np.deg2rad(300.0):
                    flag = -1 * abs(theta_rad[index + 1] - theta_rad[index]) / (theta_rad[index + 1] - theta_rad[index])
                    theta_rad[index + 1:] = theta_rad[index + 1:] + np.pi * 2 * flag

        theta_deg = np.rad2deg(theta_rad)
        return theta_rad, r, theta_deg

    def ConvertPrzek2Vrzek_boris(self, all_points):
        # Convert (r, pr), (z, pz), fi, (ek, t_inj), (rf_phase, Local_ID, Global_ID)
        # to (r, vr), (z, vz), (fi, dfidt), (t_inj), (rf_phase, Local_ID, Global_ID)
        # sequence of the columes: (r, vr), (z, vz), (fi, dfidt), (t_inj, inj_flag),
        # (rf_phase, Esc_r, Esc_z, Esc_fi), (Bunch_ID, Local_ID, Global_ID)

        r_orignal, pr_orignal = all_points[:, 0], all_points[:, 1]  # tan(pr) = vr/vf
        z_orignal, pz_orignal = all_points[:, 2], all_points[:, 3]  # tan(pz) = vz/vf
        fi_orignal, Ek_orignal = all_points[:, 4], all_points[:, 5]  # MeV
        t_inj, Injflag, survive_flag, RF_Phase = all_points[:, 6], all_points[:, 7], all_points[:, 8], all_points[:, 9]
        Esc_r, Esc_z, Esc_fi = all_points[:, 10], all_points[:, 11], all_points[:, 12]
        BunchIndex_original = all_points[:, 13]
        LocalIndex_original = all_points[:, 14]
        GlobalIndex_original = all_points[:, 15]

        v_c = self.Ek2v(Ek_orignal)
        vf_relative = 1.0
        vr_relative = np.tan(pr_orignal) * vf_relative
        vz_relative = np.tan(pz_orignal) * vf_relative
        vtotal_relative = np.sqrt(vz_relative ** 2 + vr_relative ** 2 + vf_relative ** 2)

        vr_direction = vr_relative / vtotal_relative
        vf_direction = vf_relative / vtotal_relative
        vz_direction = vz_relative / vtotal_relative

        vr, vz, vf = v_c * vr_direction, v_c * vz_direction, v_c * vf_direction
        dfdt = vf / r_orignal
        Etot_J = self.EkMeV2EtotJ(Ek_orignal)  # convert Ek in MeV to Etot in J

        all_points_v = np.column_stack((r_orignal, vr, z_orignal, vz, fi_orignal,
                                        dfdt, t_inj, Injflag, survive_flag, RF_Phase, Esc_r, Esc_z, Esc_fi,
                                        BunchIndex_original, LocalIndex_original, GlobalIndex_original))
        return all_points_v


    def ConvertPrzek2Vrzek(self, all_points):
        # Convert (r, pr), (z, pz), fi, (ek, t_inj), (rf_phase, Local_ID, Global_ID)
        # to (r, vr), (z, vz), (fi, dfidt), (t_inj), (rf_phase, Local_ID, Global_ID)
        # sequence of the columes: (r, vr), (z, vz), (fi, dfidt), (t_inj, inj_flag),
        # (rf_phase, Esc_r, Esc_z, Esc_fi), (Bunch_ID, Local_ID, Global_ID)

        r_orignal, pr_orignal = all_points[:, 0], all_points[:, 1]  # tan(pr) = vr/vf
        z_orignal, pz_orignal = all_points[:, 2], all_points[:, 3]  # tan(pz) = vz/vf
        fi_orignal, Ek_orignal = all_points[:, 4], all_points[:, 5]  # MeV
        t_inj, Inj_flag, survive_flag, RF_Phase = all_points[:, 6], all_points[:, 7], all_points[:, 8], all_points[:, 9]
        Esc_r, Esc_z, Esc_fi = all_points[:, 10], all_points[:, 11], all_points[:, 12]
        BunchIndex_original = all_points[:, 13]
        LocalIndex_original = all_points[:, 14]
        GlobalIndex_original = all_points[:, 15]

        v_c = self.Ek2v(Ek_orignal)
        vf_relative = 1.0
        vr_relative = np.tan(pr_orignal) * vf_relative
        vz_relative = np.tan(pz_orignal) * vf_relative
        vtotal_relative = np.sqrt(vz_relative ** 2 + vr_relative ** 2 + vf_relative ** 2)

        vr_direction = vr_relative / vtotal_relative
        vf_direction = vf_relative / vtotal_relative
        vz_direction = vz_relative / vtotal_relative

        vr, vz, vf = v_c * vr_direction, v_c * vz_direction, v_c * vf_direction
        # dfdt = vf / r_orignal
        Etot_J = self.EkMeV2EtotJ(Ek_orignal)  # convert Ek in MeV to Etot in J

        all_points_v = np.column_stack((r_orignal, vr, z_orignal, vz, fi_orignal,
                                        Etot_J, t_inj, Inj_flag, survive_flag, RF_Phase, Esc_r, Esc_z, Esc_fi,
                                        BunchIndex_original, LocalIndex_original, GlobalIndex_original))
        return all_points_v

    # @profile
    def ConvertVrzek2Przek_boris(self, all_points):
        # Convert (r, vr), (z, vz), (fi, dfidt), (t_inj), (rf_phase, Local_ID, Global_ID)
        # to (r, pr), (z, pz), fi, (ek, t_inj), (rf_phase, Local_ID, Global_ID)
        # sequence of the columes: (r, vr), (z, vz), (fi, dfidt), (t_inj, inj_flag),
        # (rf_phase, Esc_r, Esc_z, Esc_fi), (Bunch_ID, Local_ID, Global_ID)

        # Convert r, vr, z, vz, fi, dfidt to r, pr, z, pz, fi, ek
        r_original, vr_original = all_points[:, 0], all_points[:, 1]  # tan(pr) = vr/vf
        z_original, vz_original = all_points[:, 2], all_points[:, 3]  # tan(pz) = vz/vf
        fi_original, vf_original = all_points[:, 4], all_points[:, 5]*r_original
        t_inj, Inj_flag, survive_flag, RF_Phase = all_points[:, 6], all_points[:, 7], all_points[:, 8], all_points[:, 9]
        Esc_r, Esc_z, Esc_fi = all_points[:, 10], all_points[:, 11], all_points[:, 12]
        BunchIndex_original = all_points[:, 13]
        LocalIndex_original = all_points[:, 14]
        GlobalIndex_original = all_points[:, 15]

        v_c = np.sqrt(vr_original ** 2 + vz_original ** 2 + vf_original ** 2)
        EkMeV = self.v2Ek(v_c)

        vr_direction = vr_original / v_c
        vz_direction = vz_original / v_c
        vf_direction = np.sqrt(1 - vr_direction ** 2 - vz_direction ** 2)

        pr = np.arctan2(vr_direction, vf_direction)
        pz = np.arctan2(vz_direction, vf_direction)

        all_points_p = np.column_stack((r_original, pr, z_original, pz, fi_original,
                                        EkMeV, t_inj, Inj_flag, survive_flag, RF_Phase, Esc_r, Esc_z, Esc_fi,
                                        BunchIndex_original, LocalIndex_original, GlobalIndex_original))
        return all_points_p


    def ConvertVrzek2Przek(self, all_points):
        # Convert (r, vr), (z, vz), (fi, dfidt), (t_inj), (rf_phase, Local_ID, Global_ID)
        # to (r, pr), (z, pz), fi, (ek, t_inj), (rf_phase, Local_ID, Global_ID)
        # sequence of the columes: (r, vr), (z, vz), (fi, dfidt), (t_inj, inj_flag),
        # (rf_phase, Esc_r, Esc_z, Esc_fi), (Bunch_ID, Local_ID, Global_ID)

        # Convert r, vr, z, vz, fi, dfidt to r, pr, z, pz, fi, ek
        r_original, vr_original = all_points[:, 0], all_points[:, 1]  # tan(pr) = vr/vf
        z_original, vz_original = all_points[:, 2], all_points[:, 3]  # tan(pz) = vz/vf
        fi_original, Etot_J = all_points[:, 4], all_points[:, 5]
        t_inj, Inj_flag, survive_flag, RF_Phase = all_points[:, 6], all_points[:, 7], all_points[:, 8], all_points[:, 9]
        Esc_r, Esc_z, Esc_fi = all_points[:, 10], all_points[:, 11], all_points[:, 12]
        BunchIndex_original = all_points[:, 13]
        LocalIndex_original = all_points[:, 14]
        GlobalIndex_original = all_points[:, 15]

        EkMeV = self.EtotJ2EkMeV(Etot_J)
        v_c = self.Ek2v(EkMeV)

        vr_direction = vr_original / v_c
        vz_direction = vz_original / v_c
        vf_direction = np.sqrt(1 - vr_direction ** 2 - vz_direction ** 2)

        pr = np.arctan2(vr_direction, vf_direction)
        pz = np.arctan2(vz_direction, vf_direction)

        all_points_p = np.column_stack((r_original, pr, z_original, pz, fi_original,
                                        EkMeV, t_inj, Inj_flag, survive_flag, RF_Phase, Esc_r, Esc_z, Esc_fi,
                                        BunchIndex_original, LocalIndex_original, GlobalIndex_original))
        return all_points_p


