from io import UnsupportedOperation
import numpy as np
from regex import R
from geometry import geometry
import healpy as hp
from healpy import Alm
import numpy.linalg as LA
from scipy.special import lpmn, sph_harm
import matplotlib.pyplot as plt

import numpy as np
import types

def TQfundamental_noise_spectrum(freqs, Np=1e-24, Na=1e-30):
    # 定义天琴的位置噪声和加速度噪声
    L = np.sqrt(3)*1e8
    Sp = freqs*Np/(freqs*4*L**2)
    Sa = (1+1e-4/freqs)*Na/((2*np.pi*freqs)**4*4*L**2)
    return Sp ,Sa

def fundamental_noise_spectrum(freqs, Np=4e-41, Na=1.44e-48):
    Sp = Np*(1 + (2e-3/freqs)**4)
    Sa = Na*(1 + 16e-8/freqs**2)*(1 + (freqs/8e-3)**4)*(1.0/(2*np.pi*freqs)**4)

    return Sp,Sa

def LSaet_noise_spectrum(freqs,f0, Np=4e-41, Na=1.44e-48):
    # Get Sp and Sa
    C_xyz = LSxyz_noise_spectrum(freqs, f0, Np, Na)

    ## Upnack xyz matrix to make assembling the aet matrix easier
    CXX, CYY, CZZ = C_xyz[0, 0], C_xyz[1, 1], C_xyz[2, 2]
    CXY, CXZ, CYZ = C_xyz[0, 1], C_xyz[0, 2], C_xyz[1, 2]


    ## construct AET matrix elements
    CAA = (1/9) * (4*CXX + CYY + CZZ - 2*CXY - 2*np.conj(CXY) - 2*CXZ - 2*np.conj(CXZ) + \
                        CYZ  + np.conj(CYZ))

    CEE = (1/3) * (CZZ + CYY - CYZ - np.conj(CYZ))

    CTT = (1/9) * (CXX + CYY + CZZ + CXY + np.conj(CXY) + CXZ + np.conj(CXZ) + CYZ + np.conj(CYZ))

    CAE = (1/(3*np.sqrt(3))) * (CYY - CZZ - CYZ + np.conj(CYZ) + 2*CXZ - 2*CXY)

    CAT = (1/9) * (2*CXX - CYY - CZZ + 2*CXY - np.conj(CXY) + 2*CXZ - np.conj(CXZ) - CYZ - np.conj(CYZ))

    CET = (1/(3*np.sqrt(3))) * (CZZ - CYY - CYZ + np.conj(CYZ) + np.conj(CXZ) - np.conj(CXY))

    C_aet = np.array([ [CAA, CAE, CAT] , \
                                    [np.conj(CAE), CEE, CET], \
                                    [np.conj(CAT), np.conj(CET), CTT] ])

    return C_aet
def LSxyz_noise_spectrum(freqs,f0, Np=4e-41, Na=1.44e-48):
    C_mich = LSmich_noise_spectrum(freqs, f0, Np, Na)

    ## Noise spectra of the X, Y and Z channels
    #SX = 4*SM1* np.sin(2*f0)**2

    C_xyz =  4 * np.sin(2*f0)**2 * C_mich

    return C_xyz

def LSmich_noise_spectrum(freqs,f0, Np=4e-41, Na=1.44e-48):
    # Get Sp and Sa
    Sp, Sa = fundamental_noise_spectrum(freqs, Np, Na)
    ## Noise spectra of the michelson channels
    S_auto  = 4.0 * (2.0 * Sa * (1.0 + (np.cos(2*f0))**2)  + Sp)
    S_cross =  (-2 * Sp - 8 * Sa) * np.cos(2*f0)

    C_mich = np.array([[S_auto, S_cross, S_cross], [S_cross, S_auto, S_cross], [S_cross, S_cross, S_auto]])

    return C_mich
def TQmich_noise_spectrum(freqs, f0, Np=1e-24, Na=1e-30):

    # freqs:一个频率数组
    # f0: f/fstar
    # Np 位置噪声
    # Na 加速度噪声

    # 获取位置噪声和加速度噪声Sp Sa
    Sp, Sa = TQfundamental_noise_spectrum(freqs,Np,Na)
    # 天琴臂长：sqrt(3)*1e8  单位：m
    #L = np.sqrt(3)*1e8
    # 自相关
    S_auto = 4 * (2.0 * Sa * (1.0 + (np.cos(2 * f0)) ** 2) + Sp)
    # 互相关
    S_cross = (-2 * Sp - 8 * Sa) * np.cos(2 * f0)
    C_mich = np.array([[S_auto, S_cross, S_cross], [S_cross, S_auto, S_cross], [S_cross, S_cross, S_auto]])

    return C_mich
def TQxyz_noise_spectrum(freqs,f0, Np=1e-24, Na=1e-30):
    C_mich = TQmich_noise_spectrum(freqs, f0, Np, Na)

    ## Noise spectra of the X, Y and Z channels
    #SX = 4*SM1* np.sin(2*f0)**2

    C_xyz =  4 * np.sin(2*f0)**2 * C_mich

    return C_xyz
def TQaet_noise_spectrum(freqs,f0, Np=1e-24, Na=1e-30):
    # Get Sp and Sa
    C_xyz = TQxyz_noise_spectrum(freqs, f0, Np, Na)

    ## Upnack xyz matrix to make assembling the aet matrix easier
    CXX, CYY, CZZ = C_xyz[0, 0], C_xyz[1, 1], C_xyz[2, 2]
    CXY, CXZ, CYZ = C_xyz[0, 1], C_xyz[0, 2], C_xyz[1, 2]


    ## construct AET matrix elements
    CAA = (1/9) * (4*CXX + CYY + CZZ - 2*CXY - 2*np.conj(CXY) - 2*CXZ - 2*np.conj(CXZ) + \
                        CYZ  + np.conj(CYZ))

    CEE = (1/3) * (CZZ + CYY - CYZ - np.conj(CYZ))

    CTT = (1/9) * (CXX + CYY + CZZ + CXY + np.conj(CXY) + CXZ + np.conj(CXZ) + CYZ + np.conj(CYZ))

    CAE = (1/(3*np.sqrt(3))) * (CYY - CZZ - CYZ + np.conj(CYZ) + 2*CXZ - 2*CXY)

    CAT = (1/9) * (2*CXX - CYY - CZZ + 2*CXY - np.conj(CXY) + 2*CXZ - np.conj(CXZ) - CYZ - np.conj(CYZ))

    CET = (1/(3*np.sqrt(3))) * (CZZ - CYY - CYZ + np.conj(CYZ) + np.conj(CXZ) - np.conj(CXY))

    C_aet = np.array([ [CAA, CAE, CAT] , \
                                    [np.conj(CAE), CEE, CET], \
                                    [np.conj(CAT), np.conj(CET), CTT] ])


    return C_aet

