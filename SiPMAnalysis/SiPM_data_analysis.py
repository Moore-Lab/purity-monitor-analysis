import numpy as np
import matplotlib.pyplot as plt
import natsort, glob, h5py
from scipy.optimize import curve_fit
import scipy.signal

class SiPM_data(self):
    
    
    def gain_fit(self):
        # fitting a line of the peak centers and heights to get the slope (gain) 
        fpts = ~np.isnan(pe_locations[:,1])
#     if(np.sum(fpts) < 3):
#         return
        # fitting the peak locations to a line to find the gain (slope)
        pe_locations = self.peak_data
        gain_bp, gain_bc = curve_fit(lin_fun, pe_locations[fpts, 0], pe_locations[fpts, 1])

        x = pe_locations[fpts, 0]

        return x, gain_bp, gain_bc
    
    def bv_calc(self):
        # finding the breakdown voltage using fitted gain data of a certain wavelength (310, 405, or source)
        data_to_use = np.asarray(self.gain_data)

        # fitting the curve 
        bv_params, bv_cov = curve_fit(lin_fun, data_to_use[:, 0], data_to_use[:, 1], sigma=data_to_use[:, 2])

        return bv_params, bv_cov

class SRS_10_source_data(SiPM_data):
    
class SRS_10_405_data(SiPM_data):
    
class SRS_10_310_data(SiPM_data):
    
class SRS_100_source_data(SiPM_data):
    
class SRS_100_405_data(SiPM_data):
    
class SRS_100_405_data(SiPM_data):
    
class fit_by_hand(SiPM_data):
    
# --------------------- global functions --------------------- #

def gauss_fun(x, A, mu, sig):
    return A * np.exp( -(x-mu)**2/(2*sig**2) )

def lin_fun(x, m, x1):
    return m*(x-x1)