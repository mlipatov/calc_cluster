# Load NGC1846 data and filter it
import sys
sys.path.append('..')
import config as cf

import numpy as np
from matplotlib import pyplot as plt
from scipy import special
from scipy import optimize

# load data on stars with vsini
file = 'data/ngc1846_vsini.txt'
data0 = np.loadtxt(file).T
data0 = data0[0:12] # select fields
# record the range of these stars in RA and dec; this should be close to the MUSE field of view (FOV)
ramin = data0[-2].min()
ramax = data0[-2].max()
demin = data0[-1].min()
demax = data0[-1].max()
# delete the entries without valid IDs, observables or sky coordinates, 
# as well as those outside the color-magnitude window
mag = data0[2]
col = data0[1] - data0[3] 
m = np.all(data0 != 99, axis=0) & \
	(mag <= cf.ROI[0][1]) & (mag >= cf.ROI[0][0]) & (col <= cf.ROI[1][1]) & (col >= cf.ROI[1][0])
data0 = data0[:, m]
# record the observable fields
ID = data0[0].astype(int)
f435w = data0[1]
f555w = data0[2]
f814w = data0[3]
f435w_err = data0[4]
f555w_err = data0[5]
f814w_err = data0[6]
vsini = data0[7]
vsini_uperr = data0[8]
vsini_loerr = data0[9]
vsini_err = (vsini_loerr + vsini_uperr) / 2

# load data on low-vsini stars
file = 'data/ngc1846_lowvsini.txt'
data1 = np.loadtxt(file).T
data1 = data1[0:12] # select fields
# delete the entries without valid IDs, observables or sky coordinates, 
# those outside the color-magnitude window or outside the MUSE FOV,
# as well as those that had data with normal vsini 
mag = data1[2]
col = data1[1] - data1[3] 
ra = data1[-2]
dec = data1[-1]
ID1 = data1[0].astype(int)
m = np.all(data1 != 99, axis=0) & \
	(mag <= cf.ROI[0][1]) & (mag >= cf.ROI[0][0]) & (col <= cf.ROI[1][1]) & (col >= cf.ROI[1][0]) & \
	(ra <= ramax) & (ra >= ramin) & (dec <= demax) & (dec >= demin) & \
	~np.isin(ID1, ID)
data1 = data1[:, m]
# merge with previous data
ID_vsini = np.concatenate( (ID, ID1) )
f435w = np.concatenate( (f435w, data1[1]) )
f555w = np.concatenate( (f555w, data1[2]) )
f814w = np.concatenate( (f814w, data1[3]) )
f435w_err = np.concatenate( (f435w_err, data1[4]) )
f555w_err = np.concatenate( (f555w_err, data1[5]) )
f814w_err = np.concatenate( (f814w_err, data1[6]) )
vsini = np.concatenate( (vsini, np.zeros_like(data1[7], dtype=float)) )
vsini_err = np.concatenate( (vsini_err, np.full_like(data1[8], cf.berrs[-1][0] * cf.std[-1])) )

# load data on stars without vsini in the MUSE FOV
file = 'data/ngc1846_full.txt'
data2 = np.loadtxt(file).T
data2 = data2[0:9] # select fields
# delete the entries without valid IDs, observables or sky coordinates, 
# those outside the color-magnitude window or outside the MUSE FOV,
# as well as those that had data with vsini or low vsini
mag = data2[2]
col = data2[1] - data2[3]
ra = data2[-2]
dec = data2[-1]
ID2 = data2[0].astype(int)
m = np.all(data2 != 99, axis=0) & \
	(mag <= cf.ROI[0][1]) & (mag >= cf.ROI[0][0]) & (col <= cf.ROI[1][1]) & (col >= cf.ROI[1][0]) & \
	(ra <= ramax) & (ra >= ramin) & (dec <= demax) & (dec >= demin) & \
	~np.isin(ID2, ID)
data2 = data2[:, m]

# merge all the data; set vsini-related fields appropriately
vsini_nan = np.full(data2.shape[1], np.nan)
ID = np.concatenate( (ID, ID2 ) )
f435w = np.concatenate( (f435w, data2[1]) )
f555w = np.concatenate( (f555w, data2[2]) )
f814w = np.concatenate( (f814w, data2[3]) )
f435w_err = np.concatenate( (f435w_err, data2[4]) )
f555w_err = np.concatenate( (f555w_err, data2[5]) )
f814w_err = np.concatenate( (f814w_err, data2[6]) )
vsini = np.concatenate( (vsini, np.copy(vsini_nan)) )
vsini_err = np.concatenate( (vsini_err, np.copy(vsini_nan)) )
# get color
color = f435w - f814w
color_err = np.sqrt(f435w_err**2 + f814w_err**2)

## boundaries of the color, magnitude and vsini space where we will need model priors
obs = np.stack( (f555w, color, vsini), axis=1 )
err = np.stack( (f555w_err, color_err, vsini_err), axis=1 )
err = np.maximum(err, cf.std[np.newaxis,:]) # minimum bound on error
resvar = (err/cf.std)**2 - 1 # residual relative variance
resvar[np.less(resvar, 0, where=~np.isnan(resvar))] = 0 # correct for round-off error
margin = cf.nsig * cf.std * (1 + np.sqrt(resvar))
# minimum boundaries needed for individual stars
obmax = np.nanmax(obs + margin, axis=0)
obmin = np.nanmin(obs - margin, axis=0)
# minimum boundaries needed for the initial convolution and probability leakage checks
obmax_conv = cf.ROI[:, 1] + cf.conv_err * cf.nsig * cf.std
obmin_conv = cf.ROI[:, 0] - cf.conv_err * cf.nsig * cf.std
# boundaries that determine which models are computed;
# these are the minimum needed for individual stars, initial convolution and probability checks
obmax = np.maximum(obmax, obmax_conv)
obmin = np.minimum(obmin, obmin_conv)
# plotting boundaries - these are inside the area where minimum-error density is defined
obmax_plot = cf.ROI[:, 1] + cf.plot_err * cf.nsig * cf.std
obmin_plot = cf.ROI[:, 0] - cf.plot_err * cf.nsig * cf.std



# # histogram to obtain approximate standard deviation for the stars that are in the vsini = 0 bin
# def f(x, A, sigma):
# 	return (A / 2) * (1 + special.erf(x / (sigma * np.sqrt(2))))
# v = vsini[~np.isnan(vsini)]
# bins = np.linspace(10, 150, 15)
# x = (bins[:-1] + bins[1:])/2
# y = np.histogram(v, bins=bins)[0]
# result = optimize.curve_fit(f, x, y, p0=(50, 10), sigma=np.sqrt(y))
# xplot = np.linspace(-x.max(), x.max(), 100)
# plt.scatter(x, y)
# plt.plot(xplot, f(xplot, result[0][0], result[0][1]))
# plt.ylabel('Number of stars')
# plt.xlabel('v * sin(i)')
# plt.show()
# print('sigma = ' + str(result[0][1]))

# max_margin = np.nanmax(margin, axis=0)
# obmin = np.array(cf.ROI)[:,0] - max_margin
# obmax = np.array(cf.ROI)[:,1] + max_margin
# # record the lower limit in vsini dimension at this point, even if it's negative
# # we won't calculate the prior at negative vsini, but we'll temporarily place 
# # convolved probability distribution at such values
# vsini_low = obmin[-1] 
# # remove negative vsini from the set at which we need to get the prior
# obmin[-1] = np.maximum(obmin[-1], 0) 
