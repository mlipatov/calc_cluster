# Load NGC1846 data and filter it

import config as cf

import numpy as np
from matplotlib import pyplot as plt

# load data on stars with vsini
file = 'data/ngc1846_vsini.txt'
data = np.loadtxt(file).T
data = data[0:12] # select fields
# record the range of these stars in RA and dec; this should be close to the MUSE field of view (FOV)
ramin = data[-2].min()
ramax = data[-2].max()
demin = data[-1].min()
demax = data[-1].max()
# delete the entries without valid IDs, observables or sky coordinates, 
# as well as those outside the color-magnitude window
mag = data[2]
col = data[1] - data[3] 
m = np.all(data != 99, axis=0) & \
	(mag <= cf.ROI[0][1]) & (mag >= cf.ROI[0][0]) & (col <= cf.ROI[1][1]) & (col >= cf.ROI[1][0])
data = data[:, m]
# record the observable fields
ID_vsini = data[0].astype(int)
f435w = data[1]
f555w = data[2]
f814w = data[3]
f435w_err = data[4]
f555w_err = data[5]
f814w_err = data[6]
vsini = data[7]
vsini_uperr = data[8]
vsini_loerr = data[9]
vsini_err = (vsini_loerr + vsini_uperr) / 2

# load data on stars without vsini in the MUSE FOV
file = 'data/ngc1846_full.txt'
data = np.loadtxt(file).T
data = data[0:9] # select fields
# get RA and dec
ra = data[-2]
dec = data[-1]
# delete the entries without valid IDs, observables or sky coordinates, 
# those outside the color-magnitude window or outside the MUSE FOV,
# as well as those that had data with vsini 
mag = data[2]
col = data[1] - data[3]
ID = data[0].astype(int)
m = np.all(data != 99, axis=0) & \
	(mag <= cf.ROI[0][1]) & (mag >= cf.ROI[0][0]) & (col <= cf.ROI[1][1]) & (col >= cf.ROI[1][0]) & \
	(ra <= ramax) & (ra >= ramin) & (dec <= demax) & (dec >= demin) & \
	~np.isin(ID, ID_vsini)
data = data[:, m]
# merge with the vsini data; set vsini-related fields to NAN
vsini_nan = np.full(data.shape[1], np.nan)
ID = np.concatenate( (ID_vsini, data[0].astype(int) ) )
f435w = np.concatenate( (f435w, data[1]) )
f555w = np.concatenate( (f555w, data[2]) )
f814w = np.concatenate( (f814w, data[3]) )
f435w_err = np.concatenate( (f435w_err, data[4]) )
f555w_err = np.concatenate( (f555w_err, data[5]) )
f814w_err = np.concatenate( (f814w_err, data[6]) )
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
obmax_conv = cf.ROI[:, 1] + 9 * cf.nsig * cf.std
obmin_conv = cf.ROI[:, 0] - 9 * cf.nsig * cf.std
# boundaries that determine which models are computed;
# these are the minimum needed for individual stars, initial convolution and probability checks
obmax = np.maximum(obmax, obmax_conv)
obmin = np.minimum(obmin, obmin_conv)
# plotting boundaries - these are just inside the area where the minimum-error density is defined
obmax_plot = cf.ROI[:, 1] + 1 * cf.nsig * cf.std
obmin_plot = cf.ROI[:, 0] - 1 * cf.nsig * cf.std

# max_margin = np.nanmax(margin, axis=0)
# obmin = np.array(cf.ROI)[:,0] - max_margin
# obmax = np.array(cf.ROI)[:,1] + max_margin
# # record the lower limit in vsini dimension at this point, even if it's negative
# # we won't calculate the prior at negative vsini, but we'll temporarily place 
# # convolved probability distribution at such values
# vsini_low = obmin[-1] 
# # remove negative vsini from the set at which we need to get the prior
# obmin[-1] = np.maximum(obmin[-1], 0) 
