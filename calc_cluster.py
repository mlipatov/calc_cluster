import sys, os, time, pickle

from lib import dens_util as du
import config as cf
from lib import load_data as ld

import numpy as np
import glob
from scipy.optimize import minimize
from scipy.special import erf

# # minus the log likelihood of data for a given q
# def minus_ll(q, prob, volume):
# 	if q > 1.: q = 1.
# 	if q < 0.: q = 0. 
# 	return -1 * np.sum(np.log(q * prob + (1 - q) / volume))

nsig = cf.nsig - 1 # number of standard deviations to extend Gaussian kernels

# background probability density
back = np.empty(ld.obs.shape[0])
m1 = np.isnan(ld.obs[:, -1])
m2 = ld.obs[:, -1] == -1
m = ~m1 & ~m2
# on the CMD
back[m1] = 1 / cf.volume_cmd 
# at the vsini = 0 boundary
back[m2] = cf.v0err * cf.std[-1] / (np.sqrt(2 * np.pi) * cf.volume) 
# everywhere else
back[m] = ( 1 + erf(ld.obs[m, -1] / (np.sqrt(2) * cf.v0err * cf.std[-1])) ) / (2 * cf.volume)

filelist = list(np.sort(glob.glob('data/densities/pkl/*.pkl')))
# filelist = list(np.sort(glob.glob('data/densities/pkl/density_9p03_m0p3.pkl')))
for filepath in filelist: # for each combination of age and metallicity
	# load the pre-computed density on a grid of observables
	with open(filepath, 'rb') as f:
		# this should be normalized; its de-normalization functions should be computed
		density = pickle.load(f) 

	print('Age: ' + str(density.age)[:5])
	print('Metallicity: ' + str(density.Z))

	# for data points where vsini is not known, a version of the probability density only on the CMD;
	# marginalize and re-normalize in the vsini dimension; shouldn't need to re-normalize since marginalization
	# and normalization happen in the same region (all reals)
	density_cmd = density.copy()
	density_cmd.marginalize(2)

	# for data points where vsini is at the lower ROI boundary, convolve in vsini with the residual error kernel;
	# cannot and should not re-normalize after the convolution; integrate the probability beyond the lower boundary
	sigma = cf.std[-1] * np.sqrt(cf.v0err**2 - 1) # residual sigma = sqrt( sigma^2 - sigma_0^2 )
	kernel = du.Kernel(sigma / density.step[-1], nsig)
	density_v0 = density.copy()
	if density_v0.check_roi(-1, kernel): 
		density_v0.convolve(-1, kernel)
	density_v0.integrate_lower(2)
	
	start = time.time()
	# cluster model densities at data points
	f = [] 
	# maximum absolute de-normalization
	max_dp = 0
	for i in range(ld.obs.shape[0]): # for each star
		if np.isnan(ld.obs[i, -1]): # if vsini is not known
			# initialize cluster model probability density and background density 
			# to those in color-magnitude space only
			density1 = density_cmd.copy() 
		elif ld.obs[i, -1] == -1: # if vsini is in the zero bin
			# initialize the cluster model density and the background density 
			# to those that collect at the vsini = 0 boundary
			density1 = density_v0.copy()
		else:
			# initialize probability density in all observables
			density1 = density.copy() 
		j = density1.dim - 1 # initialize the focal dimension to the last
		norm = 1. # initialize the re-normalization factor
		while j >= 0:
			res = ld.err[i, j]**2 - cf.std[j]**2 # residual variance, sigma^2 - sigma_0^2
			if res < 0: res = 0 # correct for round-off
			sigma = np.sqrt(res) # residual standard deviation in observable units
			# the normalization correction for the last dimension in the current density grid
			dP_spline = density1.correction[j] 
			if dP_spline is not None: # the spline function exists 
				if (sigma <= dP_spline.x[j]): # if not above the range of the spline
					dp = float( dP_spline(sigma) ) # evaluate the spline
				else: # extrapolate linearly from the last two points
					x0 = dP_spline.x[-1]; y0 = dP_spline.y[-1]
					x1 = dP_spline.x[-2]; y1 = dP_spline.y[-2]
					dp = y0 + (sigma - x0) * (y1 - y0) / (x1 - x0)
				norm *= 1 / (1 + dp) # update the re-normalization factor
				if max_dp < np.abs(dp): max_dp = np.abs(dp) # update maximum de-normalization
			# find the broadened probability density at the data point
			try:
				density1.broadened_at_point(j, sigma, nsig, ld.obs[i, j])
			except du.ConvolutionException as err:
			    print('Convolution Error: ' + err.message)
			j -= 1 # decrement the focal dimension
		# apply the normalization correction to get the cluster model density at data point
		f.append( float(density1.density() * norm) )
	f = np.array(f, dtype=float) # cluster model densities at data points
	r = f / back # ratio of cluster model density to background density
	print('computation of ' + str(ld.obs.shape[0]) + ' stars: ' + str(time.time() - start) + ' seconds.')
	print('maximum absolute de-normalization: ' + str(max_dp))

	### marginalize in q with uniform prior
	q = np.linspace(0, 1, 1000)
	dq = q[1] - q[0]
	w = np.ones_like(q); w[0] = w[-1] = 1./2 # trapezoidal rule
	# likelihood factors on a grid of q and data points
	lf = 1 - q[:, np.newaxis] * (1 - r[np.newaxis, :])
	## likelihoods at q
	eps = np.finfo(float).eps # machine epsilon for float computations
	log_eps = np.log(eps) # natural log of machine epsilon
	# log likelihood, calculated by adding logs of likelihood factors 
	# everywhere except at q = 1, where due to round-off error argument of log could be negative 
	ll = np.empty_like(q, dtype=float)
	ll[:-1] = np.sum(np.log(lf[:-1]), axis=-1)
	# likelihood array is set to the exponent of log likelihood everywhere except at q = 1
	l = np.empty_like(q, dtype=float)
	l[:-1] = np.exp(ll[:-1])
	# if one of the density ratios is below epsilon, set likelihood to zero at q = 1
	if np.any(r <= eps): 
		l[-1] = 0
	else: # set it to the exponent of the sum of log likelihoods, as above
		ll[-1] = np.sum(np.log(lf[-1]))
		l[-1] = np.exp(ll[-1])
	# finally, set likelihood to zero wherever it's logarithm is below log-epsilon
	l[ll < log_eps] = 0
	# integrate
	print('log likelihood: ' + str(np.log(np.sum(l * w) * dq)))
