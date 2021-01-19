import sys, os, time, pickle

from lib import dens_util as du
import config as cf
from lib import load_data as ld

import numpy as np
import glob
from scipy.optimize import minimize

# minus the log likelihood of data for a given q
def minus_ll(q, prob, volume):
	return -1 * np.sum(np.log(q * prob + (1 - q) / volume))

nsig = cf.nsig - 1 # number of standard deviations to extend Gaussian kernels

# filelist = list(np.sort(glob.glob('data/densities/pkl/*.pkl')))
filelist = list(np.sort(glob.glob('data/densities/pkl/density_9p15_m0p45.pkl')))
for filepath in filelist: # for each combination of age and metallicity
	# load the pre-computed density on a grid of observables
	with open(filepath, 'rb') as f:
		density = pickle.load(f)

	# calculate the probability beyond the region of interest in the vsini dimension
	## --> delete this after doing this calculation with the rest of the densities
	density.Pvsini = [None] * 2
	density.P_collected(nsig, -1, 0, np.sqrt(cf.v0_err**2 - 1) * cf.std[-1])
	density.P_collected(nsig, -1, 1, np.sqrt(cf.vmax_err**2 - 1) * cf.std[-1])

	print('Age: ' + str(density.age)[:5])
	print('Metallicity: ' + str(density.Z))

	density.scale()
	# a version of the probability density only on the CMD, for situations when vsini is not known
	density_cmd = density.copy()
	density_cmd.marginalize(2)
	density_cmd.normalize()
	density_cmd.scale()
	
	start = time.time()
	# probabilities of individual data points under the cluster model
	prob = [] 
	# maximum absolute de-normalization
	max_dp = 0
	for i in range(ld.obs.shape[0]): # for each star
		if np.isnan(ld.obs[i, -1]): # if vsini is not known
			density1 = density_cmd.copy() # initialize probability density in color-magnitude space only 
		else:
			density1 = density.copy() # initialize probability density of one star
		j = density1.dim - 1 # initialize the focal dimension to the last
		norm = 1. # initialize the re-normalization factor
		while density1.dim > 0: # while probability is on a grid with more than one entry
			res = ld.err[i, j]**2 - cf.std[j]**2 # residual variance, sigma^2 - sigma_0^2
			if res < 0: res = 0 # correct for round-off
			sigma = np.sqrt(res) # residual standard deviation in observable units
			# update the normalization correction for the last dimension in the current density grid
			dP_spline = density1.dP[-1] 
			if dP_spline is not None: # the spline function exists 
				if (sigma <= dP_spline.x[-1]): # if not above the range of the spline
					dp = float( dP_spline(sigma) ) # evaluate the spline
				else: # extrapolate linearly from the last two points
					x0 = dP_spline.x[-1]; y0 = dP_spline.y[-1]
					x1 = dP_spline.x[-2]; y1 = dP_spline.y[-2]
					dp = y0 + (sigma - x0) * (y1 - y0) / (x1 - x0)
				norm *= 1 / (1 + dp) # update the re-normalization factor
				if max_dp < np.abs(dp): max_dp = np.abs(dp) # update maximum de-normalization
			# integrate to find the probability density at the data point
			try:
				density1.integrate_kernel(-1, sigma, nsig, ld.obs[i, j])
			except du.ConvolutionException as err:
			    print('Convolution Error: ' + err.message)
			j -= 1 # decrement the focal dimension
		# apply the normalization correction to get the probability density of one star 
		# at the star's data space location when it is part of the set described by the cluster model
		p1 = density1.dens * norm 
		prob.append(float(p1))
	prob = np.array(prob, dtype=float)
	print('computation of ' + str(ld.obs.shape[0]) + ' stars: ' + str(time.time() - start) + ' seconds.')
	print('maximum absolute de-normalization: ' + str(max_dp))

	# calculate maximum-likelihood q
	q = minimize(minus_ll, 0.5, args=(prob, cf.volume), bounds=[(0., 1.)]).x
	print('proportion of stars in the ROI described by the cluster model: ' + str(q))
