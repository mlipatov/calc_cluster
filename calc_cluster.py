import sys, os, time, pickle

from lib import dens_util as du
import config as cf
from lib import load_data as ld

import numpy as np
import glob
from scipy.optimize import root_scalar
from scipy.special import erf
from scipy.ndimage import map_coordinates

eps = np.finfo(float).eps

def dlogL(x, qi):
	if (x == 1.): q = 1 - eps
	elif (x == 0.): q = eps
	else: q = x	  
	return np.sum(1 / (q - qi))

nsig = cf.nsig - 1 # number of standard deviations to extend Gaussian kernels
npts = ld.obs.shape[0] # number of data points
ndim = ld.obs.shape[1] # number of observable dimensions

# background probability density
back = np.empty(npts)
mn = np.isnan(ld.obs[:, -1]) # mask of observations with no vsini
m0 = ld.obs[:, -1] == -1 # mask of observations with vsini = 0
mv = ~mn & ~m0 # observations with valid vsini
# on the CMD
back[mn] = 1 / cf.volume_cmd 
# at the vsini = 0 boundary
back[m0] = cf.v0err * cf.std[-1] / (np.sqrt(2 * np.pi) * cf.volume) 
# everywhere else
back[mv] = ( 1 + erf(ld.obs[mv, -1] / (np.sqrt(2) * cf.v0err * cf.std[-1])) ) / (2 * cf.volume)

# load a single density to get the arraysof observables
with open('data/densities/pkl/density_8p99_0p0.pkl', 'rb') as f:
	density = pickle.load(f) 

# residual standard deviation of data points, sigma^2 - sigma_0^2, in pixels
res = ld.err**2 - cf.std[np.newaxis, :]**2 
res[ np.less(res, 0, where=~np.isnan(res)) ] = 0 # correct for round-off
sigma = np.sqrt(res) / density.step[np.newaxis, :]

# fractional indices of data points in observables arrays, 
# a.k.a. observables of stars in pixels, offset by the zero-indexed observable
obs = np.empty_like(ld.obs, dtype=float)
for j in range(ndim):
	obs[:, j] = (ld.obs[:, j] - density.obs[j][0]) / density.step[j]

# start and stop indices of kernels in the observables arrays
obs0 = np.floor(obs)
obs1 = np.floor(obs - nsig * sigma)
obs2 = np.ceil(obs + nsig * sigma + 1)

kernels = [] # error kernels at data points
slices = [] # corresponding slices in the density arrays
for i in range(npts): # data point
	kernel = None
	slc = []
	for j in range(ndim): # dimension
		# check that the observable exists and isn't in a boundary bin
		if ~np.isnan(ld.obs[i, j]) and ld.obs[i, j] != -1:
			s = sigma[i, j]
			x = obs[i, j]
			# compare the standard deviation to pixel size
			if s < 1./2: 
				# interpolate linearly: weights are distances to opposite neighbors;
				# this approximates the kernel as a delta function
				x0 = obs0[i, j]
				kernel_j = np.array([x0 + 1 - x, x - x0])
				slc.append( slice(x0.astype(int), (x0 + 1).astype(int), None) )
			else:
				# multiply by a wide kernel
				x1 = obs1[i, j]; x2 = obs2[i, j]
				kernel_j = np.exp( -(np.arange(x1, x2) - x)**2 / (2*s**2) )
				slc.append( slice(x1.astype(int), x2.astype(int), None) )
			# add the dimension to the kernel
			if kernel is None: 
				kernel = kernel_j
			else: 
				kernel = np.multiply.outer(kernel, kernel_j)
	kernel /= np.sum(kernel)
	kernels.append(kernel)
	slices.append( tuple(slc) )

filelist = list(np.sort(glob.glob('data/densities/pkl/*.pkl')))
# filelist = list(np.sort(glob.glob('data/densities/pkl/density_8p99_m0p6.pkl')))
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

	start = time.time()
	# for data points where vsini is at the lower ROI boundary, convolve in vsini with the residual error kernel;
	# cannot and should not re-normalize after the convolution; integrate the probability beyond the lower boundary
	s = cf.std[-1] * np.sqrt(cf.v0err**2 - 1) # residual sigma = sqrt( sigma^2 - sigma_0^2 )
	kernel = du.Kernel(s / density.step[-1], nsig)
	density_v0 = density.copy()
	if density_v0.check_roi(-1, kernel): 
		density_v0.convolve(-1, kernel)
	density_v0.integrate_lower(2)
	print('density for vsini = 0: ' + str(time.time() - start) + ' seconds.')
	
	start = time.time()
	# cluster model densities at data points
	f = [] 
	# maximum absolute de-normalization
	max_dp = 0
	for i in range(npts): # for each star

		if np.isnan(ld.obs[i, -1]): # if vsini is not known
			# cluster model probability density is in color-magnitude space only
			density1 = density_cmd
		elif ld.obs[i, -1] == -1: # if vsini is in the zero bin
			# cluster model density collects at the vsini = 0 boundary
			density1 = density_v0
		else:
			# density in all observables
			density1 = density 

		# integrate with the kernel
		dens = np.sum(kernels[i] * density1.dens[slices[i]])
		dens /= np.prod(density1.step) # scale by density step sizes

		# normalization correction for this data point in this density grid
		norm = 1.
		for j in range(density1.dim):
			s = sigma[i, j] * density1.step[j] # standard deviation in units of the observable
			dP_spline = density1.correction[j] 
			if dP_spline is not None: # the spline function exists 
				if (s <= dP_spline.x[j]): # if not above the range of the spline
					dp = float( dP_spline(s) ) # evaluate the spline
				else: # extrapolate linearly from the last two points
					x0 = dP_spline.x[-1]; y0 = dP_spline.y[-1]
					x1 = dP_spline.x[-2]; y1 = dP_spline.y[-2]
					dp = y0 + (s - x0) * (y1 - y0) / (x1 - x0)
				norm *= 1 / (1 + dp) # update the re-normalization factor
				if max_dp < np.abs(dp): max_dp = np.abs(dp) # update maximum de-normalization

		# apply the normalization correction to get the cluster model density at data point
		f.append( float(dens * norm) )
	f = np.array(f, dtype=float) # cluster model densities at data points
	r = f / back # ratio of cluster model density to background density
	qi = 1 / (1 - r) # zeros of the likelihood
	print(str(ld.obs.shape[0]) + ' stars: ' + str(time.time() - start) + ' seconds.')
	print('maximum absolute de-normalization: ' + str(max_dp))

	### marginalize in q under uniform prior
	# maximum of the likelihood w.r.t. q
	if (dlogL(0, qi) <= 0) and (dlogL(1, qi) <= 0): qmax = 0
	elif (dlogL(0, qi) >= 0) and (dlogL(1, qi) >= 0): qmax = 1
	else:
		qmax = root_scalar(dlogL, args=(qi,), bracket=[0., 1.]).root
	# the nth root of maximum likelihood, where n is the number of data points
	nLmax = np.prod(np.power(1 - qmax / qi, 1. / npts))
	# maximum log likelihood
	llmax = npts * np.log(nLmax)
	# q values and integration weights
	q = np.linspace(0, 1, 1000)
	dq = q[1] - q[0]
	w = np.ones_like(q); w[0] = w[-1] = 1./2 # trapezoidal rule
	# likelihood factors on a grid of q and data points, 
	# each divided by the nth root of maximum likelihood
	lf = (1 - q[:, np.newaxis] / qi[np.newaxis, :]) / nLmax
	# likelihood vs q, divided by the maximum likelihood
	l = np.prod(lf, axis=-1)
	# logarithm of the integral, with logarithm of maximum likelihood added back
	ll = np.log(np.sum(l * w) * dq) + llmax
	print('log likelihood: ' + str(ll))