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

# derivative of log likelihood w.r.t. q at q = x
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

filelist = list(np.sort(glob.glob('data/densities/pkl/*.pkl')))
# filelist = list(np.sort(glob.glob('data/densities/pkl/density_9p15_m0p45*.pkl')))
# load a single density to get the arrays of observables
with open(filelist[0], 'rb') as f:
	densities = pickle.load(f) 
	density = densities[0][0][0]

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

## compute likelihoods on a grid of age, metallicity, 
## rotational population and multiplicity population proportions
# the sum of the two rotational population will be constrained to <=1
w0 = np.linspace(0, 1, 21, dtype=float) # proportion of the zero rotational population
w1 = np.linspace(0, 1, 21, dtype=float) # proportion of the maximum rotational population
b = np.linspace(0, 1, 21, dtype=float) # proportion of the binaries population

for filepath in filelist: # for each combination of age and metallicity
	# load the pre-computed density on a grid of observables
	with open(filepath, 'rb') as f:
		# the densities should be normalized; their de-normalization functions should be computed
		densities = pickle.load(f)
		# densities.reverse() # remove the reversal after re-calculating the densities
	age = densities[0][0][0].age
	met = densities[0][0][0].Z
	nrot = len(densities) - 1 # number of rotational populations
	nmul = len(densities[0]) # number of multiplicity populations
	om_sigma = densities[-1]

	print('Age: ' + str(age)[:5])
	print('Metallicity: ' + str(met))

	# cluster model densities at data points for each rotational population
	f = np.zeros( (npts, nrot, nmul), dtype=float ) 
	# maximum absolute de-normalization
	max_dp = 0
	start = time.time()
	for r in range(nrot): # for each rotational population
		for m in range(nmul): # for each multiplicity population
			for i in range(npts): # for each star
				# status w.r.t. vsini measurement
				if np.isnan(ld.obs[i, -1]): density1 = densities[r][m][1]				
				elif ld.obs[i, -1] == -1: 	density1 = densities[r][m][2]
				else: 						density1 = densities[r][m][0] 
				# integration with the kernel
				dens = np.sum(kernels[i] * density1.dens[slices[i]])
				dens /= np.prod(density1.step) # scale by density step sizes
				# normalization correction for this data point in this density grid
				norm = 1.
				for d in range(density1.dim):
					s = sigma[i, d] * density1.step[d] # standard deviation in units of the observable
					dP_spline = density1.correction[d] 
					if dP_spline is not None: # the spline function exists 
						if (s <= dP_spline.x[d]): # if not above the range of the spline
							dp = float( dP_spline(s) ) # evaluate the spline
						else: # extrapolate linearly from the last two points
							x0 = dP_spline.x[-1]; y0 = dP_spline.y[-1]
							x1 = dP_spline.x[-2]; y1 = dP_spline.y[-2]
							dp = y0 + (s - x0) * (y1 - y0) / (x1 - x0)
						norm *= 1 / (1 + dp) # update the re-normalization factor
						if max_dp < np.abs(dp): max_dp = np.abs(dp) # update maximum de-normalization
				# cluster model density at this data point for this rotational and multiplicity populations
				f[i, r, m] = float(dens * norm)
	print('probability densities for ' + str(ld.obs.shape[0]) + ' data points: ' + \
		str(time.time() - start) + ' seconds.') # about half a second
	# print('maximum absolute de-normalization: ' + str(max_dp)) # should be about 0.1

	## marginalize in q under uniform prior for combinations of population proportions
	ll = np.full( (len(w0), len(w1), len(b)), np.nan )
	qm = np.full_like( ll, np.nan ) # array of maximum-likelihood q
	start = time.time()
	for i in range(len(w0)):
		for j in range(len(w1) - i):
			for k in range(len(b)):
				wi0 = w0[i]; wi1 = w1[j]; bi = b[k];
				# cluster model density, as a weighted sum of rotational and multiplicity population densities
				fc = (1 - bi) * ( f[:, 0, 0] * wi0 + f[:, 1, 0] * (1 - wi0 - wi1) + f[:, 2, 0] * wi1 ) + \
						   bi * ( f[:, 0, 1] * wi0 + f[:, 1, 1] * (1 - wi0 - wi1) + f[:, 2, 1] * wi1 )
				r = fc / back # ratio of cluster model density to background density
				qi = 1 / (1 - r) # zeros of the likelihood

				# maximum of the likelihood w.r.t. q
				if (dlogL(0, qi) <= 0) and (dlogL(1, qi) <= 0): qmax = 0
				elif (dlogL(0, qi) >= 0) and (dlogL(1, qi) >= 0): qmax = 1
				else: qmax = root_scalar(dlogL, args=(qi,), bracket=[0., 1.]).root
				qm[i, j, k] = qmax
				# the nth root of maximum likelihood, where n is the number of data points
				nLmax = np.prod(np.power(1 - qmax / qi, 1. / npts))
				# maximum log likelihood
				llmax = npts * np.log(nLmax)
				# q values and integration weights
				q = np.linspace(0, 1, 201)
				dq = q[1] - q[0]
				w = np.ones_like(q); w[0] = w[-1] = 1./2 # trapezoidal rule
				# likelihood factors on a grid of q and data points, 
				# each divided by the nth root of maximum likelihood
				lf = (1 - q[:, np.newaxis] / qi[np.newaxis, :]) / nLmax
				# likelihood vs q, divided by the maximum likelihood
				l = np.prod(lf, axis=-1)
				# logarithm of the integral, with logarithm of maximum likelihood added back
				ll[i, j, k] = np.log(np.sum(l * w) * dq) + llmax
	print('marginalization in q on a grid of w_0, w_1 and b: ' + str(time.time() - start) + ' seconds.')
	i0, i1, i2 = np.unravel_index(np.nanargmax(ll),ll.shape)
	print('max ln likelihood: ' + str(np.nanmax(ll)) + ' at w_0 = ' + str(w0[i0])[:4] + \
		', 1 - w_0 - w_1 = ' + str(1 - w0[i0] - w1[i1]) + ', w_1 = ' + str(w1[i1])[:4] +\
		' and b = ' + str(b[i2])[:4])
	print('min ln likelihood: ' + str(np.nanmin(ll)))

	# package the likelihoods, the ML q values and the rotational distribution standard deviations
	with open('data/likelihoods/pkl/ll_' + str(age).replace('.','p')[:4] + '_' + \
			str(met).replace('-', 'm').replace('.', 'p') + \
			'_os' + '_'.join([('%.2f' % n).replace('.','') for n in om_sigma]) + '.pkl', 'wb') as f:
	    pickle.dump([ll, qm, densities[-1]], f)    