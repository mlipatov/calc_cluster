# Calculate minimum-error probability densities for all cluster parameter combinations
import sys, os, time, pickle
sys.path.append(os.path.abspath(os.path.join('..', 'paint_atmospheres')))
from pa.lib import surface as sf

from lib import mist_util as mu
from lib import dens_util as du
import config as cf
from lib import load_data as ld

import numpy as np
import glob

# calculate the factors of each discrete ordinate
# for the trapezoidal rule with variable differentials
# Input: abscissae
def trap(x):
	if len(x) > 1:
		# calculate the differences between initial masses
		diff = np.diff(x)
		# array of averaged differentials for the trapezoidal rule
		d = 0.5 * ( np.append(diff, 0) + np.insert(diff, 0, 0) )
	else:
		d = np.array([1])
	return d

# number of entries needed by data space grids with ranges that will allow 
# probability leakage checks, the initial convolution and calculations for individual stars;
# with coarse steps at least as small as the minimum standard deviations
nobs = cf.downsample * ( np.ceil((ld.obmax - ld.obmin) / cf.std + 1).astype(int) )
obs = [] # observable grids
for i in range(len(nobs)):
	ogrid = np.linspace(ld.obmin[i], ld.obmax[i], nobs[i]) # grid for this observable
	obs.append(ogrid)

nsig = cf.nsig - 1 # number of standard deviations to extend Gaussian kernels

# slowest rotational population is centered on omega = 0, fastest on omega = 1
# standard deviations of the rotational populations
s_slow = 0.5 
s_middle = 0.1
s_fast = 0.1 
a = s_fast / s_slow
# medium rotating population: 
# mean is the location where the slow and fast distributions are equal
if a == 1:
	om_middle = 1./2
else:
	om_middle = ( 1 - a * np.sqrt(1 - 2*(1 - a**2)*np.log(a)*s_slow**2) ) / (1 - a**2)
om_mean = np.array([0, om_middle, 1])
om_sigma = np.array([s_slow, s_middle, s_fast])
print('omega means: ' + str(om_mean))
print('omega standard deviations: '+ str(om_sigma))

# age prior parameters
age_mean = 9.164
age_sigma = 0.015

print('-------- Observables on model grids ----------')
obs_allages = []
filelist = list(np.sort(glob.glob('data/model_grids/pkl/*.pkl')))
# filelist = list(np.sort(glob.glob('data/model_grids/pkl/grid_t9p19463_9p21477_m0p45.pkl')))
start = time.time()
for filepath in filelist: # for each age range
	# load the pre-computed observables on a grid of model parameters
	with open(filepath, 'rb') as f:
		grid, mag, r = pickle.load(f)

	print('Age: ' + '%.4f' % grid.t.min() )# + ' - ' '%.4f' % grid.t.max())

	# # observables of unary models
	# # dimensions: initial mass, initial omega, age, inclination, mag / color / vsini
	# obs_unary = np.full( grid.mag.shape, np.nan )
	# obs_unary[..., 0] = grid.mag[..., 1] # F555W magnitude
	# obs_unary[..., 1] = grid.mag[..., 0] - grid.mag[..., 2] # F435W - F814W color
	# obs_unary[..., 2] = grid.vsini # vsini

	# combined magnitudes of the non-rotating companion and its primary;
	# companion dimensions: initial primary mass, binary mass ratio, filter
	# primary dimensions: initial primary mass, initial omega, age, inclination, filter
	# final dimensions: initial primary mass, binary mass ratio, initial omega, age, inclination, filter
	mag_binary = \
		mu.combine_mags(mag[..., np.newaxis, np.newaxis, np.newaxis, :], grid.mag[:, np.newaxis, ...])
	# insert the unary magnitudes, which evaluated to NAN above
	mag_binary[:, 0, ...] = grid.mag
	# print('Combining primary and companion magnitudes: ' + str(time.time() - start) + ' seconds.') 
	# observables of binary models;
	# dimensions: initial primary mass, binary mass ratio, initial omega, inclination, mag / color / vsini;
	# the last dimension remains the same if three filters are now magnitude, color and vsini
	obs_binary = np.full( mag_binary.shape, np.nan )
	obs_binary[..., 0] = mag_binary[..., 1] # F555W magnitude
	obs_binary[..., 1] = mag_binary[..., 0] - mag_binary[..., 2] # F435W - F814W color
	obs_binary[..., 2] = grid.vsini[:, np.newaxis, ...] # vsini

	# differences in F555W magnitude along the r dimension
	diff = np.abs(np.diff(obs_binary[..., 0], axis=1))
	# mask along the r dimension where the differences are not all NaN
	m = ~np.all(np.isnan(diff), axis=(0, 2, 3, 4)) 
	# maximum differences in magnitude along the r dimension
	dm = np.full(diff.shape[1], np.nan)
	dm[m] = np.nanmax(diff[:, m,...], axis=(0, 2, 3, 4))

	# largest companions may be so close to TAMS that magnitude differences in the r dimension are too large
	ind = np.argwhere( ~np.less_equal(dm, cf.dmax * cf.std[0], where=~np.isnan(dm)) ).flatten()
	if ind.size == 0:
		print('all magnitude differences are small in the r dimension')
	else:
	# elif np.all(ind == np.array(range(len(dm) - len(ind), len(dm)))): 
		i = ind[0] # index of the first interval in r dimension with excessive magnitude difference
		print('magnitude differences are small up to r = ' + str(r[:-1][i])) # left boundary of that interval
		# cull the magnitude and r arrays
		r = r[:i+1]
		obs_binary = obs_binary[:, :i+1, ...]
	# else:
	# 	raise ValueError('intervals with excessive magnitude differences in r dimension do \
	# 		not form a contiguous set that ends at r = 1 for t = ' + str(grid.t.min())[:5] + ' and Z = ' + str(cf.Z))

	obs_allages.append([obs_binary, grid.Mini, r, grid.omega0, grid.t, grid.inc])
print('%.2f' % (time.time() - start) + ' seconds.') 

print('-------- Priors on observable grids ----------')
start = time.time()
# dimensions: rotational population, multiplicity population
priors = [ [None for i in range(2)] for j in range(len(om_mean))]
for obs_binary, Mini, r, omega0, t, inc in obs_allages:

	print('Ages: ' + '%.4f' % t.min() + ' - ' '%.4f' % t.max())
	# arrays of ordinate multipliers (weights) for the numerical integration in model space;
	# these include the varying discrete distances between adjacent abscissas;
	# dimensions: initial mass, binary mass ratio, omega, inclination
	w_Mini = trap(Mini)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
	w_r = trap(r)[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
	w_omega0 = trap(omega0)[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
	w_t = trap(t)[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
	w_inc = trap(inc)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
	# non-uniform priors in non-omega model dimensions
	pr_Mini = (Mini**-2.35)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
	pr_t = np.exp(-0.5*((t - age_mean) / age_sigma)**2)[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
	pr_inc = np.sin(inc)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
	# prior without the omega distribution
	pr0 = pr_Mini * pr_t * pr_inc * (w_Mini * w_r * w_omega0 * w_t * w_inc)
	# omega distribution prior; first dimension is the rotational population
	pr_om = np.exp(-0.5*((omega0[np.newaxis, :] - om_mean[:, np.newaxis]) / om_sigma[:, np.newaxis])**2)
	pr_om = pr_om[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]

	# multiplicities
	for k, mult in enumerate(['unary', 'binary']): 
		if (mult == 'unary'):
			obs_models = obs_binary[:, 0, ...]
			pr_noom = pr0[:, 0, ...]
		else:
			obs_models = obs_binary
			pr_noom = pr0

		# start = time.time()
		ind = [] # index of each model in the data space grid
		for i in range(len(nobs)): # for each data space dimension
			ind.append( np.searchsorted(obs[i], obs_models[..., i], side='right').flatten() - 1 )
		ind = np.array(ind) # make a list into a numpy array
		# choose models that are not NAN and not -1 or len(observable grid dimension) - 1, i.e. outside the grid	
		m = np.all( (ind != nobs[:, np.newaxis] - 1) & (ind != -1), axis=0 ) & \
			np.all( ~np.isnan(obs_models), axis=-1 ).flatten()
		ind = ind[:, m]
		# if mult == 'binary':
		# 	print('\tGetting the indices of binary models on observable grid: ' + '%.2f' % (time.time() - start) + ' seconds.') 

		# rotational populations
		for j in range(len(om_mean)):
			if mult == 'binary': 
				print('Rotational population ' + str(j))
			# prior on the model grid, weighted according to the integration numerical approximation
			pr =  pr_noom * pr_om[j]
			# initialize the prior at this rotational population and this multiplicity
			if priors[j][k] is None:
				priors[j][k] = np.zeros(nobs, dtype=np.float32)
			## distribute the prior over the data space grid
			# transfer the prior from the model grid to the grid of observables
			# start = time.time()
			np.add.at(priors[j][k], tuple(ind), pr.flatten()[m])
			# if mult == 'binary':
			# 	print('\tPlacing the binary prior on a fine grid: ' + '%.2f' % (time.time() - start) + ' seconds.') 
print('%.4f' % (time.time() - start) + ' seconds.') 

print('-------- Convolutions ----------')
# densities for different rotational and multiplicity populations
densities = [ [None for i in range(2)] for j in range(len(om_mean))]
# rotational populations
for j in range(len(om_mean)):
	print('Rotational population ' + str(j))
	# multiplicities
	for k, mult in enumerate(['Unaries', 'Binaries']):
		print('\t' + mult)
		# package the prior density with the grids of observables 
		prior = du.Grid(priors[j][k], obs, cf.ROI, cf.norm, Z=cf.Z)
		
		# # normalize the prior for plotting
		# prior.normalize()
		# # marginalize the prior in vsini
		# prior_cmd = prior.copy()
		# prior_cmd.marginalize(2)
		
		# convolve and normalize the prior with the minimum-error Gaussians in each observable dimension
		start = time.time()
		density = prior.copy()
		kernels = [ du.Kernel(cf.std[i]/density.step[i], nsig, ds=cf.downsample) for i in range(len(cf.std)) ]
		for i in range(len(density.obs)): # for each observable dimension
			kernel = kernels[i]
			# check that the kernel, evaluated at the ROI boundaries, fits within the grid
			if density.check_roi(i, kernel): 
				density.convolve(i, kernel, ds=cf.downsample)
		print('\tFirst convolution: ' + str(time.time() - start) + ' seconds.') 
		density.normalize()
		# calculate the dependence of probability change on standard deviation of further convolving kernel
		density.dP_sigma(nsig)
		# for data points where vsini is not known, a version of the probability density only on the CMD;
		# marginalize in the vsini dimension; shouldn't need to re-normalize since marginalization
		# and normalization happen in the same region (all reals); didn't need to do the initial convolution 
		# in the vsini dimension in this case, though that shouldn't matter, since that convolution multiplies the
		# total probability at a given magnitude and color by the same factor, 1 / (sigma_vsini * sqrt(2 pi))
		density_cmd = density.copy()
		density_cmd.marginalize(2)
		# for data points where vsini is at the lower ROI boundary, convolve in vsini with the residual error kernel;
		# cannot and should not re-normalize after the convolution; integrate the probability beyond the lower boundary
		s = cf.std[-1] * np.sqrt(cf.v0err**2 - 1) # residual sigma = sqrt( sigma^2 - sigma_0^2 )
		kernel = du.Kernel(s / density.step[-1], nsig)
		density_v0 = density.copy()
		if density_v0.check_roi(-1, kernel): 
			density_v0.convolve(-1, kernel)
		density_v0.integrate_lower(2)
		# save the densities
		densities[j][k] = [density, density_cmd, density_v0]

# tack on the standard deviations of the rotational population distributions
# and the parameters of the age distributions
densities.append(om_sigma) 
densities.append([age_mean, age_sigma])

# save the densities
with open('data/densities/pkl/density_' + str(cf.Z).replace('-', 'm').replace('.', 'p') + \
	'_os' + '_'.join([('%.2f' % n).replace('.','') for n in om_sigma]) + \
	'.pkl', 'wb') as f:
	    pickle.dump(densities, f)