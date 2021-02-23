# Calculate minimum-error probability densities for all cluster parameter combinations
import sys, os, time, pickle

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

filelist = list(np.sort(glob.glob('data/model_grids/*.pkl')))
for filepath in filelist: # for each combination of age and metallicity
	# load the pre-computed observables on a grid of model parameters
	with open(filepath, 'rb') as f:
		grid = pickle.load(f)

	print('Age: ' + str(grid.age)[:5])
	print('Metallicity: ' + str(grid.Z))

	# arrays of ordinate multipliers (weights) for the numerical integration in model space;
	# these include the varying discrete distances between adjacent abscissas
	w_Mini = trap(grid.Mini)[:, np.newaxis, np.newaxis]
	w_omega0 = trap(grid.omega0)[np.newaxis, :, np.newaxis]
	w_inc = trap(grid.inc)[np.newaxis, np.newaxis, :]
	# meshgrid of model parameters
	Mv, ov, iv = np.meshgrid(grid.Mini, grid.omega0, grid.inc, sparse=True, indexing='ij')
	# omega prior parameters
	om_max = grid.omega0.max() # about 0.83 for omega_MESA_max = 0.7
	om_mean = [0, om_max/2., om_max]
	om_sigma = [om_max/7., 3*om_max/7., om_max/7.]
	# densities for different omega distributions
	densities = []
	for j in range(len(om_mean)):
		# priors on the model grid, weighted according to the integration numerical approximation
		pr = ( Mv**-2.35 * ((ov - om_mean[j])**2 / (2 * om_sigma[j]**2)) * np.sin(iv) ) * w_Mini * w_omega0 * w_inc
		# distribute the prior over the data space grid
		ind = [] # index of each model in the data space grid
		for i in range(len(nobs)): # searchsorted only works on one array at a time
			ind.append( np.searchsorted(obs[i], grid.obs[..., i], side='right').flatten() - 1 )
		ind = np.array(ind) # make a list into a numpy array
		# choose models that are not NAN and not -1 or len(observable grid dimension) - 1, i.e. outside the grid	
		m = np.all( (ind != nobs[:, np.newaxis] - 1) & (ind != -1), axis=0 ) & \
			np.all( ~np.isnan(grid.obs), axis=-1 ).flatten()
		ind = ind[:, m]
		# transfer the prior from the model grid to the grid of observables
		prior = np.zeros(nobs, dtype=np.float32) 
		np.add.at(prior, tuple(ind), pr.flatten()[m])
		# package the prior density with the grids of observables 
		prior = du.Grid(prior, obs, cf.ROI, cf.norm, grid.age, grid.Z)

		# convolve and normalize the prior with the minimum-error Gaussians in each observable dimension
		start = time.time()
		density = prior.copy()
		kernels = [ du.Kernel(cf.std[i]/density.step[i], nsig, ds=cf.downsample) for i in range(len(cf.std)) ]
		for i in range(len(density.obs)): # for each observable dimension
			kernel = kernels[i]
			# check that the kernel, evaluated at the ROI boundaries, fits within the grid
			if density.check_roi(i, kernel): 
				density.convolve(i, kernel, ds=cf.downsample)
		print('Rotation population ' + str(j))
		print('\tFirst convolution: ' + str(time.time() - start) + ' seconds.') 
		density.normalize()

		# calculate the dependence of probability change on standard deviation of further convolving kernel
		start = time.time()
		density.dP_sigma(nsig)
		print( '\tEstimation of de-normalization: ' + str(time.time() - start) + ' seconds.' )

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
		print('\tDensity for vsini = 0: ' + str(time.time() - start) + ' seconds.')

		densities.insert(0, [density, density_cmd, density_v0])

	# save the densities
	with open('data/densities/pkl/density_' + str(grid.age).replace('.','p')[:4] + '_' + \
		str(grid.Z).replace('-', 'm').replace('.', 'p') + '.pkl', 'wb') as f:
		    pickle.dump(densities, f)