import sys, os, time, pickle

from lib import dens_util as du
import config as cf
import load_data as ld

import numpy as np
import glob
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 12

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

	# arrays of ordinate multipliers (weights) for the numerical integration in model space;
	# these include the varying discrete distances between adjacent abscissas
	w_Mini = trap(grid.Mini)[:, np.newaxis, np.newaxis]
	w_otilde = trap(grid.otilde0)[np.newaxis, :, np.newaxis]
	w_inc = trap(grid.inc)[np.newaxis, np.newaxis, :]
	# meshgrid of model parameters
	Mv, ov, iv = np.meshgrid(grid.Mini, grid.otilde0, grid.inc, sparse=True, indexing='ij')
	# priors on the model grid, weighted according to the integration numerical approximation
	pr = ( Mv**-2.35 * 1 * np.sin(iv) ) * w_Mini * w_otilde * w_inc

	# distribute the prior times the model space differential element over the data space grid
	ind = [] # index of each model in the data space grid
	for i in range(len(nobs)): # searchsorted only works on one array at a time
		ind.append( np.searchsorted(obs[i], grid.obs[..., i], side='right').flatten() - 1 )
	ind = np.array(ind) # make a list into a numpy array
	# choose models that are not NAN and not -1 or len(observable grid dimension) - 1, i.e. outside the grid	
	m = np.all( (ind != nobs[:, np.newaxis] - 1) & (ind != -1), axis=0 ) & \
		np.all( ~np.isnan(grid.obs), axis=-1 ).flatten()
	ind = ind[:, m]
	# transfer the prior from the model grid to the grid of observables
	prior = np.zeros(nobs) 
	np.add.at(prior, tuple(ind), pr.flatten()[m])
	# package the prior density with the grids of observables 
	prior = du.Grid(prior, obs, grid.age, grid.Z)
	# normalize it
	prior.normalize(cf.RON)

	# convolve the prior with the minimum-error Gaussians in each observable dimension
	start = time.time()
	density = prior.copy()
	kernels = [ du.Kernel(cf.std[i]/density.step[i], nsig, ds=cf.downsample) for i in range(len(cf.std)) ]
	for i in range(len(density.obs)): # for each observable dimension
		kernel = kernels[i]
		# check that the kernel, evaluated at the ROI boundaries, fits within the grid
		if density.check(i, kernel, cf.ROI[i]): 
			density.convolve(i, kernel, ds=cf.downsample)
	density.normalize(cf.RON)
	# with open('data/densities/pkl/density_' + str(grid.age).replace('.','p')[:4] + '_' + \
	# 	str(grid.Z).replace('-', 'm').replace('.', 'p') + '.pkl', 'wb') as f:
	# 	    pickle.dump(density, f)
	print('first convolution: ' + str(time.time() - start) + ' seconds.')

	# calculate and plot the dependence of probability change on sigma
	suffix = os.path.basename(filepath).split('.')[0][5:]
	prefix = 'data/normalization/dP_'
	dP = du.dP_sigma(density, nsig, prefix, suffix, grid.age, grid.Z)
	# probability within the RON change at minimum sigma

	# # a version of the probability density only on the CMD, for situations when vsini is not known
	# density_cmd = density.copy()
	# density_cmd.marginalize(2)
	# ron = np.delete(cf.RON, 2, axis=0)
	# density_cmd.normalize(ron)
	
	# ## compute the probabilities of individual stars
	# start = time.time()
	# for i in range(ld.obs.shape[0]): # for each star
	# 	if np.isnan(ld.obs[i, -1]): # if vsini is not known
	# 		density1 = density_cmd.copy() # initialize probability to color-magnitude space only 
	# 	else:
	# 		density1 = density.copy() # initialize the probability of one star
	# 	j = density1.dim - 1 # initialize the focal dimension
	# 	while density1.dim > 0: # while probability is on a grid with more than one entry
	# 		res = ld.err[i, j]**2 - cf.std[j]**2 # residual variance, sigma^2 - sigma_0^2
	# 		if res < 0: res = 0 # correct for round-off
	# 		sigma = np.sqrt(res) # residual standard deviation in observable units
	# 		density1.integrate_kernel(j, sigma, nsig, ld.obs[i, j])
	# 		j -= 1 # decrement focal dimension
	# 	# print(np.log(density1.dens))
	# print('computation of ' + str(ld.obs.shape[0]) + ' stars: ' + str(time.time() - start) + ' seconds.')
	# move the probability at vsini < 0 over to the vsini = 0 bin
	# calculate maximum-likelihood q