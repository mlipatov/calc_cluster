# Conduct all the operations on models grids that produce 
# 	minimum-error isochrone probability distributions in observable space.
# This includes 
#	refinement of (M, omega, i) grids at r = 0 and different t,
# 	computation of the observables on (M, r, omega, i) grids at different t,
# 	checking that observable differences between neighboring models are small enough in the r dimension,
#	checking that the differences between neighboring isochrones are small enough,
#	placing the prior on the observable grid, convolving it with the minimum-error kernel and coarsening.

# PARS imports
import sys, os, time, pickle
sys.path.append(os.path.abspath(os.path.join('..', 'paint_atmospheres')))
from pa.lib import surface as sf
from pa.lib import util as ut
from pa.opt import grid as gd
# cluster imports
from lib import mist_util as mu
from lib import dens_util as du
from lib import load_data as ld
import config as cf
# Python imports
import numpy as np

# pre-compute Roche model volume versus PARS's omega
# and PARS's omega versus MESA's omega
sf.calcVA()
sf.calcom()

# Load and filter MIST models
print('Loading MIST...', end='')
start = time.time()
st = mu.Set('data/mist_grid.npy')
print('%.2f' % (time.time() - start) + ' seconds.')

st.select_MS() # select main sequence
st.select_Z(cf.Z) # select metallicity
st.select_valid_rotation() # select rotation with omega < 1
st.set_omega0() # set omega from omega_M; ignore the L_edd factor

nt = 6 # number of ages to take from the MIST grid
t = np.unique(st.t)[105 : 105 + nt] # ages around 9.154
st.select(np.isin(st.t, t)) # select the ages
splits = [5] * (nt - 1)  # number of ages for each interval to give linspace
# split time intervals: each array begins and ends with a MIST grid age, intermediate ages in between
ts = [np.linspace(t[i], t[i+1], splits[i]) for i in range(nt - 1)] 
# ages to interpolate from: 
# the age itself for a MIST grid age, the closest left and right MIST grid ages otherwise
t_interp = sum( [ [np.array([x[0]])] + list(np.tile(x.take([0, -1]), (x.shape[0] - 2, 1))) for x in ts], [] )
t_interp.append( np.array([t[-1]]) )
t = np.unique(np.concatenate(ts)) # refined ages
# non-rotating models at these ages; full mass range
stc = st.copy(); stc.select(stc.omega0 == 0)  

print('Loading PARS...', end='', flush=True)
start = time.time()
with open('data/pars_grid_2.pkl', 'rb') as f: pars = pickle.load(f)
print('%.2f' % (time.time() - start) + ' seconds.' + '\n', flush=True)
mu.Grid.pars = pars # give a PARS grid reference to the grid class
# apply the lower mass cut-off for the primaries according the region of interest on the CMD 
start = time.time()
print('Calculating the mass cut-off...', end='', flush=True)
Mmin = mu.Mlim(st)
st.select_mass(Mmin=Mmin)
print('minimum mass = ' + '%.4f' % Mmin + '; %.2f' % (time.time() - start) + ' seconds.', flush=True)

# number of entries needed by data space grids with ranges that will allow 
# probability leakage checks, the initial convolution and calculations for individual stars;
# with coarse steps at least as small as the minimum standard deviations
nobs = cf.downsample * ( np.ceil((ld.obmax - ld.obmin) / cf.std + 1).astype(int) )
obs = [] # observable grids
for i in range(len(nobs)):
	ogrid = np.linspace(ld.obmin[i], ld.obmax[i], nobs[i]) # grid for this observable
	obs.append(ogrid)

nsig = cf.nsig - 1 # number of standard deviations to extend Gaussian kernels

for i in range(len(t)):
	### refinement of (M, omega, i) grids at r = 0 and different t
	print('\nt = ' + '%.4f' % t[i]); t_str = '_t' + ('%.4f' % t[i]).replace('.', 'p')
	# select ages to interpolate from
	st1 = st.copy(); st1.select( np.isin(st1.t, t_interp[i]) )		# primary model set
	stc1 = stc.copy(); stc1.select( np.isin(stc1.t, t_interp[i]) )	# non-rotating model set for the companions
	# refine model grids
	start = time.time(); print('refining the mass, omega and inclination grids...', flush=True)
	grid = mu.refine_coarsen(st1, t[i]); print('%.2f' % (time.time() - start) + ' seconds.', flush=True)
	# print plots of maximum differences versus model parameter
	grid.plot_diff(0, 'data/model_grids/png/diff_vs_Mini_' + t_str + '_' + cf.z_str + '.png')
	grid.plot_diff(1, 'data/model_grids/png/diff_vs_omega0_' + t_str + '_' + cf.z_str + '.png')
	grid.plot_diff(3, 'data/model_grids/png/diff_vs_inc_' + t_str + '_' + cf.z_str + '.png')
	# get the EEPs of models on the grid
	EEP = grid.get_EEP()

	# get non-rotating companion magnitudes on a M * r grid
	mag = mu.companion_grid(cf.r, grid.Mini, stc1, t[i], pars, cf.A_V, cf.modulus)
	# combine the magnitudes of the non-rotating companion and its primary;
	# companion dimensions: initial primary mass, binary mass ratio, filter
	# primary dimensions: initial primary mass, initial omega, age, inclination, filter
	# final dimensions: initial primary mass, binary mass ratio, initial omega, age, inclination, filter
	mag_binary = \
		mu.combine_mags(mag[..., np.newaxis, np.newaxis, np.newaxis, :], grid.mag[:, np.newaxis, ...])
	# insert the unary magnitudes, which evaluated to NAN above
	mag_binary[:, 0, ...] = grid.mag
	# compute the observables of binary models;
	# dimensions: initial primary mass, binary mass ratio, initial omega, inclination, mag / color / vsini;
	# the last dimension remains the same if three filters are now magnitude, color and vsini
	# print('Combining primary and companion magnitudes: ' + str(time.time() - start) + ' seconds.') 
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
		# left boundary of the first interval in r dimension with excessive magnitude difference
		print('magnitude differences are small up to r = ' + str(cf.r[:-1][ind[0]]))
		# cull the magnitude and r arrays
		cf.r = cf.r[:ind[0]+1]
		obs_binary = obs_binary[:, :ind[0]+1, ...]

	# compute the distance from the previous-age isochrone
	if i > 0:
		emin = int(np.floor(min(np.nanmin(EEP_prev), np.nanmin(EEP))))
		emax = int(np.ceil(max(np.nanmax(EEP_prev), np.nanmax(EEP))))
		EEPrange = np.array(range(emin, emax + 1))
		numbers = []
		diff = []
		for eep in EEPrange:
			# indices of locations in the EEP grids where EEP rounds to some integer
			i0 = np.argwhere(np.around(EEP_prev) == eep)
			i1 = np.argwhere(np.around(EEP) == eep)
			# magnitudes of corresponding models: look at all r and all i
			obs0 = obs_binary_prev[i0.T[0], :, i0.T[1], i0.T[2]].reshape(-1, 3)
			obs1 = obs_binary[i1.T[0], :, i1.T[1], i1.T[2]].reshape(-1, 3)
			obs0 = obs0[~np.isnan(obs0[:, 0])]
			obs1 = obs1[~np.isnan(obs1[:, 0])]
			if obs0.shape[0] > 0 and obs1.shape[0] > 0: 
				diff_means = np.mean(obs1, axis=0) - np.mean(obs0, axis=0)
				numbers.append(obs0.shape[0] + obs1.shape[0])
				diff.append(diff_means)
		numbers = np.array(numbers)
		diff = np.array(diff)
		d = np.nansum(diff * numbers[:, np.newaxis], axis=0) / np.nansum(numbers)
		print('magnitude, color, vsini minimum-error distances from the previous isochrone: ' +\
			', '.join('%.4f' % x for x in d/cf.std))

	# make copies of observables and EEPs for the next iteration
	obs_binary_prev = obs_binary.copy()
	EEP_prev = EEP.copy()

	# arrays of ordinate multipliers (weights) for the numerical integration in model space;
	# these include the varying discrete distances between adjacent abscissas;
	# dimensions: mass, r, omega, inclination
	w_Mini = du.trap(grid.Mini)[:, np.newaxis, np.newaxis, np.newaxis]
	w_r = du.trap(cf.r)[np.newaxis, :, np.newaxis, np.newaxis]
	w_omega0 = du.trap(grid.omega0)[np.newaxis, np.newaxis, :, np.newaxis]
	w_inc = du.trap(grid.inc)[np.newaxis, np.newaxis, np.newaxis, :]
	# non-uniform priors in non-omega model dimensions
	pr_Mini = (grid.Mini**-2.35)[:, np.newaxis, np.newaxis, np.newaxis]
	# pr_t = np.exp(-0.5*((t - age_mean) / age_sigma)**2)[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
	pr_inc = np.sin(grid.inc)[np.newaxis, np.newaxis, np.newaxis, :]
	# overall prior without the omega distribution: prior on r is flat
	pr0 = pr_Mini * pr_inc * (w_Mini * w_r * w_omega0 *  w_inc)
	# omega distribution prior; 
	# dimensions: rotational population, omega
	pr_om = np.exp(-0.5*((grid.omega0[np.newaxis, :] - cf.om_mean[:, np.newaxis]) / cf.om_sigma[:, np.newaxis])**2)
	# dimensions: rotational population, mass, r, omega, inclination
	pr_om = pr_om[:, np.newaxis, np.newaxis, :, np.newaxis]

	# densities for different rotational and multiplicity populations
	densities = [ [None for i in range(2)] for j in range(len(cf.om_mean))]
	# multiplicities
	for k, mult in enumerate(['unary', 'binary']):
		start = time.time() 
		m_str = '_m' + str(k)

		if (mult == 'unary'): # pick out the unary prior
			obs_models = obs_binary[:, 0, ...]
			pr_noom = pr0[:, 0, ...]
		else:
			obs_models = obs_binary
			pr_noom = pr0

		ind = [] # index of each model in the data space grid
		for i in range(len(nobs)): # for each data space dimension
			ind.append( np.searchsorted(obs[i], obs_models[..., i], side='right').flatten() - 1 )
		ind = np.array(ind) # make a list into a numpy array
		# choose models that are not NAN and not -1 or len(observable grid dimension) - 1, i.e. outside the grid	
		m = np.all( (ind != nobs[:, np.newaxis] - 1) & (ind != -1), axis=0 ) & \
			np.all( ~np.isnan(obs_models), axis=-1 ).flatten()
		ind = ind[:, m]
		# print('\tGetting the indices of binary models on observable grid: ' + '%.2f' % (time.time() - start) + ' seconds.') 

		# rotational populations
		for j in range(len(cf.om_mean)):
			rot_str = '_rot' + str(j)
			# prior on the model grid
			pr =  pr_noom * pr_om[j]
			# transfer the prior from the model grid to the grid of observables
			pr_obs = np.zeros(nobs, dtype=np.float32) # initialize the prior on observable grid
			np.add.at(pr_obs, tuple(ind), pr.flatten()[m]) 
			# print('\tPlacing the binary prior on a fine grid: ' + '%.2f' % (time.time() - start) + ' seconds.') 
			# package the prior density with the grids of observables 
			density = du.Grid(pr_obs, [x.copy() for x in obs], cf.ROI, cf.norm, Z=cf.Z)		
			# convolve and normalize the prior with the minimum-error Gaussians in each observable dimension
			kernels = [ du.Kernel(cf.std[i]/density.step[i], nsig, ds=cf.downsample) for i in range(len(cf.std)) ]
			for i in range(len(density.obs)): # for each observable dimension
				kernel = kernels[i]
				# check that the kernel, evaluated at the ROI boundaries, fits within the grid
				if density.check_roi(i, kernel): 
					density.convolve(i, kernel, ds=cf.downsample)
			densities[j][k] = density

		print(mult + ' convolutions: ' + str(time.time() - start) + ' seconds.') 

	# save the convolved priors
	with open('data/densities/pkl/density' + t_str + '.pkl', 'wb') as f:
		pickle.dump(densities, f)