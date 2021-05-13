# Conduct all the operations on models grids that produce 
# 	isochrone probability densities at data points for each rotational/multiplicity population.
# This includes 
#	refinement of (M, omega, i) grids at r = 0 and different t,
# 	computation of the observables on (M, r, omega, i) grids at different t,
# 	checking that observable differences between neighboring models are small enough in the r dimension,
#	checking that the differences between neighboring isochrones are small enough,
#	placing the prior on the observable grid, convolving it with the minimum-error kernel and coarsening,
#	integrating the convolved prior with the residual error kernel for each data point.

# PARS imports
import sys, os, time, pickle
sys.path.append(os.path.abspath(os.path.join('..', 'paint_atmospheres')))
from pa.lib import surface as sf
from pa.lib import util as ut
from pa.opt import grid as gd
# cluster parameters imports
from lib import mist_util as mu
from lib import dens_util as du
import load_data as ld
import config as cf
# Python imports
import numpy as np
from scipy.special import erf

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
it = 105 # first index of the MIST ages to take
lt = 5; splits = [lt] * (nt - 1)  # number of ages for each interval to give linspace
t = np.unique(st.t)[it : it + nt] # ages around 9.154
st.select(np.isin(st.t, t)) # select the ages
# split time intervals: each array begins and ends with a MIST grid age, intermediate ages in between
ts = [np.linspace(t[i], t[i+1], splits[i]) for i in range(nt - 1)]
t_original = [True]
for i in range(nt - 1): t_original = [True] + [False]*(lt - 2) + t_original
t = np.unique(np.concatenate(ts)) # refined ages
# non-rotating models at these ages and full mass range
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

nsig = cf.nsig - 1 # number of standard deviations to extend Gaussian kernels
npts = ld.obs.shape[0] # number of data points
nrot = len(cf.om_mean) # number of rotational populations
nmul = len(cf.mult) # number of multiplicity populations
ndim = ld.obs.shape[1] # number of observable dimensions

### quantities needed for the computation of probability densities at point locations;
### assumes all observable grids have the same step
## background probability density
back = np.empty(npts)
mn = np.isnan(ld.obs[:, -1]) # mask of observations with no vsini
m0 = ld.obs[:, -1] == -1 # mask of observations with vsini = 0
mv = ~mn & ~m0 # observations with valid vsini
# on the CMD
back[mn] = 1 / cf.volume_cm 
# at the vsini = 0 boundary
back[m0] = cf.v0err * cf.std[-1] / (np.sqrt(2 * np.pi) * cf.volume) 
# everywhere else
back[mv] = ( 1 + erf(ld.obs[mv, -1] / (np.sqrt(2) * ld.std[mv, -1])) ) / (2 * cf.volume)

# residual standard deviation of data points, sigma^2 - sigma_0^2, in coarse pixels
res = ld.std**2 - cf.std[np.newaxis, :]**2 
res[ np.less(res, 0, where=~np.isnan(res)) ] = 0 # correct for round-off
sigma = np.sqrt(res) / (ld.step[np.newaxis, :] * cf.downsample)

# probability densities at data point locations
# dimensions: age, multiplicity population, rotational population, data point
points = np.full( (len(t), nmul, nrot, npts), np.nan )
for it in range(len(t)):
	### refinement of (M, omega, i) grids at r = 0 and different t
	print('\nt = ' + '%.4f' % t[it]); t_str = '_t' + ('%.4f' % t[it]).replace('.', 'p')
	# select age
	st1 = st.copy(); st1.select_age( t[it] ) # primary model set
	stc1 = stc.copy(); stc1.select_age( t[it] )	# non-rotating model set for the companions
	# refine model grids
	start = time.time(); print('refining the mass, omega and inclination grids...', flush=True)
	if t_original[it]:
		grid = mu.refine_coarsen(st1, t[it])
		# save the omega and inclination grids for the use in the next age
		omega0_grid = grid.omega0
		inc_grid = grid.inc
	else:
		# refine with the omega and inclination grids fixed
		grid = mu.refine_coarsen(st1, t[it], o0=omega0_grid, inc=inc_grid)
	print('%.2f' % (time.time() - start) + ' seconds.', flush=True)

	# print plots of maximum differences versus model parameter
	grid.plot_diff(0, 'data/model_grids/png/diff_vs_Mini_' + t_str + '_' + cf.z_str + '.png')
	grid.plot_diff(1, 'data/model_grids/png/diff_vs_omega0_' + t_str + '_' + cf.z_str + '.png')
	grid.plot_diff(3, 'data/model_grids/png/diff_vs_inc_' + t_str + '_' + cf.z_str + '.png')
	# get the EEPs of models on the grid
	EEP = grid.get_EEP()

	# get non-rotating companion magnitudes on a M * r grid
	mag = mu.companion_grid(cf.r, grid.Mini, stc1, t[it], pars, cf.A_V, cf.modulus)
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
	if it > 0:
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

	# VCM, CM and vsini <= 0 densities for different rotational and multiplicity populations
	densities = [ [None for i in range(nmul)] for j in range(nrot)]
	densities_cmd = [ [None for i in range(nmul)] for j in range(nrot)]
	densities_v0 = [ [None for i in range(nmul)] for j in range(nrot)]
	# multiplicities
	for k, mult in enumerate(cf.mult):
		start = time.time() 
		m_str = '_m' + str(k)

		if (mult == 'unary'): # pick out the unary prior
			obs_models = obs_binary[:, 0, ...]
			pr_noom = pr0[:, 0, ...]
		else:
			obs_models = obs_binary
			pr_noom = pr0

		ind = [] # index of each model in the data space grid
		for i in range(len(ld.nobs)): # for each data space dimension
			ind.append( np.searchsorted(ld.obs_grids[i], obs_models[..., i], side='right').flatten() - 1 )
		ind = np.array(ind) # make a list into a numpy array
		# choose models that are not NAN and not -1 or len(observable grid dimension) - 1, i.e. outside the grid	
		m = np.all( (ind != ld.nobs[:, np.newaxis] - 1) & (ind != -1), axis=0 ) & \
			np.all( ~np.isnan(obs_models), axis=-1 ).flatten()
		ind = ind[:, m]
		# print('\tGetting the indices of binary models on observable grid: ' + '%.2f' % (time.time() - start) + ' seconds.') 

		# rotational populations
		for j in range(len(cf.om_mean)):
			rot_str = '_rot' + str(j)
			# prior on the model grid
			pr =  pr_noom * pr_om[j]
			# transfer the prior from the model grid to the grid of observables
			pr_obs = np.zeros(ld.nobs, dtype=np.float32) # initialize the prior on observable grid
			np.add.at(pr_obs, tuple(ind), pr.flatten()[m]) 
			# print('\tPlacing the binary prior on a fine grid: ' + '%.2f' % (time.time() - start) + ' seconds.') 
			# package the prior density with the grids of observables 
			density = du.Grid(pr_obs, [x.copy() for x in ld.obs_grids], cf.ROI, cf.norm, Z=cf.Z)		
			# convolve and normalize the prior with the minimum-error Gaussians in each observable dimension
			min_kernels = [ du.Kernel(cf.std[i] / density.step[i], nsig, ds=cf.downsample) \
							for i in range(len(cf.std)) ]
			for i in range(len(density.obs)): # convolve in each observable dimension
				kernel = min_kernels[i]
				# check that the kernel, evaluated at the ROI boundaries, fits within the grid
				if density.check_roi(i, kernel): 
					density.convolve(i, kernel, ds=cf.downsample) # convolve
			# normalize
			density.normalize() 
			# calculate the dependence of probability change on standard deviation of further convolving kernel
			density.dP_sigma(nsig)
			densities[j][k] = density
			# compute and save the CMD density
			density_cmd = density.copy()
			density_cmd.marginalize(2)
			densities_cmd[j][k] = density_cmd
			# for data points where vsini is at the lower ROI boundary, convolve in vsini with the residual error kernel;
			# cannot and should not re-normalize after the convolution; integrate the probability beyond the lower boundary
			s = cf.std[-1] * np.sqrt(cf.v0err**2 - 1) # residual sigma = sqrt( sigma^2 - sigma_0^2 )
			kernel = du.Kernel(s / density.step[-1], nsig)
			density_v0 = density.copy()
			if density_v0.check_roi(-1, kernel): 
				density_v0.convolve(-1, kernel)
			density_v0.integrate_lower(2)
			densities_v0[j][k] = density_v0
		print(mult + ' convolutions: ' + str(time.time() - start) + ' seconds.') 
	# at the first time point, 
	# calculate residual kernels and corresponding slices for individual data points
	if it == 0: kernels, slices = du.calc_kernels(densities[0][0], sigma, nsig)

	# save the convolved priors; this takes up lots of memory, only do it if you want to plot these densities
	with open('data/densities/pkl/density' + t_str + '.pkl', 'wb') as f:
		pickle.dump(densities, f)

	# calculate the probability densities at data point locations
	f = np.zeros( (npts, nrot, nmul), dtype=float ) 
	# maximum absolute de-normalization
	max_dp = 0
	start = time.time()
	for j in range(nrot): # for each rotational population
		for k in range(nmul): # for each multiplicity population
			for i in range(npts): # for each star
				# status w.r.t. the vsini measurement
				if np.isnan(ld.obs[i, -1]): density1 = densities_cmd[j][k] # sigma_vsini = infinity				
				elif ld.obs[i, -1] == -1: 	density1 = densities_v0[j][k] # vsini = v_0 = 0
				else: 						density1 = densities[j][k] # vsini > v_0 
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
				# dimensions: age, multiplicity population, rotational population, data point
				points[it, k, j, i] = float(dens * norm)
# save the data point densities
with open('data/points.pkl', 'wb') as f:
	pickle.dump([points, t], f)