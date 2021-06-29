# Point density calculations with enhanced mixing
# Takes up 4 Gb of memory

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
import gc 
import glob

dens_dir = 'data/mix/densities/pkl/'
points_dir = 'data/mix/points/'

nsig = cf.nsig - 1 # number of standard deviations to extend Gaussian kernels
npts = ld.obs.shape[0] # number of data points
nrot = len(cf.om_mean) # number of rotational populations
nmul = len(cf.mult) # number of multiplicity populations
ndim = ld.obs.shape[1] # number of observable dimensions

### quantities needed for the computation of probability densities at point locations;
### assumes all observable grids have the same step
# residual standard deviation of data points, sigma^2 - sigma_0^2, in coarse pixels
res = ld.std**2 - cf.std[np.newaxis, :]**2 
res[ np.less(res, 0, where=~np.isnan(res)) ] = 0 # correct for round-off
sigma = np.sqrt(res) / (ld.step[np.newaxis, :] * cf.downsample)

filelist = list(np.sort(glob.glob(cf.obs_dir + '*.pkl'))) # observables 
# get the ages; this should be optimized later
start = time.time(); print('Getting the ages...', end='', flush=True)
t_ar = []
for file in filelist:
	with open(file, 'rb') as f: 
		obs_binary, age, Mini, r, omega0, inc = pickle.load(f)
		t_ar.append(age)
t_ar = np.array(t_ar)
omax = omega0[-1] # maximum omega0 on the grids
print(str(time.time() - start) + ' seconds.', flush=True)
# set the parameters for the combining of isochrones
t0min = t_ar[0] + cf.amax * omax / np.log(10) # minimum intercept parameter
t0_ar = t_ar[(t_ar > t0min) & (t_ar >= cf.t0min) & (t_ar <= cf.t0max)] # t0 parameters
a_ar = cf.a_ar # slope parameters

# bins in omega according to the age as a function of omega
def om_bins(t, t0, a):
	om = (t0 - t) / (a / np.log(10))
	# split the omega domain into intervals corresponding to ages, with boundaries
	# halfway between omega values at each age; double the domain for the lowest age
	bins = (om[:-1] + om[1:]) / 2 # half-way points
	bins = np.insert(bins, 0, 2 * om[0] - bins[0]) # insert the left lowest-age boundary
	bins = np.append(bins, 0.) # insert the right highest-age boundary
	return bins

# initialize the priors on observable grids for all multiplicities and rotations
pr_obs = []
for j in range(nrot):
	pr_obs.append([])
	for k in range(nmul):
		pr_obs[j].append(np.zeros(ld.nobs, dtype=np.float32))
# probability densities at data point locations
# dimensions: age, multiplicity population, rotational population, data point
points = np.full( (len(t0_ar), len(a_ar), nmul, nrot, npts), np.nan )
kernels = None; slices = None # error kernels on the observable grid
for it0 in range(len(t0_ar)):
	t0 = t0_ar[it0]
	t0_str = '_t' + ('%.4f' % t0).replace('.', 'p')
	for ia in reversed(range(len(a_ar))):
		a = a_ar[ia]
		a_str = '_a' + ('%.3f' % a).replace('.', 'p')
		print('t-intercept = ' + '%.4f' % t0 + ', slope = ' + '%.2f' % a, flush=True)
		start = time.time()
		# set the priors on the observables grid to zero
		for j in range(nrot): 
			for k in range(nmul): 
				pr_obs[j][k].fill(0)
		# get omega bin boundaries and age indices 
		if a == 0:
			ilo = ihi = np.argwhere(t_ar == t0)[0][0]
			bins = np.array([1, 0])
		else:
			# find the ages that are close to the line t = t_0 * (1 - a * omega)
			t10 = t0 - (a / np.log(10)) * np.array([1, 0]) # age at omega = 1 and 0
			ilo, ihi = np.searchsorted(t_ar, t10) # indices of ages at omega = 1 and 0
			if ilo > 0: ilo -= 1 # encompass another age, in case it's relevant
			bins = om_bins(t_ar[ilo : ihi + 1], t0, a) # bins of omega for each relevant age
		## put the prior for each relevant age onto the observables grid
		for it in range(ilo, ihi + 1):
			with open(filelist[it], 'rb') as f: 
				obs_binary, age, Mini, r, omega0, inc = pickle.load(f)
			# mask to choose omegas that are relevant at this age
			om1 = bins[it - ilo]
			om0 = bins[it - ilo + 1]
			m = (omega0 < om1) & (omega0 >= om0)
			omega0 = omega0[m]
			if len(omega0) > 0:
				obs_binary = obs_binary[:, :, m]
				# arrays of ordinate weights for the numerical integration in model space;
				# these include the varying discrete distances between adjacent abscissas;
				# dimensions: mass, r, omega, inclination
				w_Mini = du.trap(Mini)[:, np.newaxis, np.newaxis, np.newaxis]
				w_r = du.trap(r)[np.newaxis, :, np.newaxis, np.newaxis]
				w_inc = du.trap(inc)[np.newaxis, np.newaxis, np.newaxis, :]
				# mask the omegas
				w_omega0 = du.trap(omega0)
				# add the weights corresponding to the distances to the outer boundaries
				w_omega0[0] = w_omega0[0] + omega0[0] - om0
				w_omega0[-1] = w_omega0[-1] + om1 - omega0[-1]
				w_omega0 = w_omega0[np.newaxis, np.newaxis, :, np.newaxis]
				# non-uniform priors in non-omega model dimensions
				pr_Mini = (Mini**-2.35)[:, np.newaxis, np.newaxis, np.newaxis]
				pr_inc = np.sin(inc)[np.newaxis, np.newaxis, np.newaxis, :]
				# overall prior without the omega distribution: prior on r is flat
				pr0 = pr_Mini * pr_inc * (w_Mini * w_r * w_omega0 * w_inc)
				# omega distribution prior; 
				# dimensions: rotational population, omega
				pr_om = np.exp(-0.5*((omega0[np.newaxis, :] - cf.om_mean[:, np.newaxis]) \
							/ cf.om_sigma[:, np.newaxis])**2)
				# dimensions: rotational population, mass, r, omega, inclination
				pr_om = pr_om[:, np.newaxis, np.newaxis, :, np.newaxis]

				# index of each model in the data space grid;
				# dimensions: observable, mass, r, omega, inc
				ind = [] 
				for i in range(len(ld.nobs)): # for each data space dimension
					ind.append( np.searchsorted(ld.obs_grids[i], obs_binary[..., i], side='right') - 1 )
				ind = np.array(ind) # make the list into a numpy array
				ind = np.moveaxis(ind, 0, -1) # move the observable axis to the end
				# mask that chooses models where none of the observables are NAN 
				# and none of the observables map to -1 or len(observable grid dimension) - 1, 
				# i.e. outside the observable grid; dimensions: mass, r, omega, inc
				mask = np.all(ind != ld.nobs - 1, axis=-1) & np.all( ~np.isnan(obs_binary), axis=-1 )

				# rotational populations
				for j in range(len(cf.om_mean)):
					# prior on the model grid
					# dimensions: mass, r, omega, inc
					pr = pr0 * pr_om[j]
					# transfer the unary prior from the model grid to the grid of observables
					m = mask[:, 0]
					indices = tuple([ind[:, 0, ..., i][m] for i in range(ind.shape[-1])])
					np.add.at(pr_obs[j][0], indices, pr[:, 0][m])
					# transfer the binary prior from the model grid to the grid of observables
					indices = tuple([ind[..., i][mask] for i in range(ind.shape[-1])])
					np.add.at(pr_obs[j][1], indices, pr[mask])
		print('Putting the priors on the observables grid: ' + '%.2f' % (time.time() - start) + ' seconds.', flush=True)
		start = time.time()
		## package the prior density with the grids of observables for these t0 and a
		for j in range(nrot): # for each rotation
			rot_str = '_rot' + str(j)
			for k in range(nmul): # for each multiplicity
				mul_str = '_mul' + str(k)
				if np.count_nonzero(pr_obs[j][k]) == 0:
					points[it0, ia, k, j, i] = 0.0
				else:
					density = du.Grid(pr_obs[j][k], [x.copy() for x in ld.obs_grids], cf.ROI, cf.norm, Z=cf.Z, age=age)		
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
					# compute the CMD density
					density_cmd = density.copy()
					density_cmd.marginalize(2)
					# for data points where vsini is at the lower ROI boundary, convolve in vsini with the residual error kernel;
					# do not re-normalize after the convolution; integrate the probability beyond the lower boundary
					s = cf.std[-1] * np.sqrt(cf.v0err**2 - 1) # residual sigma = sqrt( sigma^2 - sigma_0^2 )
					kernel = du.Kernel(s / density.step[-1], nsig)
					density_v0 = density.copy()
					if density_v0.check_roi(-1, kernel): 
						density_v0.convolve(-1, kernel)
					density_v0.integrate_lower(2)

					# at the first density calculation, 
					# calculate residual kernels and corresponding slices for individual data points
					if kernels is None: kernels, slices = du.calc_kernels(density, sigma, nsig)

					# save the convolved priors; this takes up lots of memory, 
					# only do it if you want to plot these densities
					with open(dens_dir + 'density' + t0_str + a_str + rot_str + mul_str + '.pkl', 'wb') as f:
						pickle.dump([density, density_cmd, density_v0], f)

					## calculate the probability densities at data point locations
					max_dp = 0 # maximum absolute de-normalization
					start = time.time()
					for i in range(npts): # for each star
						# status w.r.t. the vsini measurement
						if np.isnan(ld.obs[i, -1]): density1 = density_cmd # sigma_vsini = infinity				
						elif ld.obs[i, -1] == -1: 	density1 = density_v0 # vsini = v_0 = 0
						else: 						density1 = density # vsini > v_0 
						# integration with the kernel, which is normalized up to the product of step sizes
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
						points[it0, ia, k, j, i] = float(dens * norm)
		print('Computing point densities: ' + '%.2f' % (time.time() - start) + ' seconds.', flush=True)
	# save the data point densities at these ages for these rotational population distributions; 
	# do this at every age, in case the program crashes; delete the previously saved file every time
	file = points_dir + 'points_os' + ('_'.join(['%.2f' % n for n in cf.om_sigma])).replace('.','') + \
		( '_t0' + '%.4f' % t0_ar[0] + '_' + '%.4f' % t0 ).replace('.','p') + '.pkl'
	with open(file, 'wb') as f: pickle.dump([points[:it0+1], t0_ar[:it0+1], cf.om_sigma], f)
	if it0 > 0: os.remove(prev_file)
	prev_file = file

		# gc.collect() # collect garbage / free up memory    
		# # look at the sizes of the largest variables
		# for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
		# 						 key= lambda x: -x[1])[:10]:
		# 	print("{:>30}: {:>8}".format(name, mu.sizeof_fmt(size)))