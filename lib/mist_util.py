import sys, os, time, warnings
sys.path.append(os.path.abspath(os.path.join('..', 'paint_atmospheres')))
from pa.lib import surface as sf
from pa.lib import util as ut
from pa.opt import grid as gd

import numpy as np
from scipy.interpolate import griddata
from matplotlib import pyplot as plt

# a set of MIST models
class Set:

	# load the MIST models
	def __init__(self, filename=None, age=None, Z=None):
		self.age = age
		self.Z = Z
		if filename is not None:
			# logZ oM0 EEP log10_isochrone_age_yr initial_mass star_mass log_L log_L_div_Ledd log_Teff\
			# log_R surf_avg_omega surf_r_equatorial_div_r surf_r_polar_div_r surf_avg_omega_div_omega_crit
			md = np.load(filename, allow_pickle=False) # this will hold all the model parameters
			# select only the variables of interest
			self.models = md[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 13]]
			self.set_vars()

	def copy(self):
		st = Set()
		st.models = np.copy(self.models)
		st.age = self.age
		st.Z = self.Z
		st.set_vars()
		return st

	# set model parameter arrays
	def set_vars(self):	
		self.logZm = self.models[:, 0] # logarithmic metallicity in MIST
		self.oM0 = self.models[:, 1] # initial omega_MESA
		self.EEP = self.models[:, 2] # eeps
		self.t = self.models[:, 3] # log age in years
		self.Mini = self.models[:, 4] # initial mass in solar masses
		self.M = self.models[:, 5] # mass in solar masses
		self.logL = self.models[:, 6] # log(luminosity) in solar luminosities
		self.logL_div_Ledd = self.models[:, 7] # log of the Eddington ratio
		self.R = 10**self.models[:, 8] # volume-averaged radius
		self.oM = self.models[:, 9] # observed Omega / Omega_c
		self.otc = self.oM * np.sqrt(1 - 10**self.logL_div_Ledd) # Omega / Omega_c, corrected for luminosity

	# clear parameter arrays
	def clear_vars(self):
		self.logZm = None
		self.oM0 = None
		self.EEP = None
		self.t = None
		self.Mini = None
		self.M = None
		self.logL = None
		self.logL_div_Ledd = None
		self.R = None
		self.oM = None
		self.otc = None	

	# select a set of models
	def select(self, mask):
		self.models = self.models[mask, :]
		self.set_vars()

	# select models with valid rotation
	def select_valid_rotation(self):
		m = (self.otc >= sf.omin) & (self.otc <= sf.omax) # mask out models with negative or super-critical rotation
		m = m & (self.oM0 < 0.8) # mask out models likely to be super-critical on the pre-main sequence
		self.select(m)

	def select_MS(self):
		m = (self.EEP >= 202) & (self.EEP < 454)
		self.select(m)

	def select_mass(self, Mmin=0, Mmax=np.inf):
		m = (self.Mini >= Mmin) & (self.Mini <= Mmax)
		self.select(m)

	def select_tZ(self, age, Z):
		m = (self.t == age) & (self.logZm == Z)
		self.select(m)
		self.age = age
		self.Z = Z

# a grid of MIST models at some age and metallicity
class Grid:
	# independent variables that define the grid, axes of the corresponding array of observables
	ivars = ['Mini', 'oM0', 'inc']
	# labels of the independent variables
	lvars = [r'M_{i}', r'\omega_{M,i}', r'i']
	# standard deviations of the observables
	std = None
	# distance modulus of the cluster
	modulus = None
	# PARS grid
	pars = None

	# Interpolates between MIST models in initial mass and MESA omega
	#	to find other model parameters; computes PARS omega and equatorial radii;
	# 	records a set of inclinations as well
	# Inputs:
	#	MIST model set at some age and metallicity
	#	initial masses
	# 	initial MESA omegas
	def interp(self):
		st = self.st
		Mogrid = tuple( np.meshgrid( self.Mini, self.oM0, sparse=True, indexing='ij' ) )
		# interpolate the model parameters; 
		# use the linear method because there are small discontinuities at a limited range of masses
		self.M = griddata( (st.Mini, st.oM0), st.M, Mogrid, method='linear' )
		self.L = 10**griddata( (st.Mini, st.oM0), st.logL, Mogrid, method='linear' )
		oM = griddata( (st.Mini, st.oM0), st.oM, Mogrid, method='linear' )
		logL_div_Ledd = griddata( (st.Mini, st.oM0), st.logL_div_Ledd, Mogrid, method='linear' )
		R = griddata( (st.Mini, st.oM0), st.R, Mogrid, method='linear' )
		# present-day omega_MESA, without the Eddington luminosity correction 
		otc = oM * np.sqrt(1 - 10**logL_div_Ledd)
		# mitigate round-off error due to interpolation
		otc[np.less(otc, sf.omin, where=~np.isnan(otc)) & ~np.isnan(otc)] = sf.omin
		otc[np.greater(otc, sf.omax, where=~np.isnan(otc)) & ~np.isnan(otc)] = sf.omax
		# PARS' omega and equatorial radius
		sh = otc.shape
		shf = np.prod(sh)
		omega = np.full(shf, np.nan)
		Req = np.full(shf, np.nan)
		nn = ~np.isnan(otc)
		nnf = nn.flatten()
		omega[nnf] = sf.omega(otc[nn])
		Req[nnf] = R.flatten()[nnf] * np.cbrt((4 * np.pi / 3) / sf.V(omega[nnf]))
		omega = omega.reshape(sh)
		Req = Req.reshape(sh)
		self.omega = omega
		self.Req = Req

	# clear the dependent model variables that are used for calculations
	def clear_vars(self):
		del self.M
		del self.L
		del self.omega
		del self.Req

	def __init__(self, \
			st=None, Mi=None, oM0=None, inc=None, A_V=None, verbose=False):
		# independent MIST model parameters
		self.Mini = Mi
		self.oM0 = oM0
		self.inc = inc
		self.A_V = A_V
		self.st = st # set of MIST models
		# interpolate to set the dependent parameters
		if st is not None:
			self.calc_obs(verbose=verbose)
		else:
			self.obs = None

	# return a version of the object for pickling;
	# this copy only has the independent star model variables, the observables, and the cluster variables;
	# it does not have the original MIST models.
	def pickle(self):
		grid = Grid(Mi=self.Mini, oM0=self.oM0, inc=self.inc, A_V=self.A_V)
		grid.obs = np.copy(self.obs).astype(np.float32)
		grid.age = self.st.age
		grid.Z = self.st.Z
		del grid.st
		return grid

	def calc_obs(self, verbose=False):
		if verbose:
			print('Calculating the observables for ' + str(len(self.Mini)) + ' x ' +\
				str(len(self.oM0)) + ' x ' + str(len(self.inc)) + ' = ' +\
				'{:,}'.format(len(self.Mini) * len(self.oM0) * len(self.inc)) + ' models...')
			start = time.time()
		self.interp() # calculate the dependent model variables
		# construct points for interpolating from the PARS grid;
		# 	each point is a set of values used by the PARS grid
		#	(e.g. [tau, omega, inclination, gamma, logZp, AV] 
		#	- all the dimensions of a PARS grid except the bands)
		# these points are on a grid of the independent variables 
		# 	(e.g. initial mass, initial omega_MESA),
		#	plus additional parameters for PARS that MESA models don't have 
		#	(e.g. inclination)
		tau = ut.tau(self.L, self.Req) # dimensions of the grid of MESA independent variables
		gamma = ut.gamma(self.M, self.Req) # dimensions of the grid of MESA independent variables
		points = np.full(self.M.shape + ( len(self.inc), len(self.pars.dims) - 1 ), np.nan)
		points[..., 0] = tau[..., np.newaxis] # add axes for non-MESA independent variables
		points[..., 1] = self.omega[..., np.newaxis] # add axes for non-MESA independent variables
		points[..., 2] = self.inc # if this is the only MESA-independent variable, no new axes necessary here
		points[..., 3] = gamma[..., np.newaxis] # add axes for non-MESA independent variables
		points[..., 4] = ut.logZp_from_logZm(self.st.Z) # same metallicity for each point
		points[..., 5] = self.A_V # same AV for each point
		sh = points.shape[:-1] # record the shape of all dimensions except last
		points = points.reshape( (-1, len(self.pars.dims) - 1) ) # flatten all dimensions except last
		# interpolate from the PARS grid;
		# the result is an an array of magnitudes, e.g. [F435W, F555W, F814W]
		mag = gd.interp(self.pars, points)
		# reshape magnitudes back to the grid of evolutionary model parameters
		mag = mag.reshape(sh + (len(self.pars.bands), ))
		# correct for radius and distance;
		# the radius array needs extra dimensions due to bands and PARS parameters
		# that evolutionary models don't have (e.g. inclination)
		mag = gd.correct(mag, self.Req[..., np.newaxis, np.newaxis], self.modulus)
		vsini = ut.vsini1(self.M[..., np.newaxis], self.Req[..., np.newaxis], \
			self.omega[..., np.newaxis], self.inc[np.newaxis, np.newaxis, :]) / 1e5
		self.obs = np.stack( (mag[..., 1], mag[..., 0] - mag[..., 2], vsini), axis=-1 )
		self.clear_vars() # clear the dependent model variables
		if verbose:
			print('\t' + str(time.time() - start) + ' seconds.')

	# Get the maximum (observable difference / std) in a focal model dimension 
	def get_maxdiff(self, axis):
		# absolute difference in sigmas along the axis
		diff = np.abs(np.diff(self.obs, axis=axis)) / self.std
		# move the focal model axis to the front		
		diff = np.moveaxis(diff, axis, 0) 
		# flatten all but the focal axis
		diff = diff.reshape(diff.shape[0], -1)
		# suppress the error for all-NAN slices
		warnings.filterwarnings('ignore') 
		# maximum difference across observables and non-focal model dimensions
		maxdiff = np.nanmax(diff, axis=1)
		# go back to default error reports
		warnings.filterwarnings('default')
		return maxdiff

	# Subdivide each interval into n subintervals, where n is the ceiling of the largest 
	# 	(observable difference / (std * dmin)), where dmin is a class variable
	# Inputs:
	#	model dimension to refine
	#	standard deviations of the observables
	#	refinement factor; make it smaller for a finer grid; default is 1.0
	def refine(self, axis, dmin=1.0):
		var = getattr(self, self.ivars[axis]) # get the model parameter list
		maxdiff = self.get_maxdiff(axis)
		# compute number of intervals to split each interval into
		notnan = ~np.isnan(maxdiff)
		ns = np.ones_like(maxdiff, dtype=int) 
		ns[notnan] = np.ceil(maxdiff[notnan] / dmin).astype(int)
		# split intervals: reverse so that indices remain valid
		for i in np.flip(np.where(ns > 1.)[0]): 
			# split the interval at each location and insert the result 
			var = np.insert(var, i+1, np.linspace(var[i], var[i+1], ns[i]+1)[1:-1])
		setattr(self, self.ivars[axis], var) # set the model parameter list
		self.calc_obs() # calculate the observables

	def plot_diff(self, axis, filename):
		label = self.lvars[axis]
		varname = self.ivars[axis]
		var = getattr(self, varname) # get the model parameter list
		# set it to midpoints between models
		var = (var[1:] + var[:-1]) / 2
		# difference with maximum modulus in sigmas along the axis
		maxdiff = self.get_maxdiff(axis)
		plt.scatter(var, maxdiff, s=2)
		plt.xlabel(r'$' + label + r'$')
		plt.ylabel(r'$\max{\left|\,\Delta x\left(' + label + r'\right) / \sigma_x\,\right|}$')
		plt.savefig(filename, dpi=200)
		plt.close()

	# Coarsen the model grid in a given focal dimension 
	def coarsen(self, axis, dmax=1.0):
		# print('Coarsening the model grid...')
		# start = time.time()
		obs = self.obs		
		obs = np.moveaxis(obs, axis, 0) # move the focal model axis to the front
		var = getattr(self, self.ivars[axis])
		maxdiff = self.get_maxdiff(axis)
		ind = np.argsort(maxdiff) # indices of sorted differences, NAN are at the end
		i = 0 # index in the above array of indices
		j = ind[i] # current index in the array of maximum differences
		stop = False
		# check for the stopping condition
		if np.isnan(maxdiff[j]): stop = True
		elif maxdiff[j] >= dmax: stop = True
		while not stop:
			# differences due to the merging of the focal interval with neighbors to the left or right
			maxright = np.nan
			maxleft = np.nan
			# if there is an interval to the right of the focal interval
			if j < len(maxdiff) - 1: 
				# subtract the observables at the lower bound of the corresponding interval
				# from those at the upper bound of the interval to the right; compute the maximum
				warnings.filterwarnings('ignore') # suppress the error for all-NAN slices
				maxright = np.nanmax( np.abs(obs[j+2,...] - obs[j,...]) / self.std )
				warnings.filterwarnings('default') # go back to default error reports
				# only consider the merge if the combined differences are below the maximum allowable difference
				if maxright > dmax: maxright = np.nan
			# if there is an interval to the left of the focal interval
			if j > 0:
				# get maximum differences due to the merging of the focal interval with the interval to the left
				warnings.filterwarnings('ignore') # suppress the error for all-NAN slices
				maxleft = np.nanmax( np.abs(obs[j+1,...] - obs[j-1,...]) / self.std )
				warnings.filterwarnings('default') # go back to default error reports
				if maxleft > dmax: maxleft = np.nan
			# determine whether to merge with the left neighbor, the right neighbor or not at all
			if ~np.isnan(maxleft):
				if ~np.isnan(maxright):
					if maxleft <= maxright:
						merge = 'left'
					else:
						merge = 'right'
				else:
					merge = 'left'
			elif ~np.isnan(maxright):
				merge = 'right'
			else:
				merge = None
			# if not merging, advance the index of indices
			if merge is None:
				i += 1
				if i < len(ind):
					j = ind[i]
				else:
					stop = True
			elif merge == 'left':
				## merge with the interval to the left: 
				# delete the left bound from the observables array, 
				obs = np.delete(obs, j, axis=0)
				# replace the maximum differences for the focal interval and the left neighbor 
				# with the combined maximum differences 
				maxdiff[j-1] = maxleft
				maxdiff = np.delete(maxdiff, j)
				# delete the model variable value at the left boundary of the focal interval
				var = np.delete(var, j)
			elif merge == 'right':
				## merge with the interval to the right: 
				# delete the right bound from the observables array, 
				obs = np.delete(obs, j+1, axis=0)
				# replace the maximum differences for the focal interval and the right neighbor 
				# with the combined maximum differences 
				maxdiff[j] = maxright
				maxdiff = np.delete(maxdiff, j+1)
				# delete the model variable value at the right boundary of the focal interval
				var = np.delete(var, j+1)
			if merge is not None:
				# re-set the sorted maximum differences
				ind = np.argsort(maxdiff)
				i = 0 
				j = ind[i]
			# check for the stopping condition
			if np.isnan(maxdiff[j]): stop = True
			elif maxdiff[j] >= dmax: stop = True

		self.obs = np.moveaxis(obs, 0, axis) # move the focal model axis back into its place
		setattr(self, self.ivars[axis], var) # set the model parameter list
		
		# print('\t' + str(time.time() - start) + ' seconds.')
				