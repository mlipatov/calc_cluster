# Utilities needed to convolve, downsample, normalize and evaluate probability densities on grids
# of observables
import config as cf
import numpy as np
from scipy.interpolate import interp1d
import copy

class ConvolutionException(Exception):
    pass

# a finite symmetric Gaussian probability density kernel on a discrete, evenly spaced grid
class Kernel:
	# Inputs: 
	#	standard deviation of the kernel, in units of step size
	#	lower bound on the extent of downsampled kernel, in standard deviations
	#	downsample factor that will be used when convolving with this kernel
	def __init__(self, sigma, nsig, ds:int=1):
		# number of steps in one half of the kernel
		n = np.ceil(nsig * sigma / ds).astype(int) * ds
		x = np.linspace(-n, n, 2 * n + 1) # abscissae
		y = np.exp(-0.5 * (x/sigma)**2)
		self.y = y / np.sum(y) # normalize the kernel
		self.n = n

# probability density on a discrete, evenly spaced grid of observables;
# all integration (including marginalization, convolution and normalization) 
#	assumes unit discrete steps;
# scale (divide) by the product of grid steps to get meaningful probability densities;
# two types of dimensions
# 	1. normalized over a finite region of interest (ROI)
#	2. normalized over all reals, then collected at the boundaries of an ROI
class Grid:
	# Inputs:
	# 	density on a grid of observables, an n-dimensional array
	# 	1D grids of observables, a list of n arrays
	# 	finite region of interest, a list of two-element lists
	#	list of the kind of each observable: 
	#		True for normalized over ROI, False for collected at the ROI boundaries
	#	list of two-element lists - boundary bin sizes for each dimension
	# 	age, metallicity
	def __init__(self, dens, obs, ROI, normalized, boundary_bins, boundary_errs, age, Z):
		self.dens = dens 
		self.obs = obs 
		self.dim = len(obs) # number of dimensions, e.g. 3
		# discrete step in each dimension
		step = []
		for i in range(len(obs)):
			step.append( obs[i][1] - obs[i][0] )
		self.step = np.array(step)
		self.ROI = ROI
		self.normalized = normalized
		self.boundary_bins = boundary_bins
		self.boundary_errs = boundary_errs
		# a list of objects for each dimension:
		#	when a dimension is finite-normalized, the object is the dependence of de-normalization on
		#		the standard deviation of the convolving kernel (a spline function);
		# 	when a dimension is collected at finite boundaries, the object is a two-element list of 
		#		probability densities over the remaining dimensions, one for each boundary.
		correction = []
		for n in normalized:
			if n: correction.append(None)
			else: correction.append([None] * 2)
		self.correction = correction
		# # dependence of probability change within the ROI vs sigma for each dimension
		# self.dP = [None] * len(obs) 
		# # probabilities beyond each boundary of the ROI in vsini dimension across the non-vsini dimensions,
		# # for a fixed sigma at each boundary; whenever the density is normalized in non-vsini dimensions,
		# # these should be normalized by the same factor; whenever the density is marginalized in a 
		# # non-vsini dimension, this should also be marginalized; same with the scaling of density;
		# # a correction due to these probabilities will only be applied when a data point falls on the 
		# # corresponding boundary. 
		# self.Pvsini = [None] * 2

		self.age = age
		self.Z = Z

	def copy(self):
		dens = np.copy(self.dens) 
		obs = []
		for o in self.obs:
			obs.append(np.copy(o))
		ROI = copy.deepcopy(self.ROI)
		normalized = copy.deepcopy(self.normalized)
		boundary_bins = copy.deepcopy(self.boundary_bins)
		boundary_errs = copy.deepcopy(self.boundary_errs)
		density = Grid(dens, obs, ROI, normalized, boundary_bins, boundary_errs, self.age, self.Z)
		density.correction = copy.deepcopy(self.correction)
		# density.dP = copy.deepcopy(self.dP)
		# density.Pvsini = copy.deepcopy(self.Pvsini)
		return density

	def remove_axis(self, axis):
		self.obs.pop(axis)
		self.normalized.pop(axis)
		self.boundary_bins.pop(axis)
		self.boundary_errs.pop(axis)
		self.correction.pop(axis)
		self.ROI = np.delete(self.ROI, axis, axis=0)
		self.step = np.delete(self.step, axis)
		self.dim -= 1

	# obtain the dependence of probability leakage on standard deviations of the Gaussian kernels
	# convolve the density with a number of Gaussian kernels on the grid of observables without downsampling
	# Inputs:
	# 	number of standard deviations to extend Gaussian kernels
	# Outcome:
	#	for normalized dimensions, updates the dependence of de-normalization on kernel standard deviation
	# Notes:
	#	operates on normalized, un-scaled probability density
	def dP_sigma(self, nsig):
		# standard deviations in units of grid step size, up to the size
		# that can have half a kernel fit at an edge of the ROI 
		s = np.linspace(0.5, cf.conv_err, 9)
		kernels = [Kernel(s[j], nsig) for j in range(s.shape[0])]
		sigma = np.outer(self.step, s)	
		for i in range(len(self.obs)): # for each observable dimension
			if self.normalized[i]:
				dp = []
				for j in range(len(kernels)): # for each kernel width
					kernel = kernels[j]
					d = self.copy()
					# check that the kernel, evaluated at the ROI boundaries, fits within the grid
					if d.check_roi(i, kernel): 
						d.convolve(i, kernel)
						d.integrate_all()
						dp.append(d.dens - 1)
				# the dependence of log probability change on log sigma, in units of the observable
				dp = np.array(dp)
				x = sigma[i][:dp.shape[0]]
				fit = interp1d(x, dp, kind='cubic', fill_value='extrapolate')
				# if maximum probability change is less than the cube root of precision (< 10^-5 for regular floats)
				if -np.log10(np.abs(dp).max()) > np.finfo(float).precision / 3:  
					self.correction[i] = None
				else:
					self.correction[i] = fit

	# obtain the total probability beyond a boundary of the ROI
	# after convolution with an error kernel
	# Inputs:
	#	number of standard deviations of error kernel
	# 	index of the boundary: 0 or 1
	#	standard deviation of the kernel
	# Outcome:
	#	probability density arrays across all but the current dimension are set
	def P_collected(self, nsig):
		for axis in range(len(self.normalized)):
			if not self.normalized[axis]:
				for bound in [0, 1]:
					# standard deviation in units of grid step size
					s = self.boundary_errs[axis][bound] / self.step[axis]
					kernel = Kernel(s, nsig)
					d = self.copy()
					if d.check_roi(axis, kernel):
						d.convolve(axis, kernel)
						# integrate on the region beyond the boundary
						if bound == 0:
							region = [-np.inf, self.ROI[axis][0]]
						else:
							region = [self.ROI[axis][1], np.inf]
						d.integrate(region, axis)
						self.correction[axis][bound] = np.expand_dims(d.dens, axis)

	# check if the density can be convolved in some dimension with a symmetric kernel
	# so that the calculated points cover the region of interest
	# Inputs:
	#	axis to convolve
	#	convolving kernel
	#	region of the focal observable dimension that should be calculated
	def check_roi(self, axis, kernel):
		obs = self.obs[axis]
		index = np.nonzero( (obs >= self.ROI[axis][0]) & (obs <= self.ROI[axis][1]) )[0]
		# see if the number of steps in half the kernel left of the left boundary of the region
		# drops below smallest index (zero) and similarly for the largest index
		return (index.min() >= kernel.n) & (len(obs) - index.max() > kernel.n) 

	# convolve and downsample at the same time
	# Inputs: 
	#	axis along which to convolve
	# 	a kernel to convolve with
	#	downsample factor (an integer)
	# Notes:
	# 	number of steps in one half of the kernel must be a multiple of the downsample factor 
	def convolve(self, axis, kernel, ds:int=1):
		dens = np.moveaxis(self.dens, axis, -1) # move the focal axis to the front
		# shape of the result grid, downsampled in the focal dimension
		shape = list(dens.shape)
		shape[-1] = shape[-1] // ds 
		res = np.zeros(shape) # initialize the result grid
		# number of steps in one half of the kernel, downsampled
		j_lim = kernel.n // ds 
		# number of calculated downsampled points
		n_max = shape[-1] - 2*j_lim 
		# convolve
		for j in range(kernel.n * 2 + 1):
		    res[..., j_lim:-j_lim] += kernel.y[j] * dens[...,j::ds][...,:n_max]
		# remove the strip where the convolution wasn't computed
		res = res[...,j_lim:-j_lim]
		# move the focal axis back in its place
		self.dens = np.moveaxis(res, -1, axis)
		# remove the strip where the convolution wasn't computed from the focal observable's grid, 
		# then downsample the grid and limit it to computed points only
		self.obs[axis] = self.obs[axis][kernel.n:-kernel.n][::ds][:n_max]
		# multiply step size in the focal dimension by the downsample factor
		self.step[axis] *= ds

	# weights on a 1D region
	def w(self, region, axis):
		index = np.nonzero( (self.obs[axis] >= region[0]) & (self.obs[axis] <= region[1]) )[0]
		w = np.zeros_like( self.obs[axis] )
		w[index] = 1.
		w[index.min()] = w[index.max()] = 1./2
		return w

	# integral on a 1D region
	def integrate(self, region, axis):
		w = self.w(region, axis)
		# if this is a normalized dimension, 
		# marginalize it in the probabilities beyond the ROI boundaries for each collected dimension
		if self.normalized[axis]:
			for i in range(len(self.normalized)):
				if not self.normalized[i]:
					cor = self.correction[i]
					for j in range(len(cor)):
						self.correction[i][j] = np.sum(w * np.moveaxis(cor[j], axis, -1), axis=-1)
		# move the focal axis to the front
		dens = np.moveaxis(self.dens, axis, -1) 
		# integrate density
		self.dens = np.sum(w * dens, axis=-1)
		self.remove_axis(axis)

	def marginalize(self, axis):
		self.integrate([-np.inf, np.inf], axis)

	def multiply_collected(self, factor):
		for i in range(len(self.normalized)):
			if not self.normalized[i]: 
				self.correction[i][0] *= factor
				self.correction[i][0] *= factor		

	def integrate_all(self):
		# integrate
		while self.dim > 0:
			if self.normalized[-1]: region = self.ROI[-1]
			else: region = [-np.inf, np.inf]
			self.integrate(region, -1)

	# normalize
	def normalize(self):
		d = self.copy()
		# integrate
		d.integrate_all()
		norm = d.dens
		self.dens /= norm
		# correct the probabilities beyond the vsini ROI boundaries by the normalization factor
		self.multiply_collected(1 / norm)

		# for i in range(region.shape[0]):
		# 	w = self.w(region[i], i)
		# 	if i == 0:
		# 		weights = w
		# 	else:
		# 		weights = np.multiply.outer(weights, w)
		# return np.sum(self.dens * self.weights(region))

	# divide by the product of grid steps to get properly scaled probability
	def scale(self):
		sc = np.prod(self.step)
		self.dens /= sc
		# correct the probabilities beyond the vsini ROI boundaries by the scale factor
		self.multiply_collected(1 / sc)

	# integrate the product of a Gaussian kernel centered on a data point and 
	# probability density in some dimension; if the data point is at an ROI boundary
	# in the focal dimension, add the probability that acummulates at that boundary
	# Inputs:
	#	dimension of integration
	#	standard deviation of the kernel in the units of the focal observable
	#	center of the kernel, also the value of the observable at the data point
	# Output: probability density on a grid without the focal dimension
	# Notes: if both the density and the kernel are normalized, 
	#		the result is an evaluation of a normalized probability distribution, 
	#		except for de-normalization due to the implied convolution; 
	#	additionally, either the input density should be scaled after normalization,
	#		or the output should be scaled.
	def integrate_kernel(self, axis, sigma, nsig, point):
		dens = np.moveaxis(self.dens, axis, -1) # move the focal axis to the front
		obs = self.obs[axis] # get the focal observable
		step = self.step[axis] # the step in the focal dimension
		# if the residual standard deviation is small in comparison with grid step size
		if (sigma / step < 1./2):
			# interpolate
			dens = interp1d( obs, dens, axis=-1 )(point) 
		else: 
			# weights for integrating 
			w = np.ones_like(obs); w[0] = w[-1] = 1./2 
			# the kernel around the data point
			x = (obs - point) / sigma
			kernel = np.exp(-x**2 / 2.)
			m = np.abs(x) > nsig # observable grid outside sigma cutoff for symmetry and consistency
			# check that the kernel fits within the available observable grid
			if ~(m[0] & m[-1]): # if either grid boundary is inside the kernel
				raise ConvolutionException(\
					'One of the grid boundaries (' + str(obs[0]) + ', ' + str(obs[-1]) + \
					') is inside the kernel that extends to ' + str(nsig) + ' sigmas (' + str(sigma) + \
					') from its center at ' + str(point) )
			kernel[m] = 0 
			kernel /= np.sum(kernel) # normalize the kernel
			# integrate
			dens = np.sum(w * kernel * dens, axis=-1)
		# if the focal dimension is collected and the point is at a boundary in this dimension,
		# correct the resulting probability density due to the probability that acummulates
		# in the boundary bin
		if not self.normalized[axis]:
			for j in [0, 1]:
				if point == self.ROI[axis][j]:
					dens += np.squeeze(self.correction[axis], axis=axis) / self.boundary_bins[axis][j]	
		self.dens = dens
		self.remove_axis(axis)