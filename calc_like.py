import sys, os, time, pickle

from lib import dens_util as du
import config as cf
import load_data as ld

import numpy as np
import glob
from scipy.optimize import root_scalar
from scipy.special import erf
from scipy.ndimage import map_coordinates

eps = np.finfo(float).eps

nsig = cf.nsig - 1 # number of standard deviations to extend Gaussian kernels
npts = ld.obs.shape[0] # number of data points
ndim = ld.obs.shape[1] # number of observable dimensions

# background probability density
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

## compute likelihoods on a grid of age, metallicity, 
## rotational population and multiplicity population proportions
# load the data point densities
# dimensions: age, multiplicity population, rotational population, data point
points = []
t = np.array([])
filelist = list(np.sort(glob.glob('data/points/*.pkl')))
for filepath in filelist:
	with open(filepath, 'rb') as f: 
		pts1, t1 = pickle.load(f)
		points.append(pts1)
		t = np.concatenate((t, t1))
t, indices = np.unique(t, return_index=True)
points = np.concatenate(points)
points = points[indices]
## a range of age priors
# smaller of the two distances between range boundaries and available age grid boundaries,
# divided by the number of standard deviations in half the Gaussian age prior;
# this is the largest possible standard deviation of this prior
std_max = np.minimum( cf.tm[0] - t[0], t[-1] - cf.tm[-1] ) / nsig
# means and standard deviations of the priors; 
# step size should be comparable to that in the age grids
n_mean = int((cf.tm[-1] - cf.tm[0]) / (t[1] - t[0]) + 2) * 3
n_std = int((cf.ts[-1] - cf.ts[0]) / (t[1] - t[0]) + 2) * 3
t_mean = np.linspace(cf.tm[0], cf.tm[-1], n_mean)
t_std = np.linspace(cf.ts[0], np.minimum(std_max, cf.ts[-1]), n_std)

# # proportion of the available age range that allows for variation in the mean
# p = 1./20
# # let the middle of the available age range allow for the variation in the mean
# t_mean = t[int(np.ceil(len(t)*(1-p)/2)) : int(np.floor(len(t)*(1+p)/2)) + 1]
# # let the remaining outer parts of the age range allow for the variation in the standard deviation
# t_std = np.linspace(0, (1-p)*(t[-1]-t[0]) / (2*nsig), len(t_mean)+1)[1:]

# t_mean = np.array([t[int(len(t)/2)]])
# t_std = np.linspace(0, (t[-1]-t[0]) / (2*nsig), 20)[1:]
print( 'age means: ' + ', '.join(['%.4f' % t for t in t_mean]) )
print( 'age standard deviations: ' + ', '.join(['%.4f' % t for t in t_std]) )
print( 'data point age grid step: ' + '%.5f' % (t[1] - t[0]) )
print( 'means grid step: ' + '%.5f' % (t_mean[1] - t_mean[0]) )
print( 'deviations grid step: ' + '%.5f' % (t_std[1] - t_std[0]) )

# construct age priors, normalized up to the same factor
# dimensions: age, prior mean, prior sigma
t_pr = np.exp( -0.5 * (t[:, np.newaxis, np.newaxis] - t_mean[np.newaxis, :, np.newaxis])**2 \
						 / t_std[np.newaxis, np.newaxis, :]**2 )
t_pr /= np.sum(t_pr, axis=0)
# marginalize probability densities at data point locations by the age prior,
# correct by age step if the densities have to be normalized to 1; assume age step size is constant
# dimensions of output: age prior mean, age prior sigma, multiplicity, rotational population, data point 
ft = np.sum(t_pr[..., np.newaxis, np.newaxis, np.newaxis] * points[:, np.newaxis, np.newaxis, ...], axis=0) \
						/ (t[1] - t[0])

# q values and integration weights
q = np.linspace(0, 1, 201)
dq = q[1] - q[0]
w = np.ones_like(q); w[0] = w[-1] = 1./2 # trapezoidal rule

# derivative of log likelihood w.r.t. q at q = x
def dlogL(x, qi):
	if (x == 1.): q = 1 - eps
	elif (x == 0.): q = eps
	else: q = x	  
	return np.sum(1 / (q - qi))

## marginalize in q under uniform prior for combinations of population proportions and age prior parameters
ll = np.full( (len(cf.w0), len(cf.w1), len(cf.b), len(t_mean), len(t_std)), np.nan )
qm = np.full_like( ll, np.nan ) # array of maximum-likelihood q
for itm in range(len(t_mean)):
	for its in range(len(t_std)):
		start = time.time()
		f = ft[itm, its]
		for i in range(len(cf.w0)):
			wi0 = cf.w0[i]
			j_range = np.searchsorted(cf.w1, 1 - wi0, side='right')
			for j in range(j_range):
				wi1 = cf.w1[j]
				for k in range(len(cf.b)):
					bi = cf.b[k]
					# cluster model density, as a weighted sum of rotational and multiplicity population densities
					fc = (1 - bi) * ( f[0, 0, :] * wi0 + f[0, 1, :] * (1 - wi0 - wi1) + f[0, 2, :] * wi1 ) + \
							   bi * ( f[1, 0, :] * wi0 + f[1, 1, :] * (1 - wi0 - wi1) + f[1, 2, :] * wi1 )
					r = fc / back # ratio of cluster model density to background density
					qi = 1 / (1 - r) # zeros of the likelihood
					# maximum of the likelihood w.r.t. q
					if (dlogL(0, qi) <= 0) and (dlogL(1, qi) <= 0): qmax = 0
					elif (dlogL(0, qi) >= 0) and (dlogL(1, qi) >= 0): qmax = 1
					else: qmax = root_scalar(dlogL, args=(qi,), bracket=[0., 1.]).root
					qm[i, j, k, itm, its] = qmax
					# the nth root of maximum likelihood, where n is the number of data points
					nLmax = np.prod(np.power(1 - qmax / qi, 1. / npts))
					# maximum log likelihood
					llmax = npts * np.log(nLmax)
					# likelihood factors on a grid of q and data points, 
					# each divided by the nth root of maximum likelihood
					lf = (1 - q[:, np.newaxis] / qi[np.newaxis, :]) / nLmax
					# likelihood vs q, divided by the maximum likelihood
					l = np.prod(lf, axis=-1)
					# logarithm of the integral, with logarithm of maximum likelihood added back
					ll[i, j, k, itm, its] = np.log(np.sum(l * w) * dq) + llmax
		print(str(itm*len(t_mean) + (its + 1)) + ' / ' + str(len(t_mean) * len(t_std)) + ' age priors... ' +\
												'%.2f' % (time.time() - start) + ' seconds.' )
# print('marginalization in q on a grid of w_0, w_1 and b: ' + '%.2f' % (time.time() - start) + ' seconds.')
i0, i1, i2, i3, i4 = np.unravel_index(np.nanargmax(ll),ll.shape)
print('max ln likelihood: ' + str(np.nanmax(ll)) + ' at w_0 = ' + '%.4f' % cf.w0[i0] + \
	', 1 - w_0 - w_1 = ' + '%.4f' % (1 - cf.w0[i0] - cf.w1[i1]) + ', w_1 = ' + '%.4f' % cf.w1[i1] +\
	', b = ' + '%.4f' % cf.b[i2] + ', t_mean = ' + '%.4f' % t_mean[i3] + ', t_sigma = ' + '%.4f' % t_std[i4])
print('min ln likelihood: ' + str(np.nanmin(ll)))

# package the likelihoods, the ML q values and the rotational distribution standard deviations
with open('data/likelihoods/pkl/ll_' + \
		str(cf.Z).replace('-', 'm').replace('.', 'p') + \
		'_os' + '_'.join([('%.2f' % n).replace('.','') for n in cf.om_sigma]) + '.pkl', 'wb') as f:
    pickle.dump([ll, qm, t_mean, t_std], f)    