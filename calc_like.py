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
back[mn] = 1 / cf.volume_cm 
# at the vsini = 0 boundary
back[m0] = cf.v0err * cf.std[-1] / (np.sqrt(2 * np.pi) * cf.volume) 
# everywhere else
back[mv] = ( 1 + erf(ld.obs[mv, -1] / (np.sqrt(2) * ld.err[mv, -1])) ) / (2 * cf.volume)

## compute likelihoods on a grid of age, metallicity, 
## rotational population and multiplicity population proportions
# the sum of the two rotational population will be constrained to <=1
w0 = np.linspace(0, 1, 21, dtype=float) # proportion of the zero rotational population
w1 = np.linspace(0, 1, 21, dtype=float) # proportion of the maximum rotational population
b = np.linspace(0, 1, 21, dtype=float) # proportion of the binaries population
# load the data point densities
# dimensions: age, multiplicity population, rotational population, data point
with open('data/points.pkl', 'rb') as f: points, t = pickle.load(f)
## come up with a range of age prior parameters
# let the middle half of the available age range allow for the variation in the mean
t_mean = t[len(t)/4 : 3*len(t)/4]
# let the outer quarters of the age range allow for the variation in the standard deviation
t_std = np.linspace(0, (t[-1]-t[0])/(4*nsig), len(t_mean)+1)[1:]

## marginalize in q under uniform prior for combinations of population proportions and age prior parameters
ll = np.full( (len(w0), len(w1), len(b)), np.nan )
qm = np.full_like( ll, np.nan ) # array of maximum-likelihood q
start = time.time()
for i in range(len(w0)):
	wi0 = w0[i]
	for j in range(len(w1) - i):
		wi1 = w1[j]
		for k in range(len(b)):
			bi = b[k]
			for itm in range(len(t_mean)):
				for its in range(len(t_std)):
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
					
print('marginalization in q on a grid of w_0, w_1 and b: ' + '%.2f' % (time.time() - start) + ' seconds.')
i0, i1, i2 = np.unravel_index(np.nanargmax(ll),ll.shape)
print('max ln likelihood: ' + str(np.nanmax(ll)) + ' at w_0 = ' + str(w0[i0])[:4] + \
	', 1 - w_0 - w_1 = ' + str(1 - w0[i0] - w1[i1]) + ', w_1 = ' + str(w1[i1])[:4] +\
	' and b = ' + str(b[i2])[:4])
print('min ln likelihood: ' + str(np.nanmin(ll)))

# package the likelihoods, the ML q values and the rotational distribution standard deviations
with open('data/likelihoods/pkl/ll_' + \
		str(cf.Z).replace('-', 'm').replace('.', 'p') + \
		'_os' + '_'.join([('%.2f' % n).replace('.','') for n in om_sigma]) + '.pkl', 'wb') as f:
    pickle.dump([ll, qm, densities[-2], densities[-1]], f)    