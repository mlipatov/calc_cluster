import sys, os, time, pickle

from lib import dens_util as du
import config as cf
import load_data as ld
# Python imports
import numpy as np
import glob
from scipy.optimize import root
from scipy.special import erf
from scipy.ndimage import map_coordinates

# eps = np.finfo(float).eps
eps = 1e-300

nsig = cf.nsig - 1 # number of standard deviations to extend Gaussian kernels
npts = ld.obs.shape[0] # number of data points
ndim = ld.obs.shape[1] # number of observable dimensions

# load the data point densities
# dimensions: age, multiplicity population, rotational population, data point
filelist = list(np.sort(glob.glob(cf.points_dir + '*.pkl')))
t = None
for filepath in filelist:
	with open(filepath, 'rb') as f: 
		# pts1, t1, om_sigma = pickle.load(f)
		pts1, t1 = pickle.load(f)
		if t is None: # if there are no ages from before
			points = pts1; t = t1
		else:
			m = ~np.isin(t1, t) # where the new ages are not in exisiting ages
			points = np.concatenate((points, pts1[m]), axis=0)
			t = np.concatenate((t, t1[m]))

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
## a range of age priors
# smaller of the two distances between range boundaries and available age grid boundaries,
# divided by the number of standard deviations in half the Gaussian age prior;
# this is the largest possible standard deviation of this prior
sigma_max = np.minimum( cf.tmin - t[0], t[-1] - cf.tmax ) / nsig
# means of the priors
if t[-1] <= cf.tmin: # if the largest age is below the target range
	t_mean = np.array([t[-1]]) # compute for the largest age only
elif t[0] >= cf.tmax: # if the smallest age is above the target range
	t_mean = np.array([t[0]]) # compute for the smallest age only
else: # else, compute on the intersection of the target range and allowed range
	t_mean = np.linspace(np.maximum(cf.tmin, t[0]), np.minimum(cf.tmax, t[-1]), cf.tn, dtype=float)
# standard deviations of the priors
if sigma_max <= cf.smin: # if maximum sigma is below the target sigma range
	t_sigma = np.array([sigma_max]) # compute for the one value of maximum sigma
else: # else, compute on the intersection of the target range and allowed range
	t_sigma = np.linspace(cf.smin, np.minimum(sigma_max, cf.smax), cf.sn)

print( 'age means: ' + ', '.join(['%.4f' % t for t in t_mean]) )
print( 'age standard deviations: ' + ', '.join(['%.4f' % t for t in t_sigma]) )
print( 'data point age grid step: ' + '%.5f' % (t[1] - t[0]) )
print( 'means grid step: ' + '%.5f' % (t_mean[1] - t_mean[0]) )
print( 'deviations grid step: ' + '%.5f' % (t_sigma[1] - t_sigma[0]) )
print( 'slow rotator proportion step: ' + '%.5f' % (cf.w0[1] - cf.w0[0]) )
print( 'fast rotator proportion step: ' + '%.5f' % (cf.w1[1] - cf.w1[0]) )

# construct age priors, normalized up to the same factor
# dimensions: age, prior mean, prior sigma
t_pr = np.exp( -0.5 * (t[:, np.newaxis, np.newaxis] - t_mean[np.newaxis, :, np.newaxis])**2 \
						 / t_sigma[np.newaxis, np.newaxis, :]**2 )
t_pr /= np.sum(t_pr, axis=0)
# marginalize probability densities at data point locations by the age prior,
# correct by age step if the densities have to be normalized to 1; assume age step size is constant
# dimensions of output: age prior mean, age prior sigma, multiplicity, rotational population, data point 
ft = np.sum(t_pr[..., np.newaxis, np.newaxis, np.newaxis] * points[:, np.newaxis, np.newaxis, ...], axis=0) \
						/ (t[1] - t[0])

# q values and integration weights
q = np.linspace(0, 1, 101)
dq = q[1] - q[0]
wq = np.ones_like(q); wq[0] = wq[-1] = 1./2 # trapezoidal rule
wq = wq[:, np.newaxis]

# b values and integration weights
b = np.linspace(0, 1, 101)
db = b[1] - b[0]
wb = np.ones_like(b); wb[0] = wb[-1] = 1./2 # trapezoidal rule
wb = wb[np.newaxis, :]

# derivatives of log likelihood w.r.t. q and w.r.t. b at (q, b) = x
# likelihood is the product of (1 + Ai * q + Bi * q * b) over i
def dlogL(x, A, B):
	# keep q away from boundaries, where division by zero is possible
	if (x[0] >= 1.): q = 1 - eps
	elif (x[0] <= 0.): q = eps
	else: q = x[0]
	# keep b away from the zero boundary
	if (x[1] <= 0.): b = eps
	elif (x[1] >= 1. ): b = 1
	else: b = x[1]
	# derivatives with respect to q and b
	dlogL_dq = np.sum(1 / (q + 1 / (A + B * b)) )
	dlogL_db = np.sum(1 / (b + (1 / q + A) / B) )
	return [dlogL_dq, dlogL_db]

# likelihood as a function of q, b, A and B
def L(q, b, A, B):
	return 1 + (A + B * b) * q

## marginalize in q under uniform prior for combinations of population proportions and age prior parameters
# ll = np.full( (len(cf.w0), len(cf.w1), len(cf.b), len(t_mean), len(t_sigma)), np.nan )
ll = np.full( (len(cf.w0), len(cf.w1), len(t_mean), len(t_sigma)), np.nan )
qm = np.full_like( ll, np.nan ) # array of maximum-likelihood q
bm = np.full_like( ll, np.nan ) # array of maximum-likelihood b
count = 0 # count the age priors
for itm in range(len(t_mean)):
	for its in range(len(t_sigma)):
		start = time.time()
		f = ft[itm, its]
		for i in range(len(cf.w0)):
			w0 = cf.w0[i]
			j_range = np.searchsorted(cf.w1, 1 - w0, side='right')
			for j in range(j_range):
				w1 = cf.w1[j]
				# unary probability densities for all data points
				f0 = f[0, 0, :] * w0 + f[0, 1, :] * (1 - w0 - w1) + f[0, 2, :] * w1
				# binary probability densities for all data points
				f1 = f[1, 0, :] * w0 + f[1, 1, :] * (1 - w0 - w1) + f[1, 2, :] * w1
				# coefficients of q and qb in the likelihood factors (the remaining term is 1)
				A = f0 / back - 1
				B = (f1 - f0) / back
				sol = root(dlogL, [0.5, 0.5], args=(A, B))
				if sol.success:
					qmax, bmax = sol.x
				else:
					print('maximization of log-likelihood failed.')
				qm[i, j, itm, its] = qmax
				bm[i, j, itm, its] = bmax

				# nth root of maximum likelihood at x = (q, b), where n is the number of data points;
				# each factor in the likelihood expression could be so small that the product 
				# experiences underflow; thus, we take the root of each factor before taking the product
				nLmax = np.prod(np.power( L(qmax, bmax, A, B), 1. / npts ))
				# maximum log likelihood
				llmax = npts * np.log(nLmax)

				# likelihood factors on a grid of q, b, and data points, 
				# each divided by the nth root of maximum likelihood;
				lf = L( q[:, np.newaxis, np.newaxis], b[np.newaxis, :, np.newaxis], \
					    A[np.newaxis, np.newaxis, :], B[np.newaxis, np.newaxis, :]) / nLmax
				# likelihood vs q and b, divided by the maximum likelihood
				l = np.prod(lf, axis=-1)
				# logarithm of the integral, with logarithm of maximum likelihood added back
				ll[i, j, itm, its] = np.log(np.sum(l * wq * wb) * dq * db) + llmax
		count += 1
		print(str(count) + ' / ' + str(len(t_mean) * len(t_sigma)) + ' age priors... ' +\
												'%.2f' % (time.time() - start) + ' seconds.' )
# print('marginalization in q on a grid of w_0, w_1 and b: ' + '%.2f' % (time.time() - start) + ' seconds.')
w0i, w1i, ti, si = np.unravel_index(np.nanargmax(ll),ll.shape)
print('max ln likelihood: ' + str(np.nanmax(ll)) + ' at w_0 = ' + '%.4f' % cf.w0[w0i] + \
	', 1 - w_0 - w_1 = ' + '%.4f' % (1 - cf.w0[w0i] - cf.w1[w1i]) + ', w_1 = ' + '%.4f' % cf.w1[w1i] +\
	', t_mean = ' + '%.4f' % t_mean[ti] + ', t_sigma = ' + '%.4f' % t_sigma[si])
print('min ln likelihood: ' + str(np.nanmin(ll)))

# package the likelihoods, the ML q values and the rotational distribution standard deviations
with open('data/likelihoods/pkl/ll_' + \
		str(cf.Z).replace('-', 'm').replace('.', 'p') + \
		'_os' + '_'.join([('%.2f' % n).replace('.','') for n in cf.om_sigma]) + '.pkl', 'wb') as f:
		# '_os' + '_'.join([('%.2f' % n).replace('.','') for n in om_sigma]) + '.pkl', 'wb') as f:
    pickle.dump([ll, qm, bm, t_mean, t_sigma], f)    