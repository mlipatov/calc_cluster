## ---> this takes too much memory; probably have to calculate the right metallicities directly
# convert magnitudes on a grid of Kurucz's metallicities to a grid of MIST metallicities
import mist_util as mu
from pa.lib import util as ut

import time, pickle
import numpy as np
from scipy.interpolate import interp1d

iodir = '../data/'

# Load MIST models
print('Loading MIST...')
st = mu.Set(iodir + 'mist_grid.npy')
# get the PARS metallicities corresponding to the MIST ones
logZp = np.unique(ut.logZp_from_logZm(st.logZm)) 

# Load the original PARS grid
print('Loading PARS...')
start = time.time()
with open(iodir + 'pars_grid.pkl', 'rb') as f: 
	pars = pickle.load(f)
print('\t' + str(time.time() - start) + ' seconds.\n')

# cull the metallicity values from MIST according to PARS metallicity range
logZp = logZp[(logZp <= pars.Z.max()) & (logZp >= pars.Z.min())]

# interpolate
print('Interpolating on the metallicity grid...')
start = time.time()
Mag2 = np.empty_like(pars.Mag)
for i in range(len(pars.omega)):
	mag = np.take(pars.Mag, i, axis=1)
	f = interp1d(pars.Z, mag, kind='linear', axis=3)
	Mag2[:, i, ...] = f(logZp)
	# f = interp1d(pars.Z, 10**(-mag/2.5), kind='linear', axis=3)
	# Mag2[:, i, ...] = -2.5*np.log10(f(logZp))
pars.Mag = Mag2
pars.Z = logZp
print('\t' + str(time.time() - start) + ' seconds.\n')

### Pickle the grid
with open(iodir + 'pars_grid_2.pkl', 'wb') as f:
	pickle.dump(pars, f)