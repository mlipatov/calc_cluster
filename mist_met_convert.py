# interpolate magnitudes from a PARS grid with many metallicities 
# to a grid corresponding to a single MIST metallicity
from lib import mist_util as mu
import config as cf
from pa.lib import util as ut

import time, pickle
import numpy as np
from scipy.interpolate import interp1d

iodir = './data/'

## this code is for interpolating at one specific MIST metallicity value on the PARS grid

# Load the original PARS grid
print('Loading PARS...')
start = time.time()
with open(iodir + 'pars_grid.pkl', 'rb') as f: 
	pars = pickle.load(f)
print('\t' + str(time.time() - start) + ' seconds.\n')

# construct the PARS metallicity array from the one MIST metallicity
logZp = np.array([ ut.logZp_from_logZm(cf.Z) ])

# interpolate
print('Interpolating on the metallicity grid...')
start = time.time()
sh = list(pars.Mag.shape)
sh[4] = len(logZp) # change the metallicity dimension in the shape of the PARS grid
Mag2 = np.empty(sh)
for i in range(len(pars.omega)):
	mag = np.take(pars.Mag, i, axis=1)
	f = interp1d(pars.Z, mag, kind='linear', axis=3)
	Mag2[:, i, ...] = f(logZp)
pars.Mag = Mag2
pars.Z = logZp
print('\t' + str(time.time() - start) + ' seconds.\n')

### Pickle the grid
with open(iodir + 'pars_grid_ZM' + str(cf.Z).replace('-', 'm').replace('.', 'p') + '.pkl', 'wb') as f:
	pickle.dump(pars, f)

## the code below is for interpolating at the MIST metallicity values on the PARS grid

# # Load MIST models
# print('Loading MIST...')
# st = mu.Set(iodir + 'mist_grid.npy')
# # get the PARS metallicities corresponding to the MIST ones
# logZm = np.unique(st.logZm)
# logZp = ut.logZp_from_logZm(logZm) 

# # Load the original PARS grid
# print('Loading PARS...')
# start = time.time()
# with open(iodir + 'pars_grid.pkl', 'rb') as f: 
# 	pars = pickle.load(f)
# print('\t' + str(time.time() - start) + ' seconds.\n')

# # cull the metallicity values from MIST according to PARS metallicity range
# logZp = logZp[(logZp <= pars.Z.max()) & (logZp >= pars.Z.min())]

# # interpolate
# print('Interpolating on the metallicity grid...')
# start = time.time()
# Mag2 = np.empty_like(pars.Mag)
# for i in range(len(pars.omega)):
# 	mag = np.take(pars.Mag, i, axis=1)
# 	f = interp1d(pars.Z, mag, kind='linear', axis=3)
# 	Mag2[:, i, ...] = f(logZp)
# 	# f = interp1d(pars.Z, 10**(-mag/2.5), kind='linear', axis=3)
# 	# Mag2[:, i, ...] = -2.5*np.log10(f(logZp))
# pars.Mag = Mag2
# pars.Z = logZp
# print('\t' + str(time.time() - start) + ' seconds.\n')

# ### Pickle the grid
# with open(iodir + 'pars_grid_2.pkl', 'wb') as f:
# 	pickle.dump(pars, f)