import sys, os, time, pickle
sys.path.append(os.path.abspath(os.path.join('..', 'paint_atmospheres')))
from pa.lib import surface as sf

from lib import mist_util as mu
import config as cf
from lib import load_data as ld

import numpy as np

# calculate the minimum mass cut-off given the boundaries of the observable space region
# where we need model priors
def Mlim(st, A_V):
	Mi = np.sort(np.unique(st.Mini)) # original mass grid
	omega0 = np.array([st.omega0.min(), st.omega0.max()]) 
	inc = np.array([0, np.pi/2]) # the two extreme inclinations
	grid = mu.Grid(st, Mi, omega0, inc, cf.A_V)
	notnan = ~np.isnan(grid.obs)
	m1 = np.full_like(grid.obs, False, dtype=bool)
	m2 = np.full_like(grid.obs, False, dtype=bool)
	np.greater_equal(grid.obs, ld.obmin[np.newaxis, np.newaxis, np.newaxis,:], where=notnan, out=m1) 
	np.less_equal(grid.obs, ld.obmax[np.newaxis, np.newaxis, np.newaxis,:], where=notnan, out=m2)
	j = np.nonzero( np.all(m1 & m2, axis=-1) )[0]
	if len(j) > 0:
		Mmin = grid.Mini[j].min()
		Mmax = grid.Mini[j].max()
	else:
		Mmin = np.nan
		Mmax = np.nan
	return [Mmin, Mmax]

def rc_mass_omega(st, A_V, dmax):
	Mi = np.sort(np.unique(st.Mini)) # original mass grid
	o0 = np.sort(np.unique(st.omega0)) # original omega grid
	inc = np.array([0, np.pi/2]) # relatively small inclination grid
	grid = mu.Grid(st, Mi, o0, inc, A_V)

	omax = np.nanmax(grid.get_maxdiff(1))
	mmax = np.nanmax(grid.get_maxdiff(0))
	while omax > dmax or mmax > dmax:
		while omax > dmax:
			grid.refine(1, dmin=dmax)
			omax = np.nanmax(grid.get_maxdiff(1))
		grid.coarsen(1, dmax=dmax)
		mmax = np.nanmax(grid.get_maxdiff(0))
		while mmax > dmax:
			grid.refine(0, dmin=dmax)
			mmax = np.nanmax(grid.get_maxdiff(0))
		grid.coarsen(0, dmax=dmax)
		omax = np.nanmax(grid.get_maxdiff(1))
	return [grid.Mini, grid.omega0]

# refine/coarsen the inclination dimension
def rc_inc(st, A_V, dmax, dmin):
	Mi = np.sort(np.unique(st.Mini)) # original mass grid
	o0 = np.sort(np.unique(st.omega0)) # original omega grid
	inc = np.linspace(0, np.pi/2, 20) # larger inclination grid
	grid = mu.Grid(st, Mi, o0, inc, A_V)
	grid.refine(2, dmin=dmin)
	grid.coarsen(2, dmax=dmax)
	return grid.inc 

# pre-compute Roche model volume versus PARS's omega
# and PARS's omega versus MESA's omega
sf.calcVA()
sf.calcom()

# Load and filter MIST models
print('Loading MIST...')
start = time.time()
st = mu.Set('data/mist_grid.npy') # load

# recalculate the PARS grid in the metallicities we have in the MIST models
# from pa.lib import util as ut
# np.unique(ut.logZp_from_logZm(st.logZm))

# valid rotation
st.select_valid_rotation()
# on the MS
st.select_MS()
# a range of ages and metallicities
t = np.unique(st.t)[99:99+17] # all ages between 1 and 2 Gyr; index 8 has 9.154
Z = np.unique(st.logZm)[2:2+6] # all metallicities between -0.75 and 0.0; index 2 has -0.45
m = np.isin(st.t, t) & np.isin(st.logZm, Z)
st.select(m)
st.set_omega0()
print('\t' + str(time.time() - start) + ' seconds.')

# age and metallicity to use for inclination refinement
t_inc = t[8]
z_inc = Z[3]

print('Loading PARS...')
start = time.time()
with open('data/pars_grid_2.pkl', 'rb') as f: pars = pickle.load(f)
print('\t' + str(time.time() - start) + ' seconds.\n')

# set the parameters of the grid class
mu.Grid.std = cf.std # standard deviations of observables
mu.Grid.modulus = cf.modulus # distance modulus of the cluster
mu.Grid.pars = pars # PARS grid
dmax = 2.

# compute the inclination once
start = time.time()
st1 = st.copy() # copy model set
st1.select_tZ(t_inc, z_inc)
# apply mass cut-offs given the boundaries of the observable space region where we need model priors
Mmin, Mmax = Mlim(st1, cf.A_V)
st1.select_mass(Mmin=Mmin, Mmax=Mmax)
inclination = rc_inc(st1, cf.A_V, dmax, dmax)
print('inclination: ' + str(time.time() - start) + ' seconds.')

for t0 in t:
	for z0 in Z:
		st1 = st.copy() # copy model set
		st1.select_tZ(t0, z0)
		# apply mass cut-offs given the boundaries of the observable space region where we need model priors
		Mmin, Mmax = Mlim(st1, cf.A_V)
		st1.select_mass(Mmin=Mmin, Mmax=Mmax)
		print(t0, z0)

		start = time.time()
		Mini, oM0 = rc_mass_omega(st1, cf.A_V, dmax)
		print('\t' + str(time.time() - start) + ' seconds.')

		zstr = str(z0).replace('-', 'm').replace('.', 'p')
		tstr = str(t0)[:4].replace('.', 'p')
		# create a grid that combines the separately obtained model parameter arrays
		grid = mu.Grid(st1, Mini, oM0, inclination, cf.A_V, verbose=True)
		grid.plot_diff(0, 'data/model_spacing/mass/diff_vs_Mini_' + tstr + '_' + zstr + '.png')
		grid.plot_diff(1, 'data/model_spacing/omega/diff_vs_omega0_' + tstr + '_' + zstr + '.png')
		grid.plot_diff(2, 'data/model_spacing/inc/diff_vs_inc_' + tstr + '_' + zstr + '.png')
		with open('data/model_grids/grid_' + tstr + '_' + zstr + '.pkl', 'wb') as f:
		    pickle.dump(grid.pickle(), f)