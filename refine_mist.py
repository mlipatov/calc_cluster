import sys, os, time, pickle
sys.path.append(os.path.abspath(os.path.join('..', 'paint_atmospheres')))
from pa.lib import surface as sf
from pa.lib import util as ut
from pa.opt import grid as gd

from lib import mist_util as mu
import config as cf
from lib import load_data as ld

from scipy.interpolate import griddata
import numpy as np

# calculate the mass cut-offs given the boundaries of the observable space region
# where we need model priors
def Mlim(st, A_V):
	Mi = np.sort(np.unique(st.Mini)) # original mass grid
	omega0 = np.array([st.omega0.min(), st.omega0.max()]) 
	inc = np.array([0, np.pi/2]) # the two extreme inclinations
	grid = mu.Grid(st, Mi, omega0, inc, cf.A_V)
	# maximum mass for the minimum magnitude cutoff 
	# is determined by the models without companions at inclination pi/2;
	# reducing inclination / adding a companion only reduces magnitude
	mag_min = grid.obs[..., -1, 0] # last index is magnitude, penultimate - inclination
	# minimum mass for the maximum magnitude cutoff 
	# is determined by the models at inclination 0 with equal-mass companions;
	# increasing inclination / reducing companion mass only increases magnitude
	mag_max = mu.combine_mags(grid.obs[..., 0, 0], grid.obs[..., [0], 0, 0]) # index before inclination is rotation
	m_min = np.full_like(mag_min, False, dtype=bool) # will be True where magnitude is above minimum
	m_max = np.full_like(mag_max, False, dtype=bool) # will be True where magnitude is below maximum
	np.greater_equal(mag_min, ld.obmin[np.newaxis, np.newaxis, 0], where=~np.isnan(mag_min), out=m_min) 
	np.less_equal(mag_max, ld.obmax[np.newaxis, np.newaxis, 0], where=~np.isnan(mag_max), out=m_max)
	j = np.nonzero( np.all(m_min & m_max, axis=-1) )[0]
	if len(j) > 0:
		Mmin = grid.Mini[j].min()
		Mmax = grid.Mini[j].max()
	else:
		Mmin = np.nan
		Mmax = np.nan
	return [Mmin, Mmax]

# refine and coarsen in the mass and omega dimensions
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

# valid rotation
st.select_valid_rotation()
# on the MS
st.select_MS()
# a range of ages and metallicities
# t = np.unique(st.t)[99:99+17]
t = np.unique(st.t)[106:106+3] # target ages; index 1 has 9.154
Z = np.unique(st.logZm)[3:3+4] # target metallicities between -0.75 and 0.0; index 1 has -0.45
m = np.isin(st.t, t) & np.isin(st.logZm, Z)
st.select(m)
st.set_omega0()
print('\t' + str(time.time() - start) + ' seconds.')

# age and metallicity to use for inclination refinement
t_inc = t[1]
z_inc = Z[1]

print('Loading PARS...')
start = time.time()
with open('data/pars_grid_2.pkl', 'rb') as f: pars = pickle.load(f)
print('\t' + str(time.time() - start) + ' seconds.\n')

# set the parameters of the grid class
mu.Grid.std = cf.std # standard deviations of observables
mu.Grid.modulus = cf.modulus # distance modulus of the cluster
mu.Grid.pars = pars # PARS grid

# compute the inclination once
start = time.time()
st1 = st.copy() # copy the model set
st1.select_age(t_inc)
st1.select_Z(z_inc)
# apply mass cut-offs given the boundaries of the observable space region where we need model priors
Mmin, Mmax = Mlim(st1, cf.A_V)
st1.select_mass(Mmin=Mmin, Mmax=Mmax)
inclination = rc_inc(st1, cf.A_V, cf.dmax, cf.dmax)
print('inclination: ' + str(time.time() - start) + ' seconds.')

for z0 in np.flip(Z):
	for t0 in t:
		print(t0, z0)
		st1 = st.copy() # copy the model set
		st1.select_Z(z0) # select metallicity
		st1.select_age(t0) # select age

		# non-rotating models at this age and this metallicity
		stc = st1.copy()
		stc.select(stc.omega0 == 0)

		# apply mass cut-offs according the region of interest on the CMD 
		Mmin, Mmax = Mlim(st1, cf.A_V) 
		st1.select_mass(Mmin=Mmin, Mmax=Mmax)

		start = time.time()
		print('Refining the mass and omega grids...')
		Mini, omega0 = rc_mass_omega(st1, cf.A_V, cf.dmax) # coarsen and refine mass and omega grids
		print('\t' + str(time.time() - start) + ' seconds.')

		print('Computing the companion magnitudes...')
		# primary grid that combines the separately obtained model parameter arrays
		grid = mu.Grid(st1, Mini, omega0, inclination, cf.A_V, verbose=True)
		# binary mass ratio spaced so that magnitudes are spaced evenly
		r = np.linspace(0, 1, cf.num_r)**(1 / cf.s) 
		# non-rotating companion magnitudes on a M * r grid
		mag = mu.companion_grid(r, Mini, stc, pars, cf.A_V, cf.modulus)

		print()
		zstr = str(z0).replace('-', 'm').replace('.', 'p')
		tstr = str(t0)[:4].replace('.', 'p')
		grid.plot_diff(0, 'data/model_spacing/mass/diff_vs_Mini_' + tstr + '_' + zstr + '.png')
		grid.plot_diff(1, 'data/model_spacing/omega/diff_vs_omega0_' + tstr + '_' + zstr + '.png')
		grid.plot_diff(2, 'data/model_spacing/inc/diff_vs_inc_' + tstr + '_' + zstr + '.png')
		with open('data/model_grids/grid_' + tstr + '_' + zstr + '.pkl', 'wb') as f:
		    pickle.dump([grid.pickle(), mag, r], f)