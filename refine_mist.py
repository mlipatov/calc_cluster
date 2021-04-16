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
	Mi = np.array(np.linspace(st.Mini.min(), st.Mini.max(), 100)) # mass grid
	omega0 = np.array([st.omega0.min(), st.omega0.max()]) 
	t = np.array([st.t.min(), st.t.max()])
	inc = np.array([0, np.pi/2]) # the two extreme inclinations
	grid = mu.Grid(st, Mi, omega0, t, inc, cf.A_V)
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
	j = np.nonzero( np.any(m_min & m_max, axis=-1) )[0]
	if len(j) > 0:
		jmin = j.min(); 
		if jmin > 0:
			jmin-=1
		jmax = j.max(); 
		if jmax < len(Mi) - 1:	
			jmax+=1
		Mmin = grid.Mini[jmin]
		Mmax = grid.Mini[jmax]
	else:
		Mmin = np.nan
		Mmax = np.nan
	return [Mmin, Mmax]

# refine and coarsen model dimensions
def refine_coarsen(st, A_V, dmax):
	Mi = np.unique(st.Mini) # original mass grid
	o0 = np.array([st.omega0.min(), st.omega0.max()]) # small omega grid
	t = np.unique(st.t) # original age grid
	inc = np.linspace(0, np.pi/2, 2) # a relatively small inclination grid
	grid = mu.Grid(st, Mi, o0, t, inc, A_V)

	imax = np.nanmax(grid.get_maxdiff(3))
	tmax = np.nanmax(grid.get_maxdiff(2))
	omax = np.nanmax(grid.get_maxdiff(1))
	mmax = np.nanmax(grid.get_maxdiff(0))

	while omax > dmax or mmax > dmax or imax > dmax: #or tmax > dmax:
		grid.coarsen(0, dmax=dmax)
		print('mass: ' + str(mmax) + ', ' + str(len(grid.Mini)))
		while mmax > dmax:
			grid.refine(0, dmin=dmax)
			mmax = np.nanmax(grid.get_maxdiff(0))
			print('mass: ' + str(mmax) + ', ' + str(len(grid.Mini)))
		grid.coarsen(0, dmax=dmax)
		print('mass: ' + str(mmax) + ', ' + str(len(grid.Mini)))
		omax = np.nanmax(grid.get_maxdiff(1))
		imax = np.nanmax(grid.get_maxdiff(3))
		tmax = np.nanmax(grid.get_maxdiff(2))

		while omax > dmax:
			grid.refine(1, dmin=dmax)
			omax = np.nanmax(grid.get_maxdiff(1))
			print('omega: ' + str(omax) + ', ' + str(len(grid.omega0)))
		grid.coarsen(1, dmax=dmax)
		print('omega: ' + str(omax) + ', ' + str(len(grid.omega0)))
		mmax = np.nanmax(grid.get_maxdiff(0))
		imax = np.nanmax(grid.get_maxdiff(3))
		tmax = np.nanmax(grid.get_maxdiff(2))

		while imax > dmax:
			grid.refine(3, dmin=dmax)
			imax = np.nanmax(grid.get_maxdiff(3))
			print('inc: ' + str(imax) + ', ' + str(len(grid.inc)))
		grid.coarsen(3, dmax=dmax)
		print('inc: ' + str(imax) + ', ' + str(len(grid.inc)))
		omax = np.nanmax(grid.get_maxdiff(1))
		mmax = np.nanmax(grid.get_maxdiff(0))
		tmax = np.nanmax(grid.get_maxdiff(2))

		while tmax > dmax:
			grid.refine(2, dmin=dmax)
			tmax = np.nanmax(grid.get_maxdiff(2))
			print('age: ' + str(tmax) + ', ' + str(len(grid.t)))
		grid.coarsen(2, dmax=dmax)
		print('age: ' + str(tmax) + ', ' + str(len(grid.t)))
		omax = np.nanmax(grid.get_maxdiff(1))
		mmax = np.nanmax(grid.get_maxdiff(0))
		imax = np.nanmax(grid.get_maxdiff(3))

	return [grid.Mini, grid.omega0, grid.t, grid.inc]

# pre-compute Roche model volume versus PARS's omega
# and PARS's omega versus MESA's omega
sf.calcVA()
sf.calcom()

start = time.time()
print('Loading MIST...', end='', flush=True)
st = mu.Set('data/mist_grid.npy') # load
print(str(time.time() - start) + ' seconds.')

# valid rotation
st.select_valid_rotation()
st.set_omega0()
st.select_MS() # select main sequence
t = np.unique(st.t)[106:106+4] # target ages; index 1 has 9.154
z0 = -0.45
st.select(np.isin(st.t, t)) # select the ages
st.select_Z(z0) # select metallicity

start = time.time()
print('Loading PARS...', end='', flush=True)
with open('data/pars_grid_2.pkl', 'rb') as f: pars = pickle.load(f)
print(str(time.time() - start) + ' seconds.\n')

# set the parameters of the grid class
mu.Grid.std = cf.std # standard deviations of observables
mu.Grid.modulus = cf.modulus # distance modulus of the cluster
mu.Grid.pars = pars # PARS grid

print('Metallicity = ' + str(z0))

# non-rotating models at this age and this metallicity
stc = st.copy()
stc.select(stc.omega0 == 0)

# apply mass cut-offs according the region of interest on the CMD 
start = time.time()
print('Applying mass cut-offs...', end='', flush=True)
Mmin, Mmax = Mlim(st, cf.A_V) 
st.select_mass(Mmin=Mmin) #, Mmax=Mmax)
print(str(time.time() - start) + ' seconds.')

start = time.time()
print('Refining the mass, omega, inclination and age grids...', flush=True)
Mini, omega0, t, inclination = refine_coarsen(st, cf.A_V, cf.dmax)
print(str(time.time() - start) + ' seconds.')

# primary grid that combines the separately obtained model parameter arrays
grid = mu.Grid(st, Mini, omega0, t, inclination, cf.A_V, verbose=True)

print('Computing the companion magnitudes...')
# binary mass ratio spaced so that magnitudes are spaced evenly
r = np.linspace(0, 1, cf.num_r)**(1 / cf.s) 
# non-rotating companion magnitudes on a M * r grid
mag = mu.companion_grid(r, Mini, stc, pars, cf.A_V, cf.modulus)

print()
zstr = str(z0).replace('-', 'm').replace('.', 'p')
# tstr = str(t0)[:4].replace('.', 'p')
grid.plot_diff(0, 'data/model_spacing/mass/diff_vs_Mini_' + zstr + '.png')
grid.plot_diff(1, 'data/model_spacing/omega/diff_vs_omega0_' + zstr + '.png')
grid.plot_diff(2, 'data/model_spacing/inc/diff_vs_inc_' + zstr + '.png')
with open('data/model_grids/grid_' + zstr + '.pkl', 'wb') as f:
    pickle.dump([grid.pickle(), mag, r], f)