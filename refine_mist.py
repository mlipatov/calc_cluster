import sys, os, time, pickle
sys.path.append(os.path.abspath(os.path.join('..', 'paint_atmospheres')))
from pa.lib import surface as sf
from pa.lib import util as ut
from pa.opt import grid as gd

from lib import mist_util as mu
import config as cf
from lib import load_data as ld

from scipy.interpolate import interpn
import numpy as np
from matplotlib import pyplot as plt

# calculate the mass cut-offs given the boundaries of the observable space region
# where we need model priors
def Mlim(st, t, Z, A_V):
	Mi = np.array(np.linspace(st.Mini.min(), st.Mini.max(), 1000)) # a mass grid
	omega0 = np.array([st.omega0.min(), st.omega0.max()]) 
	inc = np.array([0, np.pi/2]) # the two extreme inclinations
	grid = mu.Grid(st, t, Z, Mi, omega0, inc, cf.A_V)
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
def rc_mass_omega_inc(st, t, Z, A_V, dmax):
	Mi = np.sort(np.unique(st.Mini)) # original mass grid
	o0 = np.sort(np.unique(st.omega0)) # original omega grid
	inc = np.linspace(0, np.pi/2, 10) # np.array([0, np.pi/2]) # relatively small inclination grid
	grid = mu.Grid(st, t, Z, Mi, o0, inc, A_V)

	imax = np.nanmax(grid.get_maxdiff(2))
	omax = np.nanmax(grid.get_maxdiff(1))
	mmax = np.nanmax(grid.get_maxdiff(0))
	while omax > dmax or mmax > dmax or imax > dmax:
		while imax > dmax:
			grid.refine(2, dmin=dmax)
			imax = np.nanmax(grid.get_maxdiff(2))
		grid.coarsen(2, dmax=dmax)
		omax = np.nanmax(grid.get_maxdiff(1))
		mmax = np.nanmax(grid.get_maxdiff(0))

		while omax > dmax:
			grid.refine(1, dmin=dmax)
			omax = np.nanmax(grid.get_maxdiff(1))
		grid.coarsen(1, dmax=dmax)
		mmax = np.nanmax(grid.get_maxdiff(0))
		imax = np.nanmax(grid.get_maxdiff(2))
		
		while mmax > dmax:
			grid.refine(0, dmin=dmax)
			mmax = np.nanmax(grid.get_maxdiff(0))
		grid.coarsen(0, dmax=dmax)
		omax = np.nanmax(grid.get_maxdiff(1))
		imax = np.nanmax(grid.get_maxdiff(2))
	return grid

# # refine and coarsen in the mass and omega dimension, 
# # given initial guesses for all model grids
# def rc_mass_omega(st, t, Z, Mi, o0, inc, A_V, dmax):
# 	grid = mu.Grid(st, t, Z, Mi, o0, inc, A_V)
# 	omax = np.nanmax(grid.get_maxdiff(1))
# 	mmax = np.nanmax(grid.get_maxdiff(0))
# 	while omax > dmax or mmax > dmax:
# 		while omax > dmax:
# 			grid.refine(1, dmin=dmax)
# 			omax = np.nanmax(grid.get_maxdiff(1))
# 		grid.coarsen(1, dmax=dmax)
# 		mmax = np.nanmax(grid.get_maxdiff(0))
# 		imax = np.nanmax(grid.get_maxdiff(2))
		
# 		while mmax > dmax:
# 			grid.refine(0, dmin=dmax)
# 			mmax = np.nanmax(grid.get_maxdiff(0))
# 		grid.coarsen(0, dmax=dmax)
# 		omax = np.nanmax(grid.get_maxdiff(1))
# 		imax = np.nanmax(grid.get_maxdiff(2))
# 	return grid

# pre-compute Roche model volume versus PARS's omega
# and PARS's omega versus MESA's omega
sf.calcVA()
sf.calcom()

# Load and filter MIST models
print('Loading MIST...', end='')
start = time.time()
st = mu.Set('data/mist_grid.npy')
print('%.2f' % (time.time() - start) + ' seconds.')

st.select_MS() # main sequence
st.select_valid_rotation()
st.set_omega0()

print('Loading PARS...', end='')
start = time.time()
with open('data/pars_grid_2.pkl', 'rb') as f: pars = pickle.load(f)
print('%.2f' % (time.time() - start) + ' seconds.' + '\n')

# set the parameters of the grid class
mu.Grid.std = cf.std # standard deviations of observables
mu.Grid.modulus = cf.modulus # distance modulus of the cluster
mu.Grid.pars = pars # PARS grid

t = np.unique(st.t)[106:106+4] # age range; 4 ages: [9.134, 9.154, 9.174, 9.194]
z0 = np.unique(st.logZm)[4] # Z = -0.45
st.select_Z(z0) # select metallicity
zstr = str(z0).replace('-', 'm').replace('.', 'p')

grids = [] # refined grids in model parameters
mg = []; og = []; ig = []; # refined grids in the model parameters
# binary mass ratio spaced so that magnitudes are spaced evenly
r = np.linspace(0, 1, cf.num_r)**(1 / cf.s) 
# for z0 in np.flip(Z):
print('Z = ' + str(z0) + '\n')
for t0 in t:
	print('t = ' + str(t0))
	st1 = st.copy() # copy the model set
	st1.select_age(t0)

	# non-rotating models at this age and this metallicity
	stc = st1.copy()
	stc.select(stc.omega0 == 0)

	# apply mass cut-offs according the region of interest on the CMD 
	start = time.time()
	print('Applying mass cut-offs...', end='')
	Mmin, Mmax = Mlim(st1, t0, z0, cf.A_V) 
	st1.select_mass(Mmin=Mmin, Mmax=Mmax)
	print('%.2f' % (time.time() - start) + ' seconds.')

	start = time.time()
	print('Refining the mass, omega and inclination grids...', end='')
	# coarsen and refine model grids
	grid = rc_mass_omega_inc(st1, t0, z0, cf.A_V, cf.dmax) 
	grids.append(grid)
	og.append(grid.omega0)
	ig.append(grid.inc)
	mg.append(grid.Mini)
	print('%.2f' % (time.time() - start) + ' seconds.')

	# non-rotating companion magnitudes on a M * r grid
	print('Computing the companion magnitudes...')
	mag = mu.companion_grid(r, grid.Mini, stc, pars, cf.A_V, cf.modulus)

	print()
	tstr = str(t0)[:4].replace('.', 'p')
	grid.plot_diff(0, 'data/model_spacing/mass/diff_vs_Mini_' + tstr + '_' + zstr + '.png')
	grid.plot_diff(1, 'data/model_spacing/omega/diff_vs_omega0_' + tstr + '_' + zstr + '.png')
	grid.plot_diff(2, 'data/model_spacing/inc/diff_vs_inc_' + tstr + '_' + zstr + '.png')
	with open('data/model_grids/grid_' + tstr + '_' + zstr + '.pkl', 'wb') as f:
	    pickle.dump([grid.pickle(), mag, r], f)

## now refine the age grid and interpolate the mass-omega-inclination grids on it

# interpolate between backward and forward closest neighbors on 1D grids g1 and g2 in 1D grid g; 
# none of the grids have to have the same dimension
def double_interp(g1, g2, g):
	# for each element in 1D grid g1, get the index of the closest element in g2
	# g1 and g2 don't have to have the same dimensions
	# output has the same dimensions as g1
	def closest(g1, g2):
		return np.array([g2[np.abs(g2 - el).argmin()] for el in g1])
	# interpolate from 1D grid g1 directionally towards 1D grid g2 on a third 1D grid g
	# in other words, interpolate between (g[0], g1) and (g[-1], g2) at g[1:-1]
	# g1 and g2 must have the same dimensions
	# output has dimensions of g1 times g
	def interp(g1, g2, g):
		enum = np.arange(len(g1)) # enumerate the first grid
		points = (g[[0,-1]], enum) # domain of known values: the boundary values of g times the enumeration 
		values = np.vstack((g1, g2)) # known values on the domain: the two grids
		xi = np.moveaxis(np.array(np.meshgrid(g, enum)), 0, -1) # desired value domain 
		return interpn(points, values, xi, method='linear').T
	# interpolate in g from g1 towards g2
	inter1 = interp(g1, closest(g1, g2), g)
	# interpolate in age between the grid at i+1th age and the grid points' neighbors at ith age
	inter2 = interp(closest(g2, g1), g2, g)
	# construct the union
	inter = np.concatenate((inter1, inter2), axis=-1)
	return [np.unique(i) for i in inter]	

tt = np.array([]) # intermediate ages
mgg = []; ogg = []; igg = []; # interpolated model space grids
n = 20 # age refinement factor
for i in range(len(t) - 1):
	ti = np.linspace(t[i], t[i+1], n)
	mgg = mgg + double_interp(mg[i], mg[i+1], ti)[1:-1]
	ogg = ogg + double_interp(og[i], og[i+1], ti)[1:-1]
	igg = igg + double_interp(ig[i], ig[i+1], ti)[1:-1]
	tt = np.concatenate((tt, ti[1:-1]))

# compute magnitudes for the intermediate grids
for i in range(len(tt)):
	t0 = tt[i]
	print('t = ' + str(t0) + '...')
	grid = mu.Grid(st, t0, z0, mgg[i], ogg[i], igg[i], cf.A_V, verbose=True)
	grids.append(grid)
	# start = time.time()
	# grid = rc_mass_omega(st, tt[i], z0, mgg[i], ogg[i], igg[i], cf.A_V, cf.dmax)
	mmax = np.nanmax(grid.get_maxdiff(0))
	omax = np.nanmax(grid.get_maxdiff(1))
	imax = np.nanmax(grid.get_maxdiff(2))
	# print (str(time.time() - start) + ' seconds.')
	print (mmax, omax, imax)

	# non-rotating companion magnitudes on a M * r grid
	print('Computing the companion magnitudes...')
	mag = mu.companion_grid(r, grid.Mini, stc, pars, cf.A_V, cf.modulus)

	tstr = str(t0)[:5].replace('.', 'p')
	grid.plot_diff(0, 'data/model_spacing/mass/diff_vs_Mini_' + tstr + '_' + zstr + '.png')
	grid.plot_diff(1, 'data/model_spacing/omega/diff_vs_omega0_' + tstr + '_' + zstr + '.png')
	grid.plot_diff(2, 'data/model_spacing/inc/diff_vs_inc_' + tstr + '_' + zstr + '.png')
	with open('data/model_grids/grid_' + tstr + '_' + zstr + '.pkl', 'wb') as f:
	    pickle.dump([grid.pickle(), mag, r], f)

# sort the ages and the grids
t = np.concatenate((t, tt))
ag = sorted(zip(t, grids))
## calculate maximum magnitude distances forward and backward in age
# for each element in 1D grid g1, return the index of the closest neighbor in 1D grid g2
# size of the output is equal to the size of g1
def closest_ind(g1, g2):
    ir = np.searchsorted(g2, g1) # right index
    ir[ir == g2.shape[0]] = g2.shape[0] - 1 # get right index back within array bounds
    il = ir - 1 # left index
    il[il == -1] = 0 # get left index back within array bounds
    i = il
    m = np.abs(g2[ir] - g1) < np.abs(g2[il] - g1)
    i[m] = ir[m]
    return i

# return maximum magnitude difference from one model grid to another
def maxdiff(g1, g2):
	mi = closest_ind(g1.Mini, g2.Mini)
	oi = closest_ind(g1.omega0, g2.omega0)
	ii = closest_ind(g1.inc, g2.inc)
	diff = g2.obs.take(mi, 0).take(oi, 1).take(ii, 2) - g1.obs
	return np.nanmax(np.abs(diff))

for i in range(len(ag) - 1):
	g1 = ag[i][1]
	g2 = ag[i+1][1]
	md = np.nanmax([maxdiff(g1, g2), maxdiff(g2, g1)]) # maximum difference
	print('magnitude difference between ages ' + '%.4f' % ag[i][0] + ' and ' + \
		'%.4f' % ag[i+1][0] + ': ' + '%.2f' % md)