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

# calculate the lower mass cut-off given the boundaries of the observable space region
# where we need model priors
def Mlim(st):
	Mi = np.array(np.linspace(st.Mini.min(), st.Mini.max(), 100)) # mass grid
	omega0 = np.array([st.omega0.min(), st.omega0.max()]) # extreme omegas
	t = np.array([st.t.max()]) # largest age; decreasing it only increases magnitude
	inc = np.array([0]) # smallest inclination; increasing it only increases magnitude
	grid = mu.Grid(st, Mi, omega0, t, inc, cf.A_V)
	# smallest mass at the maximum magnitude cutoff 
	# determined by the models with equal-mass non-rotating companions;
	# reducing companion mass only increases magnitude
	mag_max = mu.combine_mags(grid.obs[..., 0], np.expand_dims(grid.obs[:, 0, :, :, 0], 1))
	m_max = np.full_like(mag_max, False, dtype=bool) # will be true where magnitude is below maximum
	np.less_equal(mag_max, ld.obmax[np.newaxis, np.newaxis, 0], where=~np.isnan(mag_max), out=m_max)
	j = np.nonzero( np.any(m_max, axis=-1) )[0] # indices of masses that are above the minimum mass
	if len(j) > 0: # if there are masses on the CMD
		jmin = j.min();
		# subtract one from the minimum mass index if possible 
		if jmin > 0:
			jmin-=1
		Mmin = grid.Mini[jmin]
	else:
		Mmin = np.nan
	return Mmin

# refine and coarsen model grid between two ages
def refine_coarsen(st, t1, t2):
	# get maximum differences for all dimensions
	def diffs(grid):
		return np.array([np.nanmax(grid.get_maxdiff(i)) for i in range(grid.ndim)])
	# get grid lengths for all dimensions
	def lengths(grid):
		return np.array([len(getattr(grid, grid.ivars[i])) for i in range(grid.ndim)])

	Mi = np.unique(st.Mini) # original mass grid
	o0 = np.array([st.omega0.min(), st.omega0.max()]) # small omega grid
	t = np.array([t1, t2]) # age grid based on the given ages
	inc = np.linspace(0, np.pi/2, 2) # small inclination grid
	grid = mu.Grid(st, Mi, o0, t, inc, cf.A_V)
	grid.coarsen(0, dmax=cf.dmax) # coarsen the mass grid
	md = diffs(grid) 	# get maximum differences for all dimensions
	gl = lengths(grid) 	# get grid lengths for all dimensions
	print(md, flush=True)
	print(gl, flush=True)
	
	while np.any(md > cf.dmax): # while any of the maximum differences are above the cutoff 
		for i in range(len(md)): # for each of the dimensions
			while md[i] > cf.dmax: # while this dimension is not fine enough
				ivar = grid.ivars[i]
				var = getattr(grid, ivar) # get the grid in this dimension
				print('Refining the ' + ivar + ' dimension.', flush=True)
				while md[i] > cf.dmax: # while the maximum difference in this dimension is above the cutoff
					grid.refine(i, dmin=cf.dmax) # refine this dimension
					md = diffs(grid) # update maximum differences
				gl = lengths(grid) # update the grid length in this dimension
				print(md, flush=True)
				print(gl, flush=True)
				print('Coarsening the ' + ivar + ' dimension.', flush=True)
				grid.coarsen(i, dmax=cf.dmax)
				gl = lengths(grid) # get new grid lengths
				md = diffs(grid)
				print(md)
				print(gl)
				print()

	return grid

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

# t = np.unique(st.t)[107:109]
t = np.unique(st.t)[105:105+6] # target ages, around 9.154
st.select(np.isin(st.t, t)) # select the ages
splits = [11, 6, 5, 3, 2] # number of ages for each interval to give linspace

st.select_Z(cf.Z) # select metallicity
zstr = str(cf.Z).replace('-', 'm').replace('.', 'p')

print('Loading PARS...', end='', flush=True)
start = time.time()
with open('data/pars_grid_2.pkl', 'rb') as f: pars = pickle.load(f)
print('%.2f' % (time.time() - start) + ' seconds.' + '\n', flush=True)

# set the parameters of the grid class
mu.Grid.std = cf.std # standard deviations of observables
mu.Grid.modulus = cf.modulus # distance modulus of the cluster
mu.Grid.pars = pars # PARS grid

# apply mass cut-off according the region of interest on the CMD 
start = time.time()
print('Applying mass cut-off...', end='', flush=True)
Mmin = Mlim(st)
st.select_mass(Mmin=Mmin)
print('minimum mass = ' + '%.4f' % Mmin + '; %.2f' % (time.time() - start) + ' seconds.', flush=True)

grids = [] # refined grids in model parameters
# binary mass ratio spaced so that magnitudes are spaced evenly
r = np.linspace(0, 1, cf.num_r)**(1 / cf.s) 

for i in range(len(t) - 1):
	st1 = st.copy() # copy the model set
	st1.select( (st1.t == t[i]) | (st1.t == t[i+1]) )

	# non-rotating models at these ages
	stc = st1.copy()
	stc.select(stc.omega0 == 0)

	# # refine the age grid, especially at earlier ages, 
	# # where more masses span the main sequence below the magnitude cut-off
	# t1 = np.linspace(t[i], t[i+1], splits[i])

	# for j in range(len(t1) - 1):
		# print('t = ' + '%.4f' % t1[j] + ' and ' + '%.4f' % t1[j+1] + '\n')
	print('t = ' + '%.4f' % t[i] + ' and ' + '%.4f' % t[i+1] + '\n')

	start = time.time()
	print('Refining the mass, omega and inclination grids...', flush=True)
	# coarsen and refine model grid between these two ages, using the appropriate model set
	grid = refine_coarsen(st1, t[i], t[i+1]) 
	print('%.2f' % (time.time() - start) + ' seconds.', flush=True)

	# non-rotating companion magnitudes on a M * r grid
	mag = mu.companion_grid(r, grid.Mini, stc, pars, cf.A_V, cf.modulus)

	print()
	tstr = 't' + ('%.5f' % t[i]).replace('.', 'p') + '_' + ('%.5f' % t[i+1]).replace('.', 'p')
	grid.plot_diff(0, 'data/model_grids/png/diff_vs_Mini_' + tstr + '_' + zstr + '.png')
	grid.plot_diff(1, 'data/model_grids/png/diff_vs_omega0_' + tstr + '_' + zstr + '.png')
	grid.plot_diff(2, 'data/model_grids/png/diff_vs_t_' + tstr + '_' + zstr + '.png')
	grid.plot_diff(3, 'data/model_grids/png/diff_vs_inc_' + tstr + '_' + zstr + '.png')
	with open('data/model_grids/pkl/grid_' + tstr + '_' + zstr + '.pkl', 'wb') as f:
	    pickle.dump([grid.pickle(), mag, r], f)