import sys
sys.path.append('..')
# plot the dependence of probability change on sigma
import os, glob, pickle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 12
dims = ['mag', 'col', 'vsini']
dimensions = ['magnitude', 'color', r'$v\,\sin{i}$']

filelist = list(np.sort(glob.glob('../data/densities/pkl/*.pkl')))
prefix = '../data/normalization/'
for filepath in filelist: # for each combination of age and metallicity
	suffix = os.path.basename(filepath).split('.')[0][7:]
	# load the pre-computed density on a grid of observables
	with open(filepath, 'rb') as f:
		density = pickle.load(f)
	for axis in range(len(density.dP)):
		dP_spline = density.dP[axis]
		if dP_spline is not None:
			x = dP_spline.x; y = dP_spline.y
			xplot = np.linspace(x.min(), x.max(), 100)
			fig, ax = plt.subplots()
			ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
			ax.plot(xplot, dP_spline(xplot), c='g')
			ax.scatter(x, y, facecolors='w', edgecolors='k')
			ax.set_ylabel(r'$\Delta P$', labelpad=20)
			ax.set_xlabel(r'$\sigma$')
			ax.spines["top"].set_visible(False)
			ax.spines["right"].set_visible(False)
			textstr = '\n'.join((
			    r'$log_{10}{t}=' + str(density.age)[:4] + '$',
			    r'$[M/H]_{MIST}=' + str(density.Z) + '$',
				r'Dimension: ' + dimensions[axis] ))
			if y[-1] > y[0]: txt_x = 0.15
			else: txt_x = 0.70
			ax.text(txt_x, 1.05, textstr, fontsize=12, transform=ax.transAxes, horizontalalignment='left',
			        verticalalignment='top', bbox=dict(facecolor='w', alpha=0.0, edgecolor='w'))
			# write plot file
			plt.savefig(prefix + dims[axis] + '/dP_' + suffix + '.png', dpi=300)
			plt.close()