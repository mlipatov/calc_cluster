import config as cf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 12

# plot denormalization versus sigma
# Inputs:
#	prefix and suffix of the plot filepath, age and metallicity for plots
def dP_sigma(x, y, fit, prefix, suffix, age, Z, dimension, dim, use='yes'):
	xplot = np.linspace(x.min(), x.max(), 100)
	fig, ax = plt.subplots()
	ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
	ax.plot(xplot, fit(xplot), c='g')
	ax.scatter(x, y, facecolors='w', edgecolors='k')
	ax.set_ylabel(r'$\Delta P$', labelpad=20)
	ax.set_xlabel(r'$\sigma$')
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	textstr = '\n'.join((
	    r'$log_{10}{t}=' + str(age)[:4] + '$',
	    r'$[M/H]_{MIST}=' + str(Z) + '$',
	    r'$A_V=%.2f$' % (cf.A_V, ),
		r'Dimension: ' + dimension,
		r'Use: ' + use ))
	if y[-1] > y[0]: txt_x = 0.15
	else: txt_x = 0.70
	ax.text(txt_x, 1.05, textstr, fontsize=12, transform=ax.transAxes, horizontalalignment='left',
	        verticalalignment='top', bbox=dict(facecolor='w', alpha=0.0, edgecolor='w'))
	# write plot file
	plt.savefig(prefix + dim + '/dP_' + suffix + '.png', dpi=300)
	plt.close()