import sys
sys.path.append('..')

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
import pickle, glob, os

mpl.rcParams['font.size'] = 12
w0 = np.linspace(0, 1, 101, dtype=float) # proportion of the zero rotational population
w1 = np.linspace(0, 1, 101, dtype=float) # proportion of the maximum rotational population

filelist = list(np.sort(glob.glob('../data/ll*.pkl')))
for filepath in filelist: # for each combination of age and metallicity
	# load the density file
	with open(filepath, 'rb') as f:
		ll = pickle.load(f)
		ll -= np.nanmax(ll)

	# base file name
	base = os.path.basename(filepath).split('.')[0]
	if len(base.split('p')[-1]) == 1:
		base = base + '0'

	# color map
	cmapBig = mpl.cm.get_cmap('viridis_r', 512)
	cmap = mpl.colors.ListedColormap(cmapBig(np.linspace(0, 1, 256)))
	norm = mpl.colors.Normalize(vmin=np.nanmin(ll), vmax=np.nanmax(ll), clip=True)

	# text box text
	textstr = '\n'.join((
	    r'$log_{10}{t}=' + base.split('_')[-2].replace('p','.') + '$',
	    r'$[M/H]_{MIST}=' + base.split('_')[-1].replace('p','.').replace('m','-') + '$'))


	fig, [ax, ax1] = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [8, 1]})
	plt.subplots_adjust(left=0.13, right=0.8, bottom=0.13, top=0.95)
	pcm = ax.pcolormesh(w0, w1, ll, cmap=cmap, shading='nearest', norm=norm) 
	ax.set_xlabel(r'$w_1$')
	ax.set_ylabel(r'$w_0$')
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	
	# color bar
	ax1.axis('off')
	cax = fig.add_axes([0.7, 0.2, 0.03, 0.4])	
	ticks = ticker.LinearLocator(5)
	cb = fig.colorbar(mappable=pcm, ax=ax1, cax=cax, orientation='vertical', ticks=ticks, \
		format='%.1f', alpha=1.0, shrink=0.6, norm=norm)
	cb.set_label(label=r'$\ln{\cal L}$', fontsize=18, rotation=0, labelpad=35, y=0.65)

	# text box
	ax.text(1.0, 1.0, textstr, transform=ax.transAxes, fontsize=14,
	        verticalalignment='top', bbox=dict(facecolor='w', alpha=1.0, edgecolor='w'))

	plt.savefig('../data/' + base + '.png', dpi=300)
	plt.close()