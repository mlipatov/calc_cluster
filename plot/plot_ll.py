import sys
sys.path.append('..')

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
import pickle, glob, os

mpl.rcParams['font.size'] = 12
w0 = np.linspace(0, 1, 21, dtype=float) # proportion of the zero rotational population
w1 = np.linspace(0, 1, 21, dtype=float) # proportion of the maximum rotational population
b = np.linspace(0, 1, 21, dtype=float) # proportion of the binaries population

filelist = list(np.sort(glob.glob('../data/likelihoods/pkl/ll*.pkl')))
for filepath in filelist: # for each combination of age and metallicity
	# load the log likelihood and the maximum q
	with open(filepath, 'rb') as f:
		ll_3d, qm_3d, om_sigma = pickle.load(f)
		ll_3d -= np.nanmax(ll_3d)

	# likelihood at maximum-likelihood binaries proportion
	w0m, w1m, bm = np.unravel_index(np.nanargmax(ll_3d), ll_3d.shape)
	ll = ll_3d[..., bm]
	qm = qm_3d[..., bm]

	# base file name
	base = os.path.basename(filepath).split('.')[0]
	if len(base.split('p')[-1]) == 1:
		base = base + '0'

	# color map
	cmapBig = mpl.cm.get_cmap('Greys', 512)
	cmap = mpl.colors.ListedColormap(cmapBig(np.linspace(0, 1, 256)))
	norm = mpl.colors.Normalize(vmin=-100, vmax=np.nanmax(ll), clip=True)

	# text box text
	textstr = '\n'.join((
		r'$\log_{10}{t}=' + base.split('_')[1].replace('p','.') + '$',
		r'${\rm [M/H]}_{\rm MIST}=' + base.split('_')[2].replace('p','.').replace('m','-') + '$',
		r'$b = b_{\rm max} = ' + str(b[bm])[:5] + '$',
		r'$\sigma_{\rm \omega} = \{' + ', '.join([str(n)[:4] for n in om_sigma]) + '\}$',
		r'$w_{\rm 0, max} = ' + str(w0[w0m])[:4] + '$',
		r'$w_{\rm 1, max} = ' + str(w1[w1m])[:4] + '$',
		r'$q_{\rm max} = $' + str(qm[w0m, w1m])[:6]))

	fig, [ax, ax1] = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [8, 1]})
	plt.subplots_adjust(left=0.13, right=0.8, bottom=0.13, top=0.95)
	pcm = ax.pcolormesh(w0, w1, ll, cmap=cmap, shading='nearest', norm=norm) 
	ax.set_xlabel(r'$w_1$')
	ax.set_ylabel(r'$w_0$')
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)

	# maximum q values
	cs = ax.contour(w1, w0, qm, colors='r')
	ax.clabel(cs)
	
	# color bar
	ax1.axis('off')
	cax = fig.add_axes([0.75, 0.15, 0.03, 0.4])	
	ticks = ticker.LinearLocator(5)
	cb = fig.colorbar(mappable=pcm, ax=ax1, cax=cax, orientation='vertical', ticks=ticks, \
		format='%.1f', alpha=1.0, shrink=0.6, norm=norm)
	cb.set_label(label=r'$\ln{\cal L}$', fontsize=18, rotation=0, labelpad=25, y=0.65)

	# text box
	ax.text(1.0, 1.0, textstr, transform=ax.transAxes, fontsize=14,
			verticalalignment='top', bbox=dict(facecolor='w', alpha=1.0, edgecolor='w'))

	plt.savefig('../data/likelihoods/png/' + base + '.png', dpi=300)
	plt.close()