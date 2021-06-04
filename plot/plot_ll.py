import sys
sys.path.append('..')
import config as cf

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter
import pickle, glob, os

mpl.rcParams['font.size'] = 12
min_ll = -10
max_ll = 0
# color map
cmapBig = mpl.cm.get_cmap('Greys', 512)
cmap = mpl.colors.ListedColormap(cmapBig(np.linspace(0, 1, 256)))
norm = mpl.colors.Normalize(vmin=min_ll, vmax=max_ll, clip=True)

def plot(x, y, qm, bm, ll, xlabel, ylabel, textstr, filename):
	fig, [ax, ax1] = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [8, 3]})
	fig.set_figwidth(8)
	plt.subplots_adjust(left=0.13, right=0.9, bottom=0.13, top=0.95)
	pcm = ax.pcolormesh(x, y, ll, cmap=cmap, shading='nearest', norm=norm) 
	ax.set_xlabel(xlabel, fontsize=14)
	ax.set_ylabel(ylabel, fontsize=14)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)

	# maximum q values
	csq = ax.contour(x, y, qm, 3, colors='r', linestyles='dashed')
	ax.clabel(csq, fmt='%.3f')
	csq.collections[0].set_label(r'$\widehat{q}_{\rm local}$')
	# maximum b values
	csb = ax.contour(x, y, bm, 3, colors='b', linestyles='dashed')
	ax.clabel(csb, fmt='%.2f')
	csb.collections[0].set_label(r'$\widehat{b}_{\rm local}$')

	ax.legend(loc='upper left', bbox_to_anchor=(1.35, 0.63))

	# color bar
	ax1.axis('off')
	cax = fig.add_axes([0.7, 0.1, 0.03, 0.4])	
	ticks = ticker.LinearLocator(5)
	cb = fig.colorbar(mappable=pcm, ax=ax1, cax=cax, orientation='vertical', ticks=ticks, \
		alpha=1.0, shrink=0.6, norm=norm) # format='%.1f',
	cb.set_label(label=r'$\ln{\cal L}$', fontsize=18, rotation=0, labelpad=25, y=0.65)

	# text box
	ax.text(1.05, 1.0, textstr, transform=ax.transAxes, fontsize=14,
			verticalalignment='top', bbox=dict(facecolor='w', alpha=1.0, edgecolor='w'))

	plt.savefig(filename, dpi=300)
	plt.close()

# filelist = list(np.sort(glob.glob('../data/likelihoods/pkl/ll*.pkl')))
filelist = list(np.sort(glob.glob('../data/likelihoods/pkl/*.pkl')))
for filepath in filelist: # for each combination of age and metallicity
	# load the log likelihood and the maximum q
	with open(filepath, 'rb') as f:
		ll_4d, qm_4d, bm_4d, t_mean, t_std, w0, w1, om_sigma = pickle.load(f)
	# get base file name
	base = os.path.basename(filepath).split('.')[0]
	if len(base.split('p')[-1]) == 1:
		base = base + '0'
	# set maximum likelihood to zero
	ll_4d -= np.nanmax(ll_4d) 	
	# indices of ML parameters
	w0m, w1m, tmm, tsm = np.unravel_index(np.nanargmax(ll_4d), ll_4d.shape)  

	# plot likelihood vs. rotational proportions, 
	# at maximum-likelihood binaries proportion and age priors
	ll = ll_4d[..., tmm, tsm] # log-likelihoods
	qm = qm_4d[..., tmm, tsm] # maximum q
	bm = bm_4d[..., tmm, tsm] # maximum b
	# text box text
	textstr = '\n'.join((
		r'$A_{\rm V}=' + '%.2f' % cf.A_V + '$',		
		r'${\rm [M/H]}_{\rm MIST}=' + str(cf.Z) + '$',
	    r'$\sigma_{\rm \omega} = \{' + ', '.join(['%.2f' % n for n in cf.om_sigma]) + '\}$',
		r'$\widehat{w} = \{' + '%.2f' % w0[w0m] + ', ' + '%.2f' % (1 - w0[w0m] - w1[w1m]) +\
			', ' + '%.2f' % w1[w1m] + '\}$',
	    r'$\mu_{\log_{10}{t}} = \widehat{\mu}_{\log_{10}{t}}=' + '%.4f' % t_mean[tmm] + '$',
	    r'$\sigma_{\log_{10}{t}} = \widehat{\sigma}_{\log_{10}{t}}=' + '%.4f' % t_std[tsm] + '$',
		r'$\widehat{q} = $' + '%.3f' % qm[w0m, w1m],
		r'$\widehat{b} = $' + '%.2f' % bm[w0m, w1m]))
	filename = '../data/likelihoods/png/' + base + '_rotation' + '.png'
	plot(cf.w1, cf.w0, qm, bm, ll, r'$w_1$', r'$w_0$', textstr, filename)

	# plot likelihood vs. age priors, 
	# at maximum-likelihood binaries proportion and rotational population proportions
	ll = ll_4d[w0m, w1m, ...] # log-likelihoods
	qm = qm_4d[w0m, w1m, ...] # maximum q
	bm = bm_4d[w0m, w1m, ...] # maximum b
	# text box text
	textstr = '\n'.join((	
		r'$A_{\rm V}=' + '%.2f' % cf.A_V + '$',	
		r'${\rm [M/H]}_{\rm MIST}=' + str(cf.Z) + '$',
		r'$\sigma_{\rm \omega} = \{' + ', '.join(['%.2f' % n for n in om_sigma]) + '\}$',
		r'$w = \widehat{w} = \{' + '%.2f' % w0[w0m] + ', ' + '%.2f' % (1 - w0[w0m] - w1[w1m]) +\
			', ' + '%.2f' % w1[w1m] + '\}$',
	    r'$\widehat{\mu}_{\log_{10}{t}}=' + '%.4f' % t_mean[tmm] + '$',
	    r'$\widehat{\sigma}_{\log_{10}{t}}=' + '%.4f' % t_std[tsm] + '$',
		r'$\widehat{q} = $' + '%.3f' % qm[tmm, tsm],
		r'$\widehat{b} = $' + '%.2f' % bm[tmm, tsm]))
	filename = '../data/likelihoods/png/' + base + '_age' + '.png'
	plot(t_std, t_mean, qm, bm, ll, r'$\sigma_{\log_{10}{t}}$', r'$\mu_{\log_{10}{t}}$', textstr, filename)