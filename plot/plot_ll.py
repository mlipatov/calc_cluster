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

def plot(x, y, q, ll, xlabel, ylabel, textstr, filename):
	fig, [ax, ax1] = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [8, 3]})
	fig.set_figwidth(8)
	plt.subplots_adjust(left=0.13, right=0.9, bottom=0.13, top=0.95)
	pcm = ax.pcolormesh(x, y, ll, cmap=cmap, shading='nearest', norm=norm) 
	ax.set_xlabel(xlabel, fontsize=14)
	ax.set_ylabel(ylabel, fontsize=14)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
	ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

	# maximum q values
	cs = ax.contour(x, y, qm, 3, colors='r')
	ax.clabel(cs, fmt='%.4f')

	# color bar
	ax1.axis('off')
	cax = fig.add_axes([0.75, 0.1, 0.03, 0.4])	
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
		ll_5d, qm_5d, t_mean, t_std = pickle.load(f)
	# get base file name
	base = os.path.basename(filepath).split('.')[0]
	if len(base.split('p')[-1]) == 1:
		base = base + '0'
	# set maximum likelihood to zero
	ll_5d -= np.nanmax(ll_5d) 	
	# indices of ML parameters
	w0m, w1m, bm, tmm, tsm = np.unravel_index(np.nanargmax(ll_5d), ll_5d.shape)  

	# plot likelihood vs. rotational proportions, 
	# at maximum-likelihood binaries proportion and age priors
	ll = ll_5d[..., bm, tmm, tsm] # log-likelihoods
	qm = qm_5d[..., bm, tmm, tsm] # maximum q
	# text box text
	textstr = '\n'.join((		
		r'${\rm [M/H]}_{\rm MIST}=' + str(cf.Z) + '$',
		r'$\sigma_{\rm \omega} = \{' + ', '.join(['%.2f' % n for n in cf.om_sigma]) + '\}$',
		r'$b = \widehat{b} = ' + '%.3f' % cf.b[bm] + '$',
	    r'$\mu_{\log_{10}{t}} = \widehat{\mu}_{\log_{10}{t}}=' + '%.3f' % t_mean[tmm] + '$',
	    r'$\sigma_{\log_{10}{t}} = \widehat{\sigma}_{\log_{10}{t}}=' + '%.3f' % t_std[tsm] + '$',
		r'$\widehat{w}_{\rm 0} = ' + '%.3f' % cf.w0[w0m] + '$',
		r'$\widehat{w}_{\rm 1} = ' + '%.3f' % cf.w1[w1m] + '$',
		r'$\widehat{q} = $' + '%.4f' % qm[w0m, w1m]))
	filename = '../data/likelihoods/png/' + base + '_rotation' + '.png'
	plot(cf.w1, cf.w0, qm, ll, r'$w_1$', r'$w_0$', textstr, filename)

	# plot likelihood vs. age priors, 
	# at maximum-likelihood binaries proportion and rotational population proportions
	ll = ll_5d[w0m, w1m, bm, ...] # log-likelihoods
	qm = qm_5d[w0m, w1m, bm, ...] # maximum q
	# text box text
	textstr = '\n'.join((		
		r'${\rm [M/H]}_{\rm MIST}=' + str(cf.Z) + '$',
		r'$\sigma_{\rm \omega} = \{' + ', '.join(['%.2f' % n for n in cf.om_sigma]) + '\}$',
		r'$b = \widehat{b} = ' + '%.3f' % cf.b[bm] + '$',
	    r'$\widehat{\mu}_{\log_{10}{t}}=' + '%.3f' % t_mean[tmm] + '$',
	    r'$\widehat{\sigma}_{\log_{10}{t}}=' + '%.3f' % t_std[tsm] + '$',
		r'$w_{\rm 0} = \widehat{w}_{\rm 0} = ' + '%.3f' % cf.w0[w0m] + '$',
		r'$w_{\rm 1} = \widehat{w}_{\rm 1} = ' + '%.3f' % cf.w1[w1m] + '$',
		r'$\widehat{q} = $' + '%.4f' % qm[tmm, tsm]))
	filename = '../data/likelihoods/png/' + base + '_age' + '.png'
	plot(t_std, t_mean, qm, ll, r'$\sigma_{\log_{10}{t}}$', r'$\mu_{\log_{10}{t}}$', textstr, filename)