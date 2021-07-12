import sys
sys.path.append('..')
import config as cf
from lib import dens_util as du

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter
import pickle, glob, os
from scipy.interpolate import interp1d

mpl.rcParams['font.size'] = 14
# color map
cmap = plt.cm.get_cmap("Dark2")
# level and corresponding color bar tick labels
levels = [0.95, 0.65, 0.35]
level_text = ['95%', '65%', '35%']
if cf.mix:
	like_dir = '../data/mix/likelihoods/'
	t0_label = r'$\log_{10}{t_0}$'
	t1_label = r'$a$'
	t0_hat = r'$\log_{10}{\widehat{t}_0}$'
	t1_hat = r'$\widehat{a}$'
else:
	like_dir = '../data/likelihoods/'
	t0_label = r'$\mu_{\rm t}$'
	t1_label = r'$\sigma_{\rm t}$'
	t0_hat = r'$\widehat{\mu}_{\rm t}$'
	t1_hat = r'$\widehat{\sigma}_{\rm t}$'

def plot(x, y, p, xlabel, ylabel, textstr, filename):
	## estimate the total probability outside the boundaries
	s = 0; c = 0 # sum and count of elements along the edges
	for i in range(len(p.shape)):
		p0 = np.take(p, 0, i); p1 = np.take(p, -1, i)
		s += np.sum(p0); s += np.sum(p1)
		c += len(p0.flatten()); c += len(p1.flatten())
	# the 2D normal distribution with zero covariance is p(x) = (1 / (2 pi)) exp(-x^2 / 2)
	# and the cumulative probability within x is F(x) = 1 - exp(-x^2 / 2), so that the cumulative density
	# as a function of the probability density is F(p) = 1 - 2 pi p;
	# assume that the domain is 2D, and estimate the probability outside the boundaries as that which 
	# we expect from a 2D normal distribution outside the boundaries' average relative probability density
	out = 2 * np.pi * (s / c) / p.max()
	f = du.CI_func(p, outside=out)

	fig, [ax, ax1] = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [8, 3]})
	fig.set_figwidth(8)
	plt.subplots_adjust(left=0.13, right=0.9, bottom=0.13, top=0.95)
	cs = ax.contourf(x, y, p, [f(l) for l in levels], cmap=cmap, extend='max')
	
	# plot the region of normalization
	ax.autoscale(False)
	if ('w_' in xlabel):
		eps = 0.005
		xh_max = 1 - y.max()
		yv_max = 1 - x.max()
		ax.plot([1, 0], [0, 1], 'k--', lw=1)
	elif ('t' in xlabel):
		eps = 0.0001
		xh_max = x.max()
		yv_max = y.max()
	line = ax.plot([x.min() + eps, xh_max], [y.max(), y.max()], 'k--', lw=1)[0]
	line.set_clip_on(False)
	line = ax.plot([x.max(), x.max()], [y.min() + eps, yv_max], 'k--', lw=1)[0]
	line.set_clip_on(False)

	ax.set_xlabel(xlabel, fontsize=14)
	ax.set_ylabel(ylabel, fontsize=14)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)

	# color bar and its axis
	ax1.axis('off')
	cax = fig.add_axes([0.7, 0.17, 0.03, 0.3])	
	cb = fig.colorbar(cs, ax=ax1, cax=cax, orientation='vertical', extendfrac=0.5) # format='%.1f',
	cb.set_label(label=r'$\int{d P}$', fontsize=18, rotation=0, labelpad=25, y=0.65)
	cb.ax.set_yticklabels(level_text)  # vertically oriented colorbar
	cb.ax.invert_yaxis()

	# text box
	ax.text(1.05, 1.0, textstr, transform=ax.transAxes, fontsize=13,
			verticalalignment='top', bbox=dict(facecolor='w', alpha=1.0, edgecolor='w'))

	plt.savefig(filename, dpi=300)
	plt.close()

# filelist = list(np.sort(glob.glob('../data/likelihoods/pkl/ll*.pkl')))
filelist = list(np.sort(glob.glob(like_dir + 'pkl/ll*.pkl')))
for filepath in filelist: 
	# load the log likelihood and the maximum q
	with open(filepath, 'rb') as f:
		ll_4d, qm_4d, bm_4d, t0_ar, t1_ar, w0, w1, om_sigma = pickle.load(f)

	# get base file name
	base = os.path.basename(filepath).split('.')[0]
	if len(base.split('p')[-1]) == 1:
		base = base + '0'
	# set maximum log likelihood to zero, convert it to likelihood
	p_4d = np.exp( ll_4d - np.nanmax(ll_4d) )
	# indices of ML parameters
	w0m, w1m, t0m, t1m = np.unravel_index(np.nanargmax(p_4d), p_4d.shape)  
	# marginalize the probability over age / rotational parameters in sampled space under uniform prior;
	# use Riemann sum; treat NAN values as zeros
	p_rot = np.nansum(p_4d, axis=(2, 3))
	p_age = np.nansum(p_4d, axis=(0, 1))

	# estimates of the upper limit of the proportion of the total probability outside the sampled parameter space
	# rotational proportions: maximum on the boundary times the total possible range
	# age parameters: maximum on the boundary times the available range

	# at maximum-likelihood binaries proportion and rotational population proportions
	qm = qm_4d[w0m, w1m, ...] # maximum q
	bm = bm_4d[w0m, w1m, ...] # maximum b
	# text box text
	textstr = '\n'.join((		
		r'$A_{\rm V}=' + '%.2f' % cf.A_V + '$',
		r'${\rm [M/H]}_{\rm MIST}=' + str(cf.Z) + '$',	
	    t0_hat + ' = ' + '%.3f' % t0_ar[t0m],
	    t1_hat + ' = ' + '%.3f' % t1_ar[t1m],
		r'$\sigma_{\rm \omega} = \{' + ', '.join(['%.2f' % n for n in om_sigma]) + '\}$',
		r'$\widehat{w} = \{' + '%.2f' % w0[w0m] + ', ' + '%.2f' % (1 - w0[w0m] - w1[w1m]) +\
			', ' + '%.2f' % w1[w1m] + '\}$',
		r'$\widehat{q} = $' + '%.2f' % qm[t0m, t1m] + \
			r'$\in$[' + '%.3f' % np.nanmin(qm_4d) + ', ' + '%.3f' % np.nanmax(qm_4d) + ']',
		r'$\widehat{b} = $' + '%.2f' % bm[t0m, t1m] + \
			r'$\in$[' + '%.2f' % np.nanmin(bm_4d) + ', ' + '%.2f' % np.nanmax(bm_4d) + ']'))

	filename = like_dir + 'png/' + base + '_age_prob' + '.png'
	plot(t1_ar, t0_ar, p_age, t1_label, t0_label, textstr, filename)

	filename = like_dir + 'png/' + base + '_rotation_prob' + '.png'
	plot(w1, w0, p_rot, r'$w_2$', r'$w_0$', textstr, filename)
