import config as cf
import load_data as ld

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
import pickle, glob, os

mpl.rcParams['font.size'] = 12
RON_kwargs = {'facecolor':'none', 'edgecolor':'grey', 'alpha':0.5, 'lw':1}
ROI_kwargs = {'facecolor':'none', 'edgecolor':'grey', 'alpha':0.5, 'lw':1, 'linestyle':'dashed'}

# patches to plot for a 2D region
def plot_region(ax, region, xind, yind, region_kwargs):
	xmina, xmaxa = ax.get_xlim()
	ymina, ymaxa = ax.get_ylim()
	y = list(region[yind]); ymin = y[0]; ymax = y[1] 
	x = list(region[xind]); xmin = x[0]; xmax = x[1]
	if xmin == -np.inf: xmin = xmina; x.remove(-np.inf)
	if xmax == np.inf: xmax = xmaxa; x.remove(np.inf)
	if ymin == -np.inf: ymin = ymina; y.remove(-np.inf)
	if ymax == np.inf: ymax = ymaxa; y.remove(np.inf)
	kwargs = region_kwargs
	ax.hlines(y, xmin, xmax, **kwargs)
	ax.vlines(x, ymin, ymax, **kwargs)

filelist = list(np.sort(glob.glob('data/densities/pkl/*.pkl')))
for filepath in filelist: # for each combination of age and metallicity
	# load the density file
	with open(filepath, 'rb') as f:
		density = pickle.load(f)

	# base file name
	base = os.path.basename(filepath).split('.')[0]
	if len(base.split('p')[-1]) == 1:
		base = base + '0'

	# marginalized probabilities
	density_cmd = density.copy()
	density_cmd.marginalize(2)
	ron = np.delete(cf.RON, 2, axis=0)
	density_cmd.normalize(ron)
	density_cmd.scale()

	density_vsini = density.copy()
	density_vsini.marginalize(1)
	ron = np.delete(cf.RON, 1, axis=0)
	density_vsini.normalize(ron)
	density_vsini.scale()

	# color map
	cmapBig = mpl.cm.get_cmap('afmhot_r', 512)
	cmap = mpl.colors.ListedColormap(cmapBig(np.linspace(0, 1, 256)))

	# text box
	textstr = '\n'.join((
	    r'$log_{10}{t}=' + base.split('_')[-2].replace('p','.') + '$',
	    r'$[M/H]_{MIST}=' + base.split('_')[-1].replace('p','.').replace('m','-') + '$' ))
	    # r'$A_V=%.2f$' % (cf.A_V, )))
	    # r'$\sigma_0=%.2f$' % (cf.std, )))

	print('Plotting...')
	for plot in ['cmd', 'vsini']:
		fig, [ax, ax1] = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [5, 1]})
		plt.subplots_adjust(left=0.13, right=0.8, bottom=0.13, top=0.95)

		if plot=='cmd':
			density_plot = density_cmd; xlab = 'c = F435W - F814W'; xp = ld.color; 
			xi = 1; cmap_lab = r'$\frac{dp}{dm\,dc}$'; cmap_max = 35; cb_format = '%.1f';
			# patches = [ plt.Rectangle((cf.RON[1][0], cf.RON[0][0]), \
			# 	cf.RON[1][1] - cf.RON[1][0], cf.RON[0][1] - cf.RON[0][0], \
			# 	facecolor='none', edgecolor='grey', alpha=0.1) ]
		elif plot=='vsini':
			density_plot = density_vsini; xlab = r'$v_r = v\,\sin{i}, \,\mathrm{km/s}$'; xp = ld.vsini; 
			xi = 2; cmap_lab = r'$\frac{dp}{dm\,dv_r}$'; cmap_max = 5e-2; cb_format = '%.0e';

		pcm = ax.pcolormesh(density_plot.obs[1], density_plot.obs[0], density_plot.dens, \
			cmap=cmap, vmin=0, vmax=cmap_max) # vmax=prob_plot.max()) #  
		ax.set_xlim(left=ld.obmin_plot[xi], right=ld.obmax_plot[xi])
		ax.set_ylim(bottom=ld.obmin_plot[0], top=ld.obmax_plot[0])
		ax.invert_yaxis()
		ax.set_ylabel('m = F555W')
		ax.set_xlabel(xlab)
		ax.scatter(xp, ld.f555w, s=1, c='k', alpha=0.25, lw=0, marker=',')
		plot_region(ax, cf.ROI, xi, 0, ROI_kwargs)
		plot_region(ax, cf.RON, xi, 0, RON_kwargs)
		ax.spines["top"].set_visible(False)
		ax.spines["right"].set_visible(False)
		
		# color bar
		ax1.axis('off')
		cax = fig.add_axes([0.7, 0.2, 0.03, 0.4])
		norm = mpl.colors.Normalize(vmin=0, vmax=cmap_max)
		ticks = ticker.LinearLocator(5)
		cb = fig.colorbar(mappable=pcm, ax=ax1, cax=cax, norm=norm, orientation='vertical', ticks=ticks, \
			format=cb_format, alpha=1.0, shrink=0.6)
		cb.set_label(label=cmap_lab, fontsize=20, rotation=0, labelpad=25, y=0.65)

		# text box
		ax.text(0.95, 0.9, textstr, transform=ax.transAxes, fontsize=14,
		        verticalalignment='top', bbox=dict(facecolor='w', alpha=1.0, edgecolor='w'))

		plt.savefig('data/densities/' + plot + '/' + base + '.png', dpi=300)
		plt.close()