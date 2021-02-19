import sys
sys.path.append('..')
from lib import load_data as ld

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
import pickle, glob, os

mpl.rcParams['font.size'] = 12
ROI_kwargs = {'facecolor':'none', 'edgecolor':'grey', 'alpha':0.5, 'lw':1}
RON_kwargs = {'facecolor':'none', 'edgecolor':'blue', 'alpha':0.5, 'lw':1, 'linestyle':'dashed'}
rot_pop = ['zero', 'intermediate', 'critical']
rot_pop.reverse() # undo reverse after re-calculating densities

# patches to plot for a 2D region
# x index = 1
# y index = 0
def plot_region(ax, region, region_kwargs):
	xmina, xmaxa = ax.get_xlim()
	ymina, ymaxa = ax.get_ylim()
	y = list(region[0]); ymin = y[0]; ymax = y[1] 
	x = list(region[1]); xmin = x[0]; xmax = x[1]
	if xmin == -np.inf: xmin = xmina; x.remove(-np.inf)
	if xmax == np.inf: xmax = xmaxa; x.remove(np.inf)
	if ymin == -np.inf: ymin = ymina; y.remove(-np.inf)
	if ymax == np.inf: ymax = ymaxa; y.remove(np.inf)
	kwargs = region_kwargs
	ax.hlines(y, xmin, xmax, **kwargs)
	ax.vlines(x, ymin, ymax, **kwargs)

filelist = list(np.sort(glob.glob('../data/densities/pkl/*.pkl')))
for filepath in filelist: # for each combination of age and metallicity
	# load the density file
	with open(filepath, 'rb') as f:
		densities = pickle.load(f)
	# base file name
	base = os.path.basename(filepath).split('.')[0]
	if len(base.split('p')[-1]) == 1:
		base = base + '0'
		
	for k in range(len(densities)):
		density = densities[k][0]
		density_cmd = densities[k][1]

		density_vsini = density.copy()
		density_vsini.marginalize(1)
		density_vsini.normalize()

		# color map
		cmapBig = mpl.cm.get_cmap('afmhot_r', 512)
		cmap = mpl.colors.ListedColormap(cmapBig(np.linspace(0, 0.8, 256)))

		# text box
		textstr = '\n'.join((
		    r'$log_{10}{t}=' + base.split('_')[-2].replace('p','.') + '$',
		    r'$[M/H]_{MIST}=' + base.split('_')[-1].replace('p','.').replace('m','-') + '$',
			str(rot_pop[k]) + ' rotation'))
		    # r'$A_V=%.2f$' % (cf.A_V, )))

		print('Plotting...')
		for plot in ['cmd', 'vsini']:

			if plot=='cmd':
				density_plot = density_cmd; 
				xlab = 'c = F435W - F814W'; xp = ld.color; xi = 1; 
				cmap_lab = r'$\ln{\frac{dp}{dm\,dc}}$'; 
				cmap_min = np.log(0.01); cmap_max = np.log(35); cb_format = '%.1f';
			elif plot=='vsini':
				density_plot = density_vsini;
				xlab = r'$v_r = v\,\sin{i}, \,\mathrm{km/s}$'; xp = ld.vsini; xi = 2; 
				cmap_lab = r'$\ln{\frac{dp}{dm\,dv_r}}$'; 
				cmap_min = np.log(1e-10); cmap_max = np.log(0.2); cb_format = '%.1f';
			dens = density_plot.density();
			dens[dens == 0] = 1e-300
			dens = np.log(dens)
			norm = mpl.colors.Normalize(vmin=cmap_min, vmax=cmap_max, clip=True)

			fig, [ax, ax1] = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [5, 1]})
			plt.subplots_adjust(left=0.13, right=0.8, bottom=0.13, top=0.95)
			pcm = ax.pcolormesh(density_plot.obs[1], density_plot.obs[0], dens, cmap=cmap, norm=norm,\
				shading='nearest') 
			ax.set_xlim(left=ld.obmin_plot[xi], right=ld.obmax_plot[xi])
			ax.set_ylim(bottom=ld.obmin_plot[0], top=ld.obmax_plot[0])
			ax.invert_yaxis()
			ax.set_ylabel('m = F555W')
			ax.set_xlabel(xlab)
			ax.scatter(xp, ld.f555w, s=1, c='k', alpha=1.0, lw=0, marker=',')
			plot_region(ax, density_plot.ROI, ROI_kwargs)
			ax.spines["top"].set_visible(False)
			ax.spines["right"].set_visible(False)
			
			# color bar
			ax1.axis('off')
			cax = fig.add_axes([0.7, 0.2, 0.03, 0.4])
			
			ticks = ticker.LinearLocator(5)
			cb = fig.colorbar(mappable=pcm, ax=ax1, cax=cax, norm=norm, orientation='vertical', ticks=ticks, \
				format=cb_format, alpha=1.0, shrink=0.6)
			cb.set_label(label=cmap_lab, fontsize=18, rotation=0, labelpad=35, y=0.65)

			# text box
			ax.text(1.0, 1.0, textstr, transform=ax.transAxes, fontsize=14,
			        verticalalignment='top', bbox=dict(facecolor='w', alpha=1.0, edgecolor='w'))

			plt.savefig('../data/densities/' + plot + '/' + base + 'om' + str(k) + '.png', dpi=300)
			plt.close()