import numpy as np

cluster = 'NGC1846'
A_V = 0.26315789 # should be one of the A_V values on the PARS grid 
modulus = 18.45
Z = -0.45 # MIST metallicity 
z_str = '_Z' + str(Z).replace('-', 'm').replace('.', 'p') # metallicity string for printing

# minimum standard deviations of the observables:
# magnitude F555W, color F435W - F814W and vsini in km/s
std = np.array([0.01, 0.01*np.sqrt(2.), 10.])

# coordinate limits in color-magnitude-vsini space that select for real main sequence stars 
# close to the turn-off; the observables are
# 	0: magnitude F555W,
# 	1: color F435W - F814W,
# 	2: vsini. 
# The region of interest (ROI) will then be the intersection of the observables grid
# with the closed cube defined here.
ROI = np.array( [[19.5, 22.], [0.4, 1.0], [0., 280.]] )
norm = [True, True, False] # in which dimensions the foreground distributions are normalized on the ROI
volume = np.prod(np.diff(ROI, axis=-1)[:, 0]) # volume of the ROI
volume_cm = np.prod(np.diff(ROI, axis=-1)[:-1, 0]) # volume of the CMD ROI
v0err = 5 # standard deviation at the vsini = 0 boundary, in units of minimum standard deviation

# s, such that magnitude = s * (-2.5 * log_10(initial mass)) for a given metallicity
s = 4.6 
# maximum number of smallest observable space standard deviations in between models in each model dimension
dmax = 3.
# number of steps in the binary mass ratio r that ensures that magnitude differences between 
# adjacent values of r are mostly less than the maximum allowed number of smallest magnitude standard deviations
num_r = int((2.5 / np.log(10)) * (1 / (dmax * std[0]))) + 30 # adjust the additive term as necessary
# binary mass ratio spaced so that magnitudes are spaced evenly
r = np.linspace(0, 1, num_r)**(1 / s)
# refinement factor for the grid over which the first convolution is performed
downsample = 3
# the number of standard deviations to assume for the truncation of Gaussian kernels in 
# alotting data space for all integrations with error kernels and plotting; 
# actual Gaussian kernels will be truncated at one less deviation 
nsig = 4
# number of coarse vsini grid steps (approximately the minimum vsini errors) in the standard deviation 
# of kernels alotted for the initial convolution and subsequent de-normalization tests
denorm_err = 9
# number of coarse vsini grid steps in the standard deviation of kernels assumed for plotting
plot_err = 1

# slowest rotational population is centered on omega = 0, fastest on omega = 1
# standard deviations of the rotational populations
s_slow = 0.5 
s_middle = 0.1
s_fast = 0.1 
a = s_fast / s_slow
# medium rotating population: 
# mean is the location where the slow and fast distributions are equal
if a == 1:
	om_middle = 1./2
else:
	om_middle = ( 1 - a * np.sqrt(1 - 2*(1 - a**2)*np.log(a)*s_slow**2) ) / (1 - a**2)
om_mean = np.array([0, om_middle, 1])
om_sigma = np.array([s_slow, s_middle, s_fast])
om_str = '_os' + '_'.join([('%.2f' % n).replace('.','') for n in om_sigma])

# multiplicity populations
mult = ['unary', 'binary']