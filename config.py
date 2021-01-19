import numpy as np

cluster = 'NGC1846'
A_V = 0.26 
modulus = 18.45

# minimum standard deviations of the observables:
# magnitude F555W, color F435 - F814W and vsini in km/s
std = np.array([0.01, 0.01*np.sqrt(2.), 10.])

# coordinate limits in color-magnitude-vsini space that select for main sequence stars 
# close to the turn-off; the observables are
# 	0: magnitude F555W,
# 	1: color F435W - F814W,
# 	2: vsini. 
# The region of interest (ROI) will then be the intersection of the observables grid
# with the closed cube defined here.
ROI = np.array( [[19.5, 22.], [0.4, 1.0], [0., 280.]] )
volume = np.prod(np.diff(ROI, axis=-1)[:, 0]) # volume of the ROI
# whether each dimension is normalized (or if its collected)
norm = [True, True, False] 
# boundary bin sizes for collected dimensions
bbins = [[], [], [10, 10]] 
# residual standard deviations at boundaries for collected dimensions
berrs = [[], [], [np.sqrt(i**2 - 1) * std[-1] for i in [5, 3]]] 

# refinement factor for the grid over which the first convolution is performed
downsample = 3
# the number of standard deviations to assume for the truncation of Gaussian kernels in 
# alotting data space for all convolutions and plotting; 
# actual Gaussian kernels will be truncated at one less deviation 
nsig = 4
# number of coarse vsini grid steps (approximately the minimum vsini errors) in the standard deviation 
# of kernels alotted for the initial convolution and subsequent de-normalization tests
conv_err = 9
# number of coarse vsini grid steps in the standard deviation of kernels assumed for plotting
plot_err = 1