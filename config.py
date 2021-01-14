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
ROI = np.array( [[19.5, 22.], [0.4, 1.0], [0., 310.]] )
volume = np.prod(np.diff(ROI, axis=-1)[:, 0]) # volume of the ROI
# region of normalization, the same as ROI, except the lower vsini boundary is negative infinity
RON = np.copy(ROI)
RON[-1][0] = -np.inf
RON[-1][1] = np.inf

# the number of standard deviations to alot for each of the two convolutions on each side
# of the ROI; the actual Gaussian kernels to be truncated at one less standard deviation on each side 
nsig = 4
# refinement factor for the grid over which the first convolution is performed
downsample = 3