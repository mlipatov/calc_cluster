# plot a histogram of observable differences
# between t = 9.1544 and t = 9.1594 at constant EEP
import sys, os, time, pickle
sys.path.append(os.path.abspath(os.path.join('..')))
import numpy as np
import config as cf

from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif", 
    "font.serif": "Computer Modern",
    "font.size": 20
})

iodir = '../../data/'
with open(iodir + 'observables/dist1.pkl', 'rb') as f: dist1 = pickle.load(f)
with open(iodir + 'observables/dist2.pkl', 'rb') as f: dist2 = pickle.load(f)

print ('Plotting observable differences between age ' + '%.4f' % dist[0]['prev'] + ' and age ' + '%.4f' % dist[0]['curr'] + '.')

# values for the following two variables obtained from printout of observables calculations
diff = dist2[0]['dist']
# diff = [[ 7.58376574e-01,  1.74280575e-01,  0.00000000e+00], [ 8.85885646e-01,  1.91864574e-01,  0.00000000e+00], [ 2.77235290e+00,  5.71733907e-01,  0.00000000e+00], [ 6.68429409e-01,  1.46631454e-01,  0.00000000e+00], [ 2.17494774e+00,  4.54713881e-01,  0.00000000e+00], [ 2.36507278e+00,  4.91535700e-01,  0.00000000e+00], [ 1.98749196e+00,  4.11641359e-01,  0.00000000e+00], [ 3.12662857e+00,  6.34827971e-01,  0.00000000e+00], [ 2.03141984e+00,  4.18898698e-01,  0.00000000e+00], [ 2.54322432e+00,  5.21063545e-01,  0.00000000e+00], [ 1.83808482e+00,  3.92767263e-01,  0.00000000e+00], [ 2.97410045e+00,  6.17405198e-01,  0.00000000e+00], [ 1.44140061e+00,  3.14368600e-01,  0.00000000e+00], [ 2.34879007e+00,  5.04426242e-01,  0.00000000e+00], [ 3.08985724e+00,  6.67025834e-01, -7.42928260e-04], [ 3.00696122e+00,  6.75291916e-01, -5.47115839e-03], [ 3.09294903e+00,  7.10989488e-01, -1.39653546e-02], [ 2.93920353e+00,  6.90181856e-01, -2.23021913e-02], [ 2.69924258e+00,  7.41722893e-01, -4.24451873e-02], [ 2.05840862e+00,  6.57044556e-01, -5.72725614e-02], [ 2.39373008e+00,  7.30760681e-01, -5.14298917e-02], [ 2.72227314e+00,  8.55672270e-01, -1.38930535e-01], [ 1.98915049e+00,  6.36176348e-01, -1.06781697e-01], [ 1.99260291e+00,  6.27199810e-01, -1.08583361e-01], [ 2.39547370e+00,  7.15001734e-01, -1.70262397e-01], [ 2.05578993e+00,  6.34297047e-01, -1.27185625e-01], [ 2.32777055e+00,  7.11356089e-01, -2.12707135e-01], [ 2.12389255e+00,  6.95384159e-01, -1.52620494e-01], [ 1.98417382e+00,  6.58696078e-01, -1.53445649e-01], [ 2.24350241e+00,  7.00942652e-01, -2.04565841e-01], [ 2.15547142e+00,  6.75567384e-01, -2.05381832e-01], [ 2.21635346e+00,  6.80764434e-01, -2.16397768e-01], [ 2.42263915e+00,  7.32566811e-01, -1.00065106e-01], [ 1.45419045e+00,  3.75479734e-01, -1.54606823e+00], [ 2.87386763e+00,  8.06553512e-01,  1.04373973e-01], [ 2.82032966e+00,  7.23951680e-01, -5.67509509e-01], [ 1.70167999e+00,  5.15276932e-01, -3.66428910e-01], [ 2.48654359e+00,  7.25552023e-01,  6.03635522e-02], [ 2.69481972e+00,  6.93473881e-01, -4.18933184e-01], [ 2.86015430e+00,  6.70111321e-01, -3.74016570e-01], [ 2.38074375e+00,  7.19873179e-01,  3.66128063e-01], [ 2.64067340e+00,  5.86621964e-01, -5.29899342e-01], [ 1.56860114e+00,  4.89141688e-01, -4.05975335e-01], [ 1.51190322e+00,  5.54432740e-01, -1.06956376e-01], [ 2.16775385e+00,  5.81860656e-01, -3.30126142e-01], [ 2.17712878e+00,  6.62540866e-01,  2.15359966e-01], [ 1.78290129e+00,  6.01148436e-01, -4.51568374e-02], [ 1.81892579e+00,  4.28672157e-01, -7.54286167e-01], [ 2.19337256e+00,  5.08848141e-01, -6.25839564e-01], [ 2.26300396e+00,  5.18983396e-01, -5.88394657e-01], [ 2.29953979e+00,  6.24636828e-01,  2.22480532e-01], [ 2.38901152e+00,  4.81964570e-01, -3.45832312e-01], [ 2.06433047e+00,  5.16187013e-01, -5.29017505e-01], [ 1.71797819e+00,  6.32707759e-01,  2.64409325e-01], [ 2.07047738e+00,  5.53045999e-01, -6.47311263e-02], [ 2.26420243e+00,  5.64036961e-01,  7.16022666e-03], [ 2.53862794e+00,  3.87889739e-01, -6.58526892e-01], [ 2.67949730e+00,  1.83238174e-01, -1.95589742e+00], [ 1.96804800e+00,  5.87955878e-01,  2.01105082e-01], [ 1.94066163e+00,  6.85258347e-01,  4.81673980e-01], [ 2.22174699e+00,  5.29879538e-01, -1.16891284e-02], [ 2.77908389e+00,  9.45204715e-02, -1.54141545e+00], [ 2.61394778e+00,  1.86666219e-01, -1.50064906e+00], [ 1.92120403e+00,  7.25928698e-01,  7.26030698e-01], [ 2.35024521e+00,  4.14264527e-01, -4.90918817e-01], [ 1.95675441e+00,  6.29861861e-01,  3.83076475e-01], [ 2.16275105e+00,  4.95857349e-01, -3.60546324e-01], [ 1.91629872e+00,  7.17096503e-01,  3.52064985e-01], [ 1.94525437e+00,  6.53813258e-01,  2.94349713e-01], [ 2.18012886e+00,  2.97051799e-01, -6.50761818e-01], [ 2.17971494e+00,  4.62932820e-01, -1.72412924e-01], [ 2.27809529e+00,  3.93044680e-01, -6.50638925e-01], [ 2.09332081e+00,  5.46453940e-01,  9.71076051e-02], [ 2.13158551e+00,  7.05812281e-01,  3.07194504e-01], [ 2.30711942e+00,  2.56074734e-01, -5.55699609e-01], [ 1.99087252e+00,  8.35061124e-01,  4.99703550e-01], [ 2.54407983e+00,  9.82558540e-02, -1.08375129e+00], [ 2.41602927e+00,  3.09617353e-01, -5.42576243e-01], [ 2.58171753e+00,  2.93548143e-01, -7.31615637e-01], [ 2.62994245e+00,  1.32777448e-01, -9.87614659e-01], [ 2.72258278e+00,  3.56452325e-02, -1.25606374e+00], [ 2.79237833e+00,  2.35630838e-01, -8.97730217e-01], [ 3.04561140e+00,  3.60628043e-01, -9.50867846e-01], [ 2.77174733e+00,  8.46921038e-01,  7.72485175e-03], [ 2.99270276e+00,  8.66267651e-01, -4.13003889e-02], [ 3.06189126e+00,  1.07751815e+00,  2.26838198e-01], [ 3.37290960e+00,  1.07586233e+00, -9.19499672e-02], [ 3.35803678e+00,  8.73671688e-01, -3.28928224e-01], [ 4.03356824e+00,  1.50274351e+00, -2.09517774e-01], [ 3.63927114e+00,  1.01841491e+00, -3.60590127e-01], [ 4.28109345e+00,  1.77954342e+00,  6.66263347e-02], [ 4.02568672e+00,  6.79294195e-01, -1.04567520e+00], [ 4.87048009e+00,  2.74679952e+00, -1.97452269e-01], [ 3.99944699e+00,  1.66213333e+00, -6.48758161e-01], [ 4.91151012e+00,  1.65147741e+00, -1.64246695e+00], [ 5.04350851e+00,  2.29352418e+00, -1.17462783e+00], [ 3.61989408e+00,  2.32186226e+00,  4.39278784e-01], [ 5.58739760e+00,  1.67388433e+00, -2.38644909e+00], [ 4.57028606e+00,  1.93945177e+00, -1.29428405e+00], [ 4.51548016e+00,  2.03072734e+00, -1.00230397e+00], [ 4.90062412e+00,  1.85715484e+00, -1.14750104e+00], [ 4.49161157e+00,  2.09473460e+00, -9.71516234e-01], [ 4.98229029e+00,  3.57870200e+00,  5.07524591e-01], [ 3.95535126e+00,  6.23219285e-02, -2.91487115e+00], [ 5.73430906e+00,  3.51649476e+00, -2.41277933e-01], [ 4.09659396e+00,  1.13425454e+00, -9.64860151e-01], [ 4.36998690e+00,  3.40734723e+00,  2.56652295e-02], [ 4.24564915e+00, -1.73713564e+00, -3.75525192e+00], [ 3.67657902e+00, -4.30719377e-02, -3.55327087e+00], [ 3.17713810e+00, -9.37208001e-01, -3.10048404e+00], [ 3.32942090e+00, -6.26324114e-01, -3.76572497e+00], [ 4.06677644e+00, -1.15789675e+00, -5.26715186e+00], [ 2.75642429e+00, -9.56052149e-01, -1.70941686e+00], [ 3.84236564e+00, -9.58447364e-01, -5.64118773e+00], [ 3.73959982e+00, -1.65016764e+00, -3.92273252e+00], [ 2.58130012e+00, -1.22302451e-01, -4.20913064e+00], [ 4.22956567e+00, -1.71159428e+00, -5.78531718e+00], [ 3.06628469e+00, -8.26103339e-01, -3.87896630e+00], [ 3.36819333e+00, -8.18255294e-01, -4.78141696e+00], [ 3.54870359e+00, -1.30718207e+00, -4.75388561e+00], [ 3.65120711e+00, -1.18994091e+00, -4.64567680e+00], [ 3.22149088e+00, -8.81221950e-01, -3.83483095e+00], [ 3.57977518e+00, -1.21182713e+00, -4.46399215e+00], [ 3.02841087e+00, -6.04928391e-01, -2.60660985e+00], [ 3.10559573e+00, -6.58186472e-01, -2.71180952e+00], [ 3.11374892e+00, -5.77707438e-01, -3.05618821e+00], [ 3.34922903e+00, -6.85535063e-01, -3.77310632e+00], [ 2.46786507e+00, -3.62618447e-02, -1.33267073e+00], [ 2.56673070e+00, -1.32753947e-01, -1.43330662e+00]]
diff_avg = np.array( [d['mean_dist'][0] for d in dist1 + dist2] )
# diff_avg = [2.7493, 1.7350, 2.0898, 1.9427, 0.2195, 2.7566, 1.3983, 1.9490, 2.6247, -0.4425, 3.1043, 1.5694, 1.7042, 1.9731, -0.3790, 2.9570, 1.6264, 1.7774, 2.2707, -0.3147, 2.7956, 1.5387, 1.5666, 1.6991, 1.4887, 2.5851, 2.2761, 1.7939, 2.0838, 2.1146, 1.9754, 2.3312, 2.8711, 1.7341, 2.3437, 2.2635, 2.2051, 2.3277, 2.4137, 2.2770, 2.1257, 2.4143, 2.3219, 2.0916, 2.4212]
# diff = np.array(diff)
# diff_avg = np.array(diff_avg)

# plot histograms for one age neighbor pair, all EEPs and observables
bins = np.linspace(-3,4,15)
majorLocator = MultipleLocator(1.0)
majorFormatter = FormatStrFormatter('%d')
minorLocator = MultipleLocator(0.5)

plt.hist(diff, bins=bins, \
    label=['magnitude', 'color', r'$v_{\rm e}\sin{i}$'], alpha=0.8)
plt.legend(loc='upper left', frameon=False)
ax = plt.gca()
ax.xaxis.set_major_locator(majorLocator)
ax.xaxis.set_major_formatter(majorFormatter)
ax.xaxis.set_minor_locator(minorLocator)
# plt.xticks(np.linspace(-3,4,8))
plt.xlabel(r'$\Delta \boldmath{x} / \boldmath{\sigma}_{\boldmath{x}}$')
plt.tight_layout()
plt.savefig(iodir + 'diff_EEP.pdf', dpi=300)
plt.close()

# plot the histogram just for magnitudes, for all age neighbor pairs
bins = np.linspace(-1,4,11)
plt.hist(diff_avg, bins=bins, histtype='step', lw=2)
ax = plt.gca()
ax.xaxis.set_major_locator(majorLocator)
ax.xaxis.set_major_formatter(majorFormatter)
ax.xaxis.set_minor_locator(minorLocator)
# plt.xticks(np.linspace(-1,5,7))
plt.xlabel(r'$\Delta m / \sigma_m$')
plt.tight_layout()
plt.savefig(iodir + 'delta_m_delta_t.pdf', dpi=300)
plt.close()