# Calculate Likelihoods of Cluster Parameters

These computer programs compute continuous probability densities in 3D observable space and corresponding likelihoods of cluster parameters, based on a set of stellar models in multi-dimensional space. Follow the instructions below to populate the `data` directory and thus re-produce the analysis for magnitude and vsini data in NGC 1846 in **Lipatov, Brandt, and Gossage (2022)**. Each set of instructions in English corresponds to the immediately following Python code. The file `data_populated.zip` is a compressed version of the populated data directory. 

## Preliminaries

Install git and Python, then complete the following preliminary steps to obtain the files that serve as input to the programs in this repository.

### MIST (MESA Isochrones and Stellar Tracks)

The MIST model library we utilize has ten distinct rotation rates. It formed the basis for earlier work in [Gossage et al 2019](https://ui.adsabs.harvard.edu/abs/2019ApJ...887..199G/abstract) 

To filter the requisite information from the MIST model library, first copy `parse_mist_iso.py` and `mist_txt_npy.py` into directory `mist/`, which should contain directories such as `feh_m0.15_afe_p0.0_vvcrit0.0_TP`. Then run these Python scripts, to obtain text file `mist_isochrones.txt` and a corresponding numpy array file `mist_grid.npy`. Move the numpy array into the `data/` directory.

```
cp prelim/parse_mist_iso.py mist/
cp prelim/mist_txt_npy.py mist/
cd mist
python parse_mist_iso.py
python mist_txt_npy.py
mv mist_txt_npy.py ../data
``` 

### PARS (Paint the Atmospheres of Rotating Stars)

Install [PARS](https://github.com/mlipatov/paint_atmospheres). Make sure `paint_atmospheres/data` contains the `filter/` directory with the three filter files from the [SVO Filter Profile Service](http://svo2.cab.inta-csic.es/theory/fps/), such as HST_ACS_WFC.F435W.dat (and the same for F555W and F814W). Also make sure that `paint_atmospheres/data` contains `limbdark/` with the intensity files for different metallicities from [Dr. R. Kurucz's website](http://kurucz.harvard.edu/grids.html), such as `im01k2.pck`, `ip00k2.pck19`, and `ip02k2.pck`. Now run the program that produces `data/ldlist.pkl`, a list of corresponding limb darkening information files.

```
cd paint_atmospheres/pa/opt
python grid_limbdark.py
```

Next, run `grid_magnitudes.py` to compute `data/pars_grid.pkl`, the PARS magnitudes on a grid of &tau;, &omega;, inclination, &gamma;, metallicity, A<sub>V</sub> reddening parameter, and filter. This can run a lot faster if one can increase the script's `sockets` variable, which corresponds to the number of cores on the computer. The output file is larger than it needs to be for a single-metallicity analysis. Move the output file from the PARS data directory to the main analysis data directory.

```
python grid_magnitudes.py
cd ../..
mv data/pars_grid.pkl ../calc_cluster/data/
```

## Setup

Go to the web page with the [release of this repository that produces the published analysis](https://github.com/mlipatov/calc_cluster/releases/), download the source code as a tar.gz file, put the file in the directory where you want to un-compress it.

Un-compress the file and go to the directory with executables.

```
tar -xf calc_cluster-x.x.x.tar.gz
cd calc_cluster-x.x.x/cc/
```

## Main Analysis

File `pseudo-code.txt` contains the pseudo-code for scripts `calc_obs.py`, `calc_dens.py`, and `calc_like.py`, which appear below. Additionally, the pseudo-code file lists the runtimes and memory output requirements for each script.

File `config.py` contains variables that are constant throughout the analysis. File `load_data.py` loads the cluster data `../data/ngc1846*.txt` and filters it. A number of scripts in this repository access variables in `config.py` and `load_data.py`.

### PARS Grid at One Metallicity

From the above grid of PARS magnitudes, compute a smaller grid at one metallicity, e.g., `../data/pars_grid_ZMm0p45.pkl`.

```
python mist_met_convert.py
```

### Observables on Model Grids

**CAUTION:** run the following command only if you have 100 GB of space on the hard drive for the output.
Compute magnitude, color, and vsini, a.k.a. the observables, on the MIST model grid. Refine the grid to make observable spacing between the models comparable to minimum instrument error. This uses file `lib/mist_util.py` and places files such as `obs_t9p0537.pkl` into `../data/observables/`. Re-run with `ages = 2` in the script for the second half of the ages.

```
python calc_obs.py
```

### Probability Densities in Observable Space

Next, compute the probability densities. This places minimum-error densities across observable space, such as `density_t9p0537.pkl`, into `../data/densities/pkl/`. The script also produces the individual data point densities, evaluated at each data point's observables, such as `../data/points/points_os060_005_015_t9p0537_9p2550.pkl`.

```
python calc_dens.py
```

### Likelihoods of Cluster Parameters

Finally, compute cluster likelihoods. This places the log-likelihoods on a grid of cluster parameters, such as `ll_m0p45_os060_005_015.pkl`, into `../data/likelihoods/pkl/`. The script also places the likelihood factors for individual data points at maximum-likelihood parameters, such as `lf_m0p45_os060_005_015.pkl`, into the same directory.

```
python calc_like.py
```

## Plots

Go to the directory with plot scripts.

```cd plot```

Each entry below consists of brief figure description, the figure's number in the published work, the script call that produces it, and the resulting figure file.

#### PARS and MIST dimensionless rotational velocities, Figure 1:
`python 01_plot_omega.py` &rarr; `../../data/omega_plot.pdf` 

#### PARS grid magnitude differences, Figure 2:
`python 02_plot_pars_diff.py` &rarr; `../../data/pars_diff.pdf`.

#### Original MIST models in observable space, Figure 4:
`python 04_plot_mist.py` &rarr; `../../data/model_grids/cvmd/mist_[cv]md_t9p1544.pdf`.

#### Probability densities in observable space, Figures 5, 6 & 7:
`python 05_plot_densities.py` &rarr; `../../data/densities/cmd/density_t9p1594_*.pdf`, `../../data/densities/vmd/density_t9p1594_*.pdf`, and `density_dist_*.pdf`.

#### De-normalization due to convolution, Figure 8:
`python 08_plot_dP.py` &rarr; `../../data/normalization/(mag|col)/dP91594_om2_mul1.pdf`.

#### Likelihood factors of individual data points, Figure 9:
`python 09_plot_lf.py` &rarr; `../../data/likelihoods/png/*_lf_m0p45_os060_005_015.pdf`.

#### Cluster parameter confidence regions, Figure 10:
`python 10_plot_prob.py` &rarr; `../../data/likelihoods/png/ll_m0p45_os060_005_015_(age|rotation)_prob.pdf`.

#### Observable distances in a refined model grid at one age, Figure 11:
`python 11_plot_diff.py` &rarr; `../../data/model_grids/png/diff_vs_Mini_t9p1544_Zm0p45.pdf`.

#### Observable distances between model grids at different ages, Figure 12:
`python 12_plot_diff_EEP.py` &rarr; `../../data/diff_EEP.pdf` & `../../data/delta_m_delta_t.pdf`.

`python 13_plot_interp_age.py` &rarr; `../../data/plots/(L|R)_EEP_interp_age.pdf`

#### Cluster parameter likelihoods, not a figure in the publication:
`python 14_plot_ll.py` &rarr; `../../data/likelihoods/png/ll_m0p45_os060_005_015_(age|rotation).pdf`.

## Acknowledgements

[Aaron Dotter](https://github.com/aarondotter) helped us understand and work with the MIST stellar evolution models. Nate Bastian and Sebastian Kamann provided the magnitude and vsini data for NGC 1846 that are described in [Kamann et al 2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.492.2177K/abstract).


## Authors

* [Mikhail Lipatov](https://github.com/mlipatov/)
* [Timothy D. Brandt](https://github.com/t-brandt)
* [Seth Gossage](https://sgossage.github.io/)