# Calculate Likelihoods of Cluster Parameters

These computer programs compute continuous probability densities in 3D observable space and corresponding likelihoods of cluster parameters, based on a set of stellar models in multi-dimensional space. The following instructions re-produce a published version of this analysis for magnitude and vsini data in NGC 1846.

## Preliminaries

Install git and Python, then complete the following preliminary steps to obtain the files that serve as input to the programs in this repository.

### Cluster Data



### MIST (MESA Isochrones and Stellar Tracks)

To filter the information you need from the MIST model library, first copy `parse_mist_iso.py` and `mist_txt_npy.py` into directory `mist/`, which should contain directories such as `feh_m0.15_afe_p0.0_vvcrit0.0_TP`. Then run these scripts, to obtain text file `mist_isochrones.txt` and a corresponding numpy array file `mist_grid.npy`. Move the numpy array into the `data/` directory.

```
cp prelim/parse_mist_iso.py mist/
cp prelim/mist_txt_npy.py mist/
cd mist
python parse_mist_iso.py
python mist_txt_npy.py
mv mist_txt_npy.py ../data
``` 

### PARS (Paint the Atmospheres of Rotating Stars)

First, install [PARS](https://github.com/mlipatov/paint_atmospheres). Make sure `paint_atmospheres/data` contains the `filter/` directory with the three filter files from the [SVO Filter Profile Service](http://svo2.cab.inta-csic.es/theory/fps/), such as HST_ACS_WFC.F435W.dat (and the same for F555W and F814W). Also make sure that `paint_atmospheres/data` contains `limbdark/` with the intensity files for different metallicities from [Dr. R. Kurucz's website](http://kurucz.harvard.edu/grids.html), such as `im01k2.pck`, `ip00k2.pck19`, and `ip02k2.pck`. Now run the program that produces `data/ldlist.pkl`, a list of corresponding limb darkening information files:

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

## Main Analysis

### Set Up

Go to the web page with the [latest release](https://github.com/mlipatov/calc_cluster/releases/latest), download the source code as a tar.gz file, put the file in the directory where you want to un-compress it.

Un-compress the file and go to the software's top directory.

```
tar -xf calc_cluster-x.x.x.tar.gz
cd calc_cluster-x.x.x
```

`cc/config.py`

### 

To compute probability densities and cluster likelihoods, run the following series of scripts.


## Authors

* Timothy D. Brandt
* [Mikhail Lipatov](https://github.com/mlipatov/)