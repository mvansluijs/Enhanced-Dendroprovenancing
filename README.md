## Introduction
The scripts in this repository are a result of the research into enhanched dendroprovenancing using gridded environmental variables. Scripts are expected to be ran in sequence in ascending order. For further documentation we refer to the inline comments

## Python package requirements
astropy
fiona
gdal
joblib
matplotlib
netCDF4
numpy
pandas
rasterio
scipy
shapely
sklearn
sklearn_quantile
statsmodels
tqdm

## System requirements
\>16gb RAM
~50gb storage space (depending on the number of chronologies)

## Folder Structure
Prepare the following folders in the same directory as the python scripts:
```none
airs
├── 0_1a_trw_download
├── 0_1b_trw_raw
├── 0_2_CRU_TS
├── 0_3_worldclim
├── 0_4_soilgrids
│   ├── source
│   ├── prepared
├── 0_5_species_distribution
├── 1_1_prepared_trw
├── 1_2_detrended_trw
├── 1_3_chronologies
├── 2_1_random_forests
├── 2_2_modelled_chronologies
├── 3_1_year_known_out
├── 3_2_both_unknown_out
├── 4_0_general_outputs
├── temp
```

## Data requirements
# ITRDB: 
Download raw TRW data for tree species of interest from ITRDB, to replicate the  case study, download Quercus robur data with temporal overlap for the period 1901-2022. Optional: place raw download .zip in `0_1a_trw_download`. Unzip and place all files in `measurements` folder(s) into `0_1b_trw_raw`.

# CRU TS:  
Download CRU TS data, and place directly into `0_2_CRU_TS`. The folder should contain four files, one for each variable (pre, tmn, tmp, tmx). Example filename for pre: `cru_ts4.06.1901.2021.pre.dat`.

# Worldclim: 
Download Worldclim data, and place directly into `0_3_worldclim`. The folder should contain four folders, one for each variable (prec, tavg, tmax, tmin). Example foldername for prec: `wc2.1_30s_prec`.

# Soilgrids: 
Download Soilgrids data, and place into `0_4_soilgrids/prepared`. The folder should contain five TIF files (resolution: 43200x21600), one for each variable (clay, nitrogen, sand, silt, soc). Example filename for clay: `clay_15_30cm`.

# Species distribution:
Download or create species distribution data and place directly into `0_5_species_distribution`. To replicate the case study, download from https://data.mendeley.com/datasets/hr5h2hcgg4 and copy all `Quercus_robur_plg_clip` files.
