# Repository of vidante2024 using MalachiteMountains

### master branch

### masterthesis

This branch contains the necessary data and scripts to reproduce results for localizing a giant flare in a scallop-shell star.

- analysis/
  - scripts/
    - data/
      - karmn_spectra/
        - *.fit (spectra of all reference stars from Gaia RVS)

      - `extracted_flare.csv` 
        - extracted flare from flare_extraction.py
      - `mw_cosi.txt`
        - posterior distribution of cos i from cos_i_mw.py
      - `reference_stars.txt`
        list of reference stars
      - `tic_206_rvs.spec.fit`
        - spectrum of TIC 206544316 from Gaia RVS
        
    - funcs/
      - *.py models from MalachiteMountains
      - `model.py` modified from MalachiteMountains to include peak bump flares

    - `cos_i_mw.py` 
      - script to generate cos i distribution from Masuda and Winn (2020) https://ui.adsabs.harvard.edu/abs/2020AJ....159...81M/abstract
    - `flare_extraction.py`
      - script to extract flares from stars with complex modulation
    - `localize.py`
      - script to localize giant flare with MCMC
      - uses data from cos_i_mw.py and flare_extraction.py
    - `localize_plot.py`
      - script for plot purposes
    - `vsini_GaiaRVS.ipynb`
      - jupyter notebook to get v sin i using cross correlation method from Reiners et. al.(2012) https://ui.adsabs.harvard.edu/abs/2012AJ....143...93R/abstract
      - documentation in progress
    

