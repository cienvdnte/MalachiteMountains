# Repository of vidante2024 using MalachiteMountains

### master branch

### masterthesis

This branch contains the necessary data and scripts to reproduce results for localizing a giant flare in a scallop-shell star.

- analysis/
  - scripts/
    - data/
      - chains/
        - *.h5 (chains from mcmc.py)
      - extracted_flare.csv (extracted flare from flare_extraction.py)
      - mw_cosi.txt (posterior distribution of cos i from cos_i_mw.py)

    - output/
      - flare_extract/
        - *.png (all outputs from flare_extraction.py)
      - mcmc/
        - *.png (all outputs from cos_i_mw.py, mcmc.py, and mcmc_plot.py)
        
    - funcs/
      - *.py models from MalachiteMountains
      - `model.py` modified from MalachiteMountains to include peak bump flares

    - `cos_i_mw.py` 
      - script to generate cos i distribution from Masuda and Winn
    - `flare_extraction.py`
      - script to extract flares from stars with complex modulation
    - `mcmc.py`
      - script to localize giant flare with MCMC
      - uses data from cos_i_mw.py and flare_extraction.py
    - `mcmc_plot.py`
      - script for plot purposes
    

