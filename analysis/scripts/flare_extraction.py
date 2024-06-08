import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
import astropy.units as u
import pandas as pd

from scipy.stats import binned_statistic
from scipy.interpolate import interp1d

#-------------------------------------------------------

if __name__ == "__main__":

    # Star parameters
    star_name = 'TIC 206544316'
    mission = 'TESS'
    sector = 1

    # Determine region to cut pre-flare region to fold (in time)
    before_trunc = 1332
    after_trunc = 1334.5

    # Determine flare starting time
    start_flare = 1334.5
    end_flare = 1336.5

    # Determine region to generate lightcurve from the pre-flare lightcurve
    # This is the lightcurve to extract/detrend the flare
    T0 = 1325.6                      # Starting time
    Tmax = 1353.1792537401057        # Where lightcurve ends

    #-------------------------------------------------------

    # Data acquisition
    # --------------------------------------------------------------------

    # Take data from TESS
    search_result_s1 = lk.search_lightcurve(star_name, mission=mission, 
                                            sector=sector)[0]

    # Assign to variable 
    lc = search_result_s1.download()

    # Get normalized lightcurve
    norm_lc = lc.normalize() 

    # Save normalized lightcurve (modulation + flare) to pandas
    lc_dict = {'Time': norm_lc.time.value,
               'Flux': norm_lc.flux.value,
               'Error': norm_lc.flux_err.value}
    
    df_lc = pd.DataFrame(data=lc_dict)

    # Cut the data to pre-flare only
    trunc = norm_lc.truncate(before=before_trunc, after=after_trunc)
    

    # Period determination
    # --------------------------------------------------------------------

    # Find period from LC periodogram
    pg = trunc.to_periodogram(method='lombscargle', minimum_period=0.1, 
                            maximum_period=0.5, oversample_factor=100)

    P = pg.period_at_max_power.value

    print('Period is = {:.2f} days'.format(P))


    # Fold pre-flare lightcurve to extract flares
    # --------------------------------------------------------------------

    # Fold the pre-flare only data
    fold = trunc.fold(period=1*P,normalize_phase=True)
    
    # Offset the phase from 0 to 1
    time = fold.time.value

    for i in range(len(time)):
        if (time[i] < 0):
            time[i] = time[i]+1

    # Save to pandas dataframe
    fold_dict = {'Phase': time,
                 'Flux': fold.flux.value,
                 'Error': fold.flux_err.value}

    df_fold = pd.DataFrame(data=fold_dict)
    df_fold = df_fold.sort_values(by=['Phase'],ignore_index=True)


    # Smoothes folded lightcurve for better extraction
    # --------------------------------------------------------------------

    # Bin folded lightcurve
    #-----------------------

    num_bins = np.linspace(0,1.0,300)
    df_bin = df_fold.groupby(pd.cut(df_fold.Phase,num_bins)).Flux.median().reset_index()

    df_bin['Center'] = df_bin['Phase'].apply(lambda x: x.mid)


    # Sigma clip binned lightcurve
    #-----------------------

    # Determine the rolling median 
    df_bin['rol_med'] = df_bin['Flux'].rolling(9, center=True).median()

    # Determine the rolling standard deviation
    df_bin['rol_std'] = df_bin['Flux'].rolling(9, center=True).std()

    # Sigma-clipped data criteria: data > local median (rol_med) + 
    # local standard deviation (rol_std). 

    # Determining the sigma-clipping criteria for 1.5 sigma
    df_bin['rol_crit'] = df_bin['rol_med'] + (1 * df_bin['rol_std'])

    # Dropping the NaN
    df_bin = df_bin.dropna()

    # Sigma-clipping for 1.5 sigma
    df_clip = df_bin.mask(df_bin['Flux'] > df_bin['rol_crit'])
    df_clip = df_clip.dropna()

    df_clip = df_clip.reset_index()


    # Generate lightcurve from folded pre-flare lightcurve
    # --------------------------------------------------------------------

    # Loop the folded lightcurve
    phase = df_clip['Center']
    fluks = df_clip['Flux']
    j = len(phase)-1
    n_i = ((Tmax - T0)/P) * j

    t = []
    f = []

    # Generate modulation-only lightcurve
    for i in range(np.int64(n_i)):
        a = np.mod(i,j)
        b = np.floor_divide(i,j)
        x = (T0-0.031) + ((b+1)*P) + (P*phase[a])
        y = fluks[a]
        
        t.append(x)
        f.append(y)
    
    mod_dict = {'Time': t,
                'Flux': f}
    
    df_mod = pd.DataFrame(data=mod_dict)


    # Extract flares
    # --------------------------------------------------------------------

    # Interpolate region
    intrp = interp1d(t,f)

    # Get flare-region 
    df_lc = df_lc[(df_lc['Time'] > start_flare) & (df_lc['Time'] < end_flare)]
    df_lc = df_lc.reset_index(drop=True)

    # Extract flares from modulation
    flare = []

    for i in range(len(df_lc['Time'])):
        fl = df_lc['Flux'][i] - intrp(df_lc['Time'][i])

        flare.append(fl)
    
    # Save flare lightcurve to pandas
    df_lc['Flux'] = flare

    # Convert time to radian
    t0 = 1334.75

    df_lc['Phase'] = (df_lc['Time'] - t0) / P
    df_lc['Radian'] = df_lc['Phase'] * 2*np.pi

    # Only consider lightcurve less than phase < 4
    df_lc = df_lc[df_lc['Phase'] < 4]


    # Drop additional flares
    #-----------------------

    # Drop flare 1
    index_flare1 = df_lc[(df_lc['Phase'] > 1.075) & (df_lc['Phase'] < 1.125) 
                            & (df_lc['Flux'] > 0.076)].index
    
    df_lc.drop(index_flare1,inplace=True)

    # Drop flare 2
    index_flare2 = df_lc[(df_lc['Phase'] > 3.3) & (df_lc['Phase'] < 3.45) 
                            & (df_lc['Flux'] > 0.02)].index
    
    df_lc.drop(index_flare2,inplace=True)

    # Making sure there is no NaN
    df_lc = df_lc.dropna()


    # Output as a CSV
    # --------------------------------------------------------------------
    df_lc.to_csv('data/extracted_flare.csv', sep=",", index=False)


    # Plot extracted flare
    # --------------------------------------------------------------------
    plt.figure(figsize=[8,4])
    plt.plot(df_lc['Phase'],df_lc['Flux'], linewidth=1, color='blue')
    plt.ylim(0,1.0)

    plt.show()