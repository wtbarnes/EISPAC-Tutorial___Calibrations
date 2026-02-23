import pathlib

import astropy.time
import numpy as np
import sunpy.time

from scipy.interpolate import interp1d
from scipy.io import readsav

from eis_calibration.eis_calib_2014 import eis_ea


def interpol_eis_ea(date, wavelength, short=False, long=False, radcal=False, quiet=False):
    date = sunpy.time.parse_time(date)
    wavelength = wavelength.to_value('AA')
    
    eis_start = astropy.time.Time('2006-10-20T10:20:00.000', format='isot', scale='utc')
    if date < eis_start:
        print('WARNING: Selected date is before the start of normal EIS science operations. Output values may be inaccurate.')

    if not short and not long:
        n_input_wave = np.size(wavelength)
        loc_short = np.where((wavelength >= 165) & (wavelength <= 213))[0]
        loc_long = np.where((wavelength >= 245) & (wavelength <= 292))[0]
        if (len(loc_short) + len(loc_long) < n_input_wave) or (len(loc_short) > 0 and len(loc_long) > 0):
            raise ValueError('ERROR: Invalid wavelength(s). Please only select values in either the short (165 - 213) or long (245 - 292) wavelength bands.')

    if short:
        wavelength = 1
    elif long:
        wavelength = 1000

    fit_ea = readsav(pathlib.Path(__file__).parent / 'fit_eis_ea_2023-05-04.sav')['fit_ea']
    # Extract the necessary data from the loaded file
    fit_dates = fit_ea.date_obs[0].astype(str)
    fit_easw = fit_ea.sw_ea[0]
    fit_ealw = fit_ea.lw_ea[0]
    sw_wave = fit_ea.sw_wave[0]
    lw_wave = fit_ea.lw_wave[0]

    ref_tai = sunpy.time.parse_time(fit_dates)

    if date < ref_tai[0]:
        if not quiet:
            print(f"WARNING: Selected date is before the first calibrated date on {fit_ea.date_obs[0]}. Returning first fit calibration")
        date = ref_tai[0]

    if date > ref_tai[-1]:
        if not quiet:
            print(f"WARNING: Selected date is after the last calibrated date on {fit_ea.date_obs[-1][-1]}. Returning last fit calibration")
        date = ref_tai[-1]

    # Select out the desired waveband
    if short or (np.size(wavelength) > 0 and np.max(wavelength) < 220):
        ref_ea = fit_easw
        ref_wave = sw_wave
    else:
        ref_ea = fit_ealw
        ref_wave = lw_wave

    # Interpolate the EA curve to the selected date and wavelength value
    n_ref_waves = len(ref_wave)
    new_ea = np.zeros(n_ref_waves)
    for w in range(n_ref_waves):
        ea_values = ref_ea[w,:]
        new_ea[w] = np.interp(date.tai_seconds, ref_tai.tai_seconds, ea_values)
        
    if not short and not long:
        out_ea = interp1d(ref_wave, new_ea, kind='cubic')(wavelength)
    else:
        wavelength = ref_wave
        out_ea = new_ea

    # Compute the radcal curve or value (if requested)
    if radcal:
        # Define unit conversion factors
        sr_factor = (725.0 / 1.496e8) ** 2
        ergs_to_photons = 6.626e-27 * 2.998e10 * 1.e8
        gain = 6.3
        phot_to_elec = 12398.5 / 3.65

        print('Returning radcal values for converting [DN/s] to [ergs/(sr cm^2 s)]')
        print('   Note: You may still need to adjust for exposure time and slitsize.')

        # Convert from [DN/s] to [photons/(arcsec^2 * cm^2 * s)]
        radcal = (wavelength * gain) / (out_ea * phot_to_elec)

        # Convert to units of [ergs/(sr cm^2 s)]
        radcal = radcal * ergs_to_photons / wavelength / sr_factor

        out_ea = radcal

    return out_ea

def calib_2023(map):
    calib_ratio_2023 = eis_ea(map.wavelength)/interpol_eis_ea(map.date, map.wavelength)
    return map * calib_ratio_2023