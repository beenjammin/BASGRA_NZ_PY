"""
This is a place to create a python wrapper for the BASGRA fortran model in fortarn_BASGRA_NZ

 Author: Matt Hanson
 Created: 12/08/2020 9:32 AM
 """
import os
import ctypes as ct
import numpy as np
import pandas as pd
from subprocess import Popen
from copy import deepcopy
from input_output_keys import param_keys, out_cols, days_harvest_keys, matrix_weather_keys_pet, \
    matrix_weather_keys_penman
from warnings import warn

# compiled with gfortran 64,
# https://sourceforge.net/projects/mingwbuilds/files/host-windows/releases/4.8.1/64-bit/threads-posix/seh/x64-4.8.1-release-posix-seh-rev5.7z/download
# compilation code: compile_basgra_gfortran.bat

# define the dll library path
_libpath_pet = os.path.join(os.path.dirname(__file__), 'fortran_BASGRA_NZ/BASGRA_pet.DLL')
_libpath_peyman = os.path.join(os.path.dirname(__file__), 'fortran_BASGRA_NZ/BASGRA_peyman.DLL')

#_libpath_pet = r"C:\Users\BTHRO\OneDrive\Documents\GitHub\BASGRA_NZ_PY\fortran_BASGRA_NZ\BASGRA_pet.DLL"
#_libpath_peyman = r"C:\Users\BTHRO\OneDrive\Documents\GitHub\BASGRA_NZ_PY\fortran_BASGRA_NZ\BASGRA_peyman.DLL"
_bat_path = os.path.join(os.path.dirname(__file__), 'fortran_BASGRA_NZ\\compile_BASGRA_gfortran.bat')
# this is the maximum number of weather days,
# it is hard coded into fortran_BASGRA_NZ/environment.f95 line 9
_max_weather_size = 36600


def run_basgra_nz(params, matrix_weather, days_harvest, doy_irr, verbose=False,
                  dll_path='default', supply_pet=True, auto_harvest=False):
    """
    python wrapper for the fortran BASGRA code
    changes to the fortran code may require changes to this function
    runs the model for the period of the weather data
    :param params: dictionary, see input_output_keys.py, README.md, or
                   https://github.com/Komanawa-Solutions-Ltd/BASGRA_NZ_PYfor more details
    :param matrix_weather: pandas dataframe of weather data, maximum entries set in _max_weather_size in line 24
                          of this file (currently 36600)
                          see documentation for input columns at https://github.com/Komanawa-Solutions-Ltd/BASGRA_NZ_PY
                          or README.md
    :param days_harvest: days harvest dataframe must be same length as matrix_weather entries
                        see documentation for input columns at https://github.com/Komanawa-Solutions-Ltd/BASGRA_NZ_PY
                        or README.md
    :param doy_irr: a list of the days of year to irrigate on, must be integers acceptable values: (0-366)
    :param verbose: boolean, if True the fortran function prints a number of statements for debugging purposes
                   (depreciated)
    :param dll_path: path to the compiled fortran DLL to use, default was made on windows 10 64 bit, if the path does
                     not exist, this function will try to run the bat file to re-make the dll.
    :param supply_pet: boolean, if True BASGRA expects pet to be supplied, if False the parameters required to
                       calculate pet from the peyman equation are expected,
                       the version must match the DLL if dll_path != 'default'
    :param auto_harvest: boolean, if True then assumes data is formated correctly for auto harvesting, if False, then
                         assumes data is formatted for manual harvesting (e.g. previous version) and re-formats
                         internally
    :return:
    """

    assert isinstance(supply_pet, bool), 'supply_pet param must be boolean'
    assert isinstance(auto_harvest, bool), 'auto_harvest param must be boolean'

    # define DLL library path
    use_default_lib = False
    if dll_path == 'default':
        use_default_lib = True
        if supply_pet:
            dll_path = _libpath_pet

        else:
            dll_path = _libpath_peyman

    # check that library path exists
    if not os.path.exists(dll_path):
        if use_default_lib:
            # try to run the bat file
            print('dll not found, trying to run bat to create DLL:\n{}'.format(_bat_path))
            p = Popen(os.path.basename(_bat_path), cwd=os.path.dirname(_bat_path), shell=True)
            stdout, stderr = p.communicate()
            print('output of bat:\n{}\n{}'.format(stdout, stderr))
            if not os.path.exists(dll_path):
                raise EnvironmentError('default DLL path not found:\n'
                                       '{}\n'
                                       'see readme for more details:\n'
                                       '{}'.format(dll_path, os.path.dirname(__file__) + 'README.md'))
        else:
            raise EnvironmentError('DLL path not found:\n{}'.format(dll_path))

    # define expected weather keys
    if supply_pet:
        _matrix_weather_keys = matrix_weather_keys_pet
    else:
        _matrix_weather_keys = matrix_weather_keys_penman

    doy_irr = np.atleast_1d(doy_irr)
    # test the input variables
    _test_basgra_inputs(params, matrix_weather, days_harvest, verbose, _matrix_weather_keys,
                        auto_harvest, doy_irr)

    nout = len(out_cols)
    ndays = len(matrix_weather)
    nirr = len(doy_irr)

    # define output indexes before data manipulation
    out_index = matrix_weather.index

    # copy everything and ensure order is correct
    params = deepcopy(params)
    matrix_weather = deepcopy(matrix_weather.loc[:, _matrix_weather_keys])
    days_harvest = deepcopy(days_harvest.loc[:, days_harvest_keys])

    # translate manual harvest inputs into fortran format
    if not auto_harvest:
        days_harvest = _trans_manual_harv(days_harvest, matrix_weather)

    # get variables into right python types
    params = np.array([params[e] for e in param_keys]).astype(float)
    matrix_weather = matrix_weather.values.astype(float)
    days_harvest = days_harvest.values.astype(float)
    doy_irr = doy_irr.astype(np.int32)

    # manage weather size,
    weather_size = len(matrix_weather)
    if weather_size < _max_weather_size:
        temp = np.zeros((_max_weather_size - weather_size, matrix_weather.shape[1]), float)
        matrix_weather = np.concatenate((matrix_weather, temp), 0)

    y = np.zeros((ndays, nout), float)  # cannot set these to nan's or it breaks fortran

    # make pointers
    # arrays # 99% sure this works
    params_p = np.asfortranarray(params).ctypes.data_as(ct.POINTER(ct.c_double))  # 1d array, float
    matrix_weather_p = np.asfortranarray(matrix_weather).ctypes.data_as(ct.POINTER(ct.c_double))  # 2d array, float
    days_harvest_p = np.asfortranarray(days_harvest).ctypes.data_as(ct.POINTER(ct.c_double))  # 2d array, float
    y_p = np.asfortranarray(y).ctypes.data_as(ct.POINTER(ct.c_double))  # 2d array, float
    doy_irr_p = np.asfortranarray(doy_irr).ctypes.data_as(ct.POINTER(ct.c_long))

    # integers
    ndays_p = ct.pointer(ct.c_int(ndays))
    nirr_p = ct.pointer(ct.c_int(nirr))
    nout_p = ct.pointer(ct.c_int(nout))
    verb_p = ct.pointer(ct.c_bool(verbose))

    # load DLL 
    for_basgra = ct.CDLL(dll_path)

    # run BASGRA
    for_basgra.BASGRA_(params_p, matrix_weather_p, days_harvest_p, ndays_p, nout_p, nirr_p, doy_irr_p, y_p, verb_p)

    # format results
    y_p = np.ctypeslib.as_array(y_p, (ndays, nout))
    y_p = y_p.flatten(order='C').reshape((ndays, nout), order='F')
    y_p = pd.DataFrame(y_p, out_index, out_cols)
    strs = ['{}-{:03d}'.format(int(e), int(f)) for e, f in y_p[['year', 'doy']].itertuples(False, None)]
    y_p.loc[:, 'date'] = pd.to_datetime(strs, format='%Y-%j')
    y_p.set_index('date', inplace=True)

    return y_p


def _trans_manual_harv(days_harvest, matrix_weather):
    """
    translates manual harvest data to the format expected by fortran, check the details of the data in here.
    :param days_harvest: manual harvest data
    :param matrix_weather: weather data, mostly to get the right size
    :return: days_harvest (correct format for fortran code)
    """
    days_harvest = days_harvest.set_index(['year', 'doy'])
    days_harvest_out = pd.DataFrame({'year': matrix_weather.loc[:, 'year'],
                                     'doy': matrix_weather.loc[:, 'doy'],
                                     'frac_harv': np.zeros(len(matrix_weather)),  # set filler values
                                     'harv_trig': np.zeros(len(matrix_weather)) - 1,  # set flag to not harvest
                                     'harv_targ': np.zeros(len(matrix_weather)),  # set filler values
                                     'weed_dm_frac': np.zeros(len(matrix_weather))*np.nan,  # set nas, filled later
                                     'reseed_trig': np.zeros(len(matrix_weather)) -1,  # set flag to not reseed
                                     'reseed_basal': np.zeros(len(matrix_weather)),  # set filler values
                                     })
    days_harvest_out = days_harvest_out.set_index(['year', 'doy'])
    for k in set(days_harvest_keys) - {'year', 'doy'}:
        days_harvest_out.loc[days_harvest.index, k] = days_harvest.loc[:, k]

    days_harvest_out = days_harvest_out.reset_index()

    # fill the weed fraction so that DMH_WEED is always calculated

    if pd.isna(days_harvest_out.weed_dm_frac).iloc[0]:
        warn('weed_dm_frac is na for the first day of simulation, setting to first valid weed_dm_frac\n'
             'this does not affect the harvesting only the calculation of the DMH_weed variable.')

        idx = np.where(pd.notna(days_harvest_out.weed_dm_frac))[0][0]  # get first non-nan value
        id_val = pd.Series(days_harvest_out.index).iloc[0]
        days_harvest_out.loc[id_val, 'weed_dm_frac'] = days_harvest_out.loc[:, 'weed_dm_frac'].iloc[idx]

    days_harvest_out.loc[:, 'weed_dm_frac'] = days_harvest_out.loc[:, 'weed_dm_frac'].fillna(method='ffill')

    return days_harvest_out


def _test_basgra_inputs(params, matrix_weather, days_harvest, verbose, _matrix_weather_keys,
                        auto_harvest, doy_irr):
    # check parameters
    assert isinstance(verbose, bool), 'verbose must be boolean'
    assert isinstance(params, dict)
    assert set(params.keys()) == set(param_keys), 'incorrect params keys'
    assert not any([np.isnan(e) for e in params.values()]), 'params cannot have na data'

    assert params['reseed_harv_delay'] >= 1, 'harvest delay must be >=1'
    assert params['reseed_harv_delay'] % 1 < 1e5, 'harvest delay must effectively be an integer'


    # check matrix weather
    assert isinstance(matrix_weather, pd.DataFrame)
    assert set(matrix_weather.keys()) == set(_matrix_weather_keys), 'incorrect keys for matrix_weather'
    assert pd.api.types.is_integer_dtype(matrix_weather.doy), 'doy must be an integer datatype in matrix_weather'
    assert pd.api.types.is_integer_dtype(matrix_weather.year), 'year must be an integer datatype in matrix_weather'
    assert len(matrix_weather) <= _max_weather_size, 'maximum run size is {} days'.format(_max_weather_size)
    assert not matrix_weather.isna().any().any(), 'matrix_weather cannot have na values'

    # check to make sure there are no missing days in matrix_weather
    start_year = matrix_weather['year'].min()
    start_day = matrix_weather.loc[matrix_weather.year == start_year, 'doy'].min()

    stop_year = matrix_weather['year'].max()
    stop_day = matrix_weather.loc[matrix_weather.year == stop_year, 'doy'].max()

    expected_days = pd.Series(pd.date_range(start=pd.to_datetime('{}-{}'.format(start_year, start_day), format='%Y-%j'),
                                            end=pd.to_datetime('{}-{}'.format(stop_year, stop_day), format='%Y-%j')))
    check = (matrix_weather['year'].values == expected_days.dt.year.values).all() and (
            matrix_weather['doy'].values == expected_days.dt.dayofyear.values).all()
    assert check, 'the date range of matrix_weather contains missing or duplicate days'

    # check harvest data
    assert isinstance(days_harvest, pd.DataFrame)
    assert set(days_harvest.keys()) == set(days_harvest_keys), 'incorrect keys for days_harvest'
    assert pd.api.types.is_integer_dtype(days_harvest.doy), 'doy must be an integer datatype in days_harvest'
    assert pd.api.types.is_integer_dtype(days_harvest.year), 'year must be an integer datatype in days_harvest'
    assert not days_harvest.isna().any().any(), 'days_harvest cannot have na data'
    assert (days_harvest['frac_harv'] <= 1).all(), 'frac_harv cannot be greater than 1'
    if params['fixed_removal'] > 0.9:
        assert (days_harvest['harv_trig'] >=
                days_harvest['harv_targ']).all(), 'when using fixed harvest mode the harv_trig>=harv_targ'

    if auto_harvest:
        assert len(matrix_weather) == len(
            days_harvest), 'days_harvest and matrix_weather must be the same length(ndays)'

        check = (days_harvest['year'].values == expected_days.dt.year.values).all() and (
                days_harvest['doy'].values == expected_days.dt.dayofyear.values).all()
        assert check, 'the date range of matrix_weather contains missing or duplicate days'
    else:

        strs = ['{}-{:03d}'.format(int(e), int(f)) for e, f in days_harvest[['year', 'doy']].itertuples(False, None)]
        harvest_dt = pd.to_datetime(strs, format='%Y-%j')
        assert harvest_dt.min() >= expected_days.min(), 'days_harvest must start at or after first day of simulation'
        assert harvest_dt.max() <= expected_days.max(), 'days_harvest must stop at or before last day of simulation'

    # doy_irr tests
    assert isinstance(doy_irr, np.ndarray), 'doy_irr must be convertable to a numpy array'
    assert doy_irr.ndim == 1, 'doy_irr must be 1d'
    assert pd.api.types.is_integer_dtype(doy_irr), 'doy_irr must be integers'
    assert doy_irr.max() <= 366, 'entries doy_irr must not be greater than 366'
    assert doy_irr.min() >= 0, 'entries doy_irr must not be less than 0'


if __name__ == '__main__':
    pass
