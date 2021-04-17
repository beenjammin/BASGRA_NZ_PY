"""
 Author: Ben Throssell - borrowed from Matt Hanson
 Created: 13/02/2021
 """

#Notes:
# To run BASGRA we need four input files
# 1.0	params 
#			--> List of default parameters to be used.  These can be customised or choose
#				from a selection of four farm choices, waikato, scott, northland or lincoln.  
#				Comes from 'Woodward_2020_BASGRA_parModes.txt'
#
# 2.0	matrix_weather 
# 				--> Dataframe with daily weather information.  Note, BASGRA can calculate PET.
#					Inputs to provide:	year
#										doy 		(day of year, julian date)
# 										tmin		(min temp, °c)
#										tmax		(max temp, °c)
#										rain		(rainfall, mm)
# 										radn		(radiation, MJ/m²)
# 										pet			(Penman ET, mm)								
#										max_irr		(irrigation applied, mm)
# 										irr_trig	(start trigger, %)
# 										irr_targ	(stop trigger, %)	
#
# 3.0	days_harvest 
# 				--> Dataframe specifying when harvest should occur.  Can use either manual (daily time series
#					represented by booleans to specify if harvesting occurs) or automatic (specify time periods 
#					and trigger fractions, when the fraction is exceeded, the crop is harvested).
#					Inputs to provide:	year
#										doy 			(day of year, julian date)
# 										frac_harv		(fraction of the crop to harverst, %
#										harv_trig		(harvestable dry matter trigger, set to -1 for harvesting not to occur)
#										harv_targ		(harvestable dry matter target (e.g. dry matter is harvested to the target))
# 										weed_dm_frac	(fraction of dm of ryegrass to attribute to weeds)
# 										reseed_trig		(when BASAL <= reseed_trig, trigger a reseeding. if <0 then do not reseed)								
#										reseed_basal	(set BASAL = reseed_basal when reseeding.)
# iff 'opt_harvfrin'= True, the harvest fraction to remove is estimated by brent zero optimisation. This step 
# is recommended as the harvest fraction is non-linearly related to the harvest as the stem and reserve harvest fraction 
# is related to a power function.  In some test runs without estimation, target 500kg removal has 
# actually removed c. 1000kg 
#
# 4.0	doy_irr 
# 				--> A list of the days of year to irrigate on, must be integers acceptable values: (0-366)




import pandas as pd
import os, datetime, copy
import numpy as np
from pathlib import Path

CWD = Path.cwd()
from supporting_functions.woodward_2020_params import get_woodward_mean_full_params
from supporting_functions.plotting import plot_multiple_results
from basgra_python import run_basgra_nz

# specify dates (yyyy, mm, dd)
irrig_start = datetime.datetime(1972, 9, 1)
irrig_end = datetime.datetime(2020, 5, 31)

#--->	1.0:
#		reads parameters from the Lincoln farm
params = get_woodward_mean_full_params('lincoln')
#PAW is 40 mm and 140 mm

#--->	2.0:
#		reads weather parameters from Lincoln
matrix_weather = pd.read_csv(CWD/'check_basgra_python'/'test_data'/'weather_Dunluce.txt',
                             delim_whitespace=True, index_col=0,
                             header=0,
                             names=['year',
                                    'doy',
                                    'tmin',
                                    'tmax',
                                    'rain',
                                    'radn',
                                    'pet'])

#add irrigation parameters to the dataframe
matrix_weather.loc[:, 'max_irr']  = 5.5
matrix_weather.loc[:, 'irr_trig'] = 0.70
matrix_weather.loc[:, 'irr_targ'] = 1

# set start date as doy 121 2011
idx = (matrix_weather.year > 1972) | ((matrix_weather.year == 1972) & (matrix_weather.doy >= 121))
matrix_weather = matrix_weather.loc[idx].reset_index(drop=True)
# set end date as doy 120, 2017
idx = (matrix_weather.year < 2020) | ((matrix_weather.year == 2020) & (matrix_weather.doy <= 120))
matrix_weather = matrix_weather.loc[idx].reset_index(drop=True)
# # set start date and trim df
# idx = (matrix_weather.year > irrig_start.year) | ((matrix_weather.year == irrig_start.year) & (matrix_weather.doy >= irrig_end.timetuple().tm_yday))
# matrix_weather = matrix_weather.loc[idx].reset_index(drop=True)
# # set end date and trim df
# idx = (matrix_weather.year < irrig_end.year) | ((matrix_weather.year == irrig_end.year) & (matrix_weather.doy <= irrig_end.timetuple().tm_yday))
# matrix_weather = matrix_weather.loc[idx].reset_index(drop=True)


#--->	3.0:
#		reads harvest parameters from Lincoln
days_harvest = pd.read_csv(CWD/'check_basgra_python'/'test_data'/'harvest_Lincoln_0.txt',
                           delim_whitespace=True,
                           names=['year', 'doy', 'percent_harvest']
                           ).astype(int)  # floor matches what simon did.  Why???


days_harvest.loc[:, 'frac_harv'] = days_harvest.loc[:, 'percent_harvest'] / 100
days_harvest.loc[:, 'harv_trig'] = 0
days_harvest.loc[:, 'harv_targ'] = 0
days_harvest.loc[:, 'weed_dm_frac'] = 0
days_harvest.loc[:, 'reseed_trig'] = -1
days_harvest.loc[:, 'reseed_basal'] = 1
days_harvest.drop(columns=['percent_harvest'], inplace=True)

#--->	3.0:
#		reads harvest parameters from Lincoln
days_harvest_no_irrig = pd.read_csv(CWD/'check_basgra_python'/'test_data'/'harvest_no_irrig_Dunluce.txt',
                           delim_whitespace=True,
                           names=['year', 'doy', 'percent_harvest']
                           ).astype(int)  # floor matches what simon did.  Why???


days_harvest_no_irrig.loc[:, 'frac_harv'] = days_harvest_no_irrig.loc[:, 'percent_harvest'] / 100
days_harvest_no_irrig.loc[:, 'harv_trig'] = 0
days_harvest_no_irrig.loc[:, 'harv_targ'] = 0
days_harvest_no_irrig.loc[:, 'weed_dm_frac'] = 0
days_harvest_no_irrig.loc[:, 'reseed_trig'] = -1
days_harvest_no_irrig.loc[:, 'reseed_basal'] = 1
days_harvest_no_irrig.drop(columns=['percent_harvest'], inplace=True)

#--->	4.0:
#		days of irrigation
doy_no_irr = [0]
doy_irr = [i for i in range(1,120)] + [i for i in range(280,366)]
#doy_irr = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]

params['opt_harvfrin'] = 1.0
#--->	5.0:
#		run BASGRA
outputs_no_irrig = run_basgra_nz(params, matrix_weather, days_harvest_no_irrig, doy_no_irr, verbose=False, dll_path='default',
                        supply_pet=True)
outputs_no_irrig.to_csv(CWD/'no_irrigation.csv')

params_irrig = copy.deepcopy(params)
params_irrig['irr_frm_paw'] = 1.0
params_irrig['IRRIGF'] = .90  
outputs_irrig = run_basgra_nz(params_irrig, matrix_weather, days_harvest, doy_irr, verbose=False, dll_path='default',
                        supply_pet=True)
outputs_irrig.to_csv(CWD/'sub_opt_irrigation.csv')

params_irrig_optH = copy.deepcopy(params_irrig)
params_irrig_optH['opt_harvfrin'] = 1.0
#params_irrig_reseed['reseed_harv_delay'] = 120
#params_irrig_reseed['reseed_LAI'] = 3
#params_irrig_reseed['reseed_TILG2'] = 10
#params_irrig_reseed['reseed_TILG1'] = 40
#params_irrig_reseed['reseed_TILV'] = 5000
#params_irrig_reseed['reseed_CLV'] = 100
#params_irrig_reseed['reseed_CRES'] = 25
#params_irrig_reseed['reseed_CST'] = 10
#params_irrig_reseed['reseed_CSTUB'] = 0.5
# days_harvest_reseed = copy.deepcopy(days_harvest)
#days_harvest_reseed.loc[:, 'reseed_trig'] = 0.75
#days_harvest_reseed.loc[:, 'reseed_basal'] = 0.88
 
outputs_irrig_optH = run_basgra_nz(params_irrig_optH, matrix_weather, days_harvest, doy_irr, verbose=False, dll_path='default',
                        supply_pet=True)
outputs_irrig_optH.to_csv(CWD/'irrigation.csv')
# plot_multiple_results({'no_irrig': outputs_no_irrig, 'irrig':outputs_irrig,'irrig_optH':outputs_irrig_optH})
plot_multiple_results({'no_irrig': outputs_no_irrig})



''' EXTRA INFORMATION FROM README

### irrigation triggering and demand modelling (v2.0.0+) 

#### New Irrigation Process
Irrigation modelling was developed to answer questions about pasture growth rates in the face of possible irrigation
 water restribtions; therefore the irrigation has been implemented as follows:

* if the day of year is within the irrigation season (doy in doy_irr)
    * if the fraction of soil water (e.g. WAL/WAFC) including the time step modification to the soil water content
     (e.g. transpiration, rainfall, etc) are BELOW the trigger for that day
        * irrigation is applied at a rate = max(IRRIGF* amount of water needed to fill to 
        irrigation target * field capacity, max_irr on the day)  
    
This modification includes bug fixes that allowed irrigation to be negative.

#### New irrigation input/outputs
There is a new input variable: doy_irr, which is the days that irrigation can occur(1d array)
a number of inputs have been added to parameters:
* 'IRRIGF',  # fraction # fraction of irrigation to apply to bring water content up to field capacity, 
this was previously set within the fortran code
* 'irr_frm_paw',  # are irrigation trigger/target the fraction of profile available water (1/True or 
                    # the fraction of field capacity (0/False). 

new columns has been added to matrix_weather:
* 'max_irr',  # maximum irrigation available (mm/d)
* 'irr_trig',  # fraction of PAW/field (see irr_frm_paw) capacity at or below which irrigation is triggered (fraction 0-1) e.g. 0.5 
means that irrigation will only be applied when soil water content is at 1/2 field capacity
 (e.g. water holding capacity)
* 'irr_targ',  # fraction of PAW/field (see irr_frm_paw) capacity to irrigate to (fraction 0-1)

New outputs have been added:

* 'IRRIG':  # mm d-1 Irrigation,
* 'WAFC': #mm # Water in non-frozen root zone at field capacity
* 'IRR_TARG',  # irrigation Target (fraction of field capacity) to fill to, also an input variable
* 'IRR_TRIG',  # irrigation trigger (fraction of field capacity at which to start irrigating
* 'IRRIG_DEM',  # irrigation irrigation demand to field capacity * IRR_TARG # mm
'''