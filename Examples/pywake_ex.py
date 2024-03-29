# Example function to evaluate the AEP of the IEA Wind 740-10-MW ROWPs using
# the software PyWake. Further information is provided in the ROWP report.
# 
# --------------------
# PREAMBLE
import numpy as np
import sys
import matplotlib.pyplot as plt
import xarray as xr
from py_wake.site import XRSite
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake import NOJ
from py_wake.rotor_avg_models import RotorCenter
from windIO.utils.yml_utils import load_yaml

# --------------------
# INPUT
ws_sw = 1           # Wind speed step width in [m/s] for wind rose discretization
wd_sw = 1           # Wind direction step width in [deg] for wind rose discretization
plot_power = 'on'   # 'on' or 'off', for plant power vs. wind speed and wind direction plots

# --------------------
# PATH (uncomment preferred option)
# a) Run from terminal:
file_path = sys.argv[1]
# b) Run in IDE:
# file_path = 'C:/IEA_Tasks/Task_37/Borssele_WindIO/ROWP_Irregular_System.yaml'

# --------------------
# CODE
#
# Load data from windio files
system_dat = load_yaml(file_path)
farm_dat = system_dat['wind_farm']
resource_dat = system_dat['site']['energy_resource']

# Extract site data
A = resource_dat['wind_resource']['weibull_a']
k = resource_dat['wind_resource']['weibull_k']
freq = resource_dat['wind_resource']['sector_probability']
wd = resource_dat['wind_resource']['wind_direction']
ws = resource_dat['wind_resource']['wind_speed']
TI =  resource_dat['wind_resource']['turbulence_intensity']['data']

# Get x and y positions
x = farm_dat['layouts']['initial_layout']['coordinates']['x']
y = farm_dat['layouts']['initial_layout']['coordinates']['y']

# Load turbine data
hh = farm_dat['turbines']['hub_height']
rd = farm_dat['turbines']['rotor_diameter']
rp = farm_dat['turbines']['performance']['rated_power']
p = farm_dat['turbines']['performance']['power_curve']['power_values']
p_ws = farm_dat['turbines']['performance']['power_curve']['power_wind_speeds']
ct = farm_dat['turbines']['performance']['Ct_curve']['Ct_values']
ct_ws = farm_dat['turbines']['performance']['Ct_curve']['Ct_wind_speeds']
cut_in = farm_dat['turbines']['performance']['cutin_wind_speed']
cut_out = farm_dat['turbines']['performance']['cutout_wind_speed']

# Interpolate Power/Ct curves (for case of different reference ws)
int_speeds = np.linspace(np.min(np.min([p_ws, ct_ws])), np.max(np.max([p_ws, ct_ws])), 10000)
ps_int = np.interp(int_speeds, p_ws, p)
cts_int = np.interp(int_speeds, ct_ws, ct)

# Define turbines and site in pywake
windTurbines = WindTurbine(name=farm_dat['turbines']['name'], diameter=rd, hub_height=hh, 
                      powerCtFunction=PowerCtTabular(int_speeds, ps_int, power_unit='W', ct=cts_int))
site = XRSite(
       ds=xr.Dataset(data_vars=
                        {'Sector_frequency': ('wd', freq['data']), 
                         'Weibull_A': ('wd', A['data']), 
                         'Weibull_k': ('wd', k['data']), 
                         'TI': (resource_dat['wind_resource']['turbulence_intensity']['dims'][0], TI)
                         },
                      coords={'wd': wd, 'ws': ws}))
site.interp_method = 'linear'

# Windrose discretization to evaluate in pywake
ws_py = np.arange(cut_in, cut_out+ws_sw, ws_sw)
wd_py =np.arange(0, 360, wd_sw)
TI = np.interp(ws_py, ws, TI)

# Run pywake
noj = NOJ(site, windTurbines, turbulenceModel=None, k=0.05, rotorAvgModel=RotorCenter())
sim_res = noj(x, y, time=False, ws=ws_py, wd=wd_py, TI=TI)
aep = sim_res.aep(normalize_probabilities=False).sum()
print('aep is ', xr.DataArray.to_numpy(aep), 'GWh')
print('(%.2f capcacity factor)' % ( aep / (len(x) * windTurbines.power(10000) * 8760 / 1e9)))

# Plot power vs. wind speed and wind direction
if plot_power == 'on':
    wind = [14,12,10,8,6]   # wind speeds to evaluate in pywake
    Res = []                # solution matrix
    fig, ax = plt.subplots(figsize=(2.7,2.9),subplot_kw={'projection': 'polar'})
    ax.set_theta_direction(-1)
    
    for i in range(len(wind)):
        cur_ws = [wind[i]]
        cur_TI = [TI[i]]
        cur_res = noj(x, y, time=False, ws=cur_ws, wd=np.arange(0, 360, 1), TI=cur_TI)
        Res.append(xr.DataArray.to_numpy(sum(cur_res.Power)/740e6))
        ax.plot(np.deg2rad(np.arange(0, 360, 1)),Res[i],label=str(cur_ws[0]) + ' m/s') #,color = colors[i])
    
    # decorate plot
    ax.set_theta_zero_location("N")
    plt.tick_params(axis='both', which='major', labelsize=8,pad=-3)
    plt.subplots_adjust(left=0.07, bottom=0.0, right=0.935, top=0.920, wspace=0.0, hspace=0.0)
    if file_path[-21:] == 'Irregular_System.yaml':
        plt.title('Irregular Layout', pad=7.5, fontsize=10)
    else:
        plt.title('Regular Layout', pad=7.5, fontsize=10)
    plt.ylim([0,1.01])
    ax.spines['polar'].set_visible(False)
    ax.set_xticks(np.linspace(0,2*np.pi*7/8,8))
    ax.set_xticklabels(['N', '', 'E', '', 'S', '', 'W', ''])
    legend = plt.legend(title="Wind speed",prop={'size': 8},bbox_to_anchor = (1.52, 0.55),loc="center right")
    legend.get_title().set_fontsize('9')