import numpy as np
from py_wake.rotor_avg_models import RotorCenter
import time
import sys
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import os
import yaml
from py_wake.site import XRSite
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.utils.gradients import autograd
from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from py_wake import NOJ, BastankhahGaussian
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.easy_drivers import EasySGDDriver, EasyScipyOptimizeDriver
from topfarm.plotting import XYPlotComp, NoPlot
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import XYBoundaryConstraint, InclusionZone, ExclusionZone
from topfarm.recorders import TopFarmListRecorder
from topfarm.constraint_components.constraint_aggregation import ConstraintAggregation
from topfarm.constraint_components.constraint_aggregation import DistanceConstraintAggregation
np.random.seed(2)

# constructor for YAML !include command
# (this is included in windio.utils)
def include_constructor(loader, node):
    filepath = loader.construct_scalar(node)
    base_dir = os.path.dirname(loader.stream.name)
    abs_filepath = os.path.join(base_dir, filepath)
    
    with open(abs_filepath, 'r') as f:
        return yaml.safe_load(f)

def includeTimeseriesNetCDF(loader, node):
    filepath = loader.construct_scalar(node)
    base_dir = os.path.dirname(loader.stream.name)
    abs_filepath = os.path.join(base_dir, filepath)
    
    timeseries = xr.open_dataset(abs_filepath)
    timeseries_dicts = [{**{'time': str(time)}, **{var: float(data_vars[var].values) for var in data_vars.keys()}}
                    for time, data_vars in timeseries.groupby('time')]
    return timeseries_dicts


def includeBathymetryNetCDF(loader, node):
    filepath = loader.construct_scalar(node)
    base_dir = os.path.dirname(loader.stream.name)
    abs_filepath = os.path.join(base_dir, filepath)

    dataset = xr.open_dataset(abs_filepath)
    bathymetry_data = {variable: list(dataset[variable].values.flatten()) for variable in dataset.variables}
    return bathymetry_data


yaml.SafeLoader.add_constructor('!includeBathymetryNetCDF', includeBathymetryNetCDF)
yaml.SafeLoader.add_constructor('!includeTimeseriesNetCDF', includeTimeseriesNetCDF)
yaml.SafeLoader.add_constructor('!include', include_constructor)


system = sys.argv[1]
#system = 'examples/plant/wind_energy_system/IEA37_case_study_3_wind_energy_system.yaml'
with open(system, "r") as stream:
    try:
        system_dat = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

farm_dat = system_dat['wind_farm']

resource_dat = system_dat['site']['energy_resource']
if 'timeseries' in resource_dat['wind_resource'].keys():
   timeseries = True
   wind_resource_timeseries = resource_dat['wind_resource']['timeseries']
   times = [d['time'] for d in wind_resource_timeseries]
   ws = [d['speed'] for d in wind_resource_timeseries]
   wd = [d['direction'] for d in wind_resource_timeseries]
   assert(len(times) == len(ws))
   assert(len(wd) == len(ws))
   site = Hornsrev1Site()
   TI = None

elif 'weibull_k' in resource_dat['wind_resource'].keys():
   A = resource_dat['wind_resource']['weibull_a']
   k = resource_dat['wind_resource']['weibull_k']
   freq = resource_dat['wind_resource']['sector_probability']
   wd = resource_dat['wind_resource']['wind_direction']
   ws = resource_dat['wind_resource']['wind_speed']
   site = XRSite(
          ds=xr.Dataset(data_vars=
                           {'Sector_frequency': ('wd', freq['data']), 
                            'Weibull_A': ('wd', A['data']), 
                            'Weibull_k': ('wd', k['data']), 
                            'TI': (resource_dat['wind_resource']['turbulence_intensity']['dims'][0], resource_dat['wind_resource']['turbulence_intensity']['data'])
                            },
                         coords={'wd': wd, 'ws': ws}))
   
   timeseries = False
   TI =  resource_dat['wind_resource']['turbulence_intensity']['data']
else:
   timeseries = False
   ws = resource_dat['wind_resource']['wind_speed']
   wd = resource_dat['wind_resource']['wind_direction']
   P = np.array(resource_dat['wind_resource']['probability']['data'])
   site = XRSite(ds=xr.Dataset(data_vars={'P': (['wd', 'ws'], P)}, coords = {'ws': ws, 'wd': wd, 'TI': resource_dat['wind_resource']['turbulence_intensity']['data']}))
   TI = resource_dat['wind_resource']['turbulence_intensity']['data']

# get x and y positions
x = farm_dat['layouts']['initial_layout']['coordinates']['x']
y = farm_dat['layouts']['initial_layout']['coordinates']['y']

# define turbine
hh = farm_dat['turbines']['hub_height']
rd = farm_dat['turbines']['rotor_diameter']
cp = farm_dat['turbines']['performance']['Cp_curve']['Cp_values']
cp_ws = farm_dat['turbines']['performance']['Cp_curve']['Cp_wind_speeds']
ct = farm_dat['turbines']['performance']['Ct_curve']['Ct_values']
ct_ws = farm_dat['turbines']['performance']['Ct_curve']['Ct_wind_speeds']
int_speeds = np.linspace(np.min(np.min([cp_ws, ct_ws])), np.max(np.max([cp_ws, ct_ws])), 10000)
cps_int = np.interp(int_speeds, cp_ws, cp)
cts_int = np.interp(int_speeds, ct_ws, ct)
windTurbines = WindTurbine(name=farm_dat['turbines']['name'], diameter=rd, hub_height=hh, 
                      powerCtFunction=PowerCtTabular(int_speeds, 0.5 * cps_int * int_speeds ** 3 * 1.225 * (rd / 2) ** 2 * np.pi, power_unit='W', ct=cts_int))

wake_model = NOJ(site, windTurbines, k=0.05, rotorAvgModel=RotorCenter())

#wind resource
dirs = np.arange(0, 360, 1) #wind directions
ws = np.arange(3, 25, 1) # wind speeds
freqs = site.local_wind(x, y, wd=dirs, ws=ws).Sector_frequency_ilk[0, :, 0]     #sector frequency
As = site.local_wind(x, y, wd=dirs, ws=ws).Weibull_A_ilk[0, :, 0]               #weibull A
ks = site.local_wind(x, y, wd=dirs, ws=ws).Weibull_k_ilk[0, :, 0]               #weibull k

# objective function and gradient function
samps = 50    #number of samples 
site.interp_method = 'linear'

#function to create the random sampling of wind speed and wind directions
def sampling():
    idx = np.random.choice(np.arange(dirs.size), samps, p=freqs)
    wd = dirs[idx]
    A = As[idx]
    k = ks[idx]
    ws = A * np.random.weibull(k)
    return wd, ws

#aep function - SGD
def aep_func(x, y, full=False, **kwargs):
    wd, ws = sampling()
    ti = np.interp(ws, np.arange(3, 25, 1), TI)
    print(len(ti), len(wd), len(ws))
    aep_sgd = wake_model(x, y, wd=wd, ws=ws, time=True, TI=ti).aep().sum().values * 1e6
    return aep_sgd

#gradient function - SGD
def aep_jac(x, y, **kwargs):
    wd, ws = sampling()
    ti = np.interp(ws, np.arange(3, 25, 1), TI)
    jx, jy = wake_model.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], x=x, y=y, ws=ws, TI=ti, wd=wd, time=True)
    daep_sgd = np.array([np.atleast_2d(jx), np.atleast_2d(jy)]) * 1e6
    return daep_sgd

#aep function - SLSQP
def aep_func2(x, y, **kwargs):
    wd = np.arange(0, 360, 1)
    #ws = np.arange(3, 25, 1)
    aep_slsqp = wake_model(x, y, wd=wd, ws=ws, TI=TI).aep().sum().values * 1e6
    return aep_slsqp

#gradient function - SLSQP
def aep_jac2(x, y, **kwargs):
    wd = np.arange(0, 360, 1)
   # ws = np.arange(3, 25, 1)
    jx, jy = wake_model.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], x=x, y=y, TI=TI, wd=wd, time=False)
    daep_slsqp = np.array([np.atleast_2d(jx), np.atleast_2d(jy)]) * 1e6
    return daep_slsqp

n_wt = len(x)
b = system_dat['site']
boundary = np.array([b['boundaries']['polygons'][0]['x'], b['boundaries']['polygons'][1]['y']]).T

#aep component - SGD
aep_comp = CostModelComponent(input_keys=['x','y'], n_wt=n_wt, cost_function=aep_func, objective=True, cost_gradient_function=aep_jac, maximize=True)

#aep component - SLSQP
aep_comp2 = CostModelComponent(input_keys=['x','y'], n_wt=n_wt, cost_function=aep_func2, objective=True, cost_gradient_function=aep_jac2, maximize=True)

cost_comps = [aep_comp2, aep_comp]

min_spacing_m = 2 * windTurbines.diameter()  #minimum inter-turbine spacing in meters
constraint_comp = XYBoundaryConstraint([InclusionZone(boundary)], 'multi_polygon')

#constraints
constraints = [[SpacingConstraint(min_spacing_m), constraint_comp],
               DistanceConstraintAggregation([SpacingConstraint(min_spacing_m), constraint_comp],n_wt, min_spacing_m, windTurbines)]

#driver specs
driver_names = ['SLSQP', 'SGD_again']
drivers = [EasyScipyOptimizeDriver(maxiter=200, tol=1e-3),
           EasySGDDriver(maxiter=10000, learning_rate=windTurbines.diameter(), max_time=1008000, gamma_min_factor=0.1, speedupSGD=True, sgd_thresh=0.05)]

driver_no = 1    #SGD driver
ec = [10,1]      #expected cost for SLSQP (10) and SGD (1) drivers

x0 = x
y0 = y
tf = TopFarmProblem(
        design_vars = {'x':x0, 'y':y0},         
        cost_comp = cost_comps[driver_no],    
        constraints = constraints[driver_no], 
        driver = drivers[driver_no],
        plot_comp = NoPlot(),
        expected_cost = ec[driver_no]
        )

if 1:
    tic = time.time()
    cost, state, recorder = tf.optimize()
    toc = time.time()
    print('Optimization with SGD took: {:.0f}s'.format(toc-tic), ' with a total constraint violation of ', recorder['sgd_constraint'][-1])
    recorder.save(f'{driver_names[driver_no]}')

