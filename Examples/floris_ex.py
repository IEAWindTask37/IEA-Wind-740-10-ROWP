# Example function to evaluate the AEP of the IEA Wind 740-10-MW ROWPs using
# the software FLORIS. Further information is provided in the ROWP report.
#
# --------------------
# PREAMBLE
import inspect
import os
import sys
import yaml
import numpy as np
from floris.tools import FlorisInterface, WindRose
from windIO.utils.yml_utils import load_yaml

# --------------------
# INPUT
ws_sw = 1               # Wind speed step width in [m/s] for wind rose discretization
wd_sw = 1               # Wind direction step width in [deg] for wind rose discretization
write_turbine = 'on'    # 'on' or 'off'. Specifies if turbine as specified in yaml files will be written to the turbine library

# --------------------
# Path (uncomment preferred option)
# a) Run from terminal:
file_path = sys.argv[1]
# b) Run in IDE:
# file_path = 'C:/IEA_Tasks/Task_37/Borssele_WindIO/ROWP_Irregular_System.yaml'

# --------------------
# CODE
#
def WriteTurbine(turbine_name, data):
    # Function to write floris turbine file (yaml) if it does not yet exist
    # Read parameters
    HH = data['wind_farm']['turbines']['hub_height']
    RD = data['wind_farm']['turbines']['rotor_diameter']
    p = data['wind_farm']['turbines']['performance']['power_curve']['power_values']
    p_ws = data['wind_farm']['turbines']['performance']['power_curve']['power_wind_speeds']
    ct = data['wind_farm']['turbines']['performance']['Ct_curve']['Ct_values']
    ct_ws = data['wind_farm']['turbines']['performance']['Ct_curve']['Ct_wind_speeds']
    # Generate 'dummy' Cp values from electr power curve for Floris
    cp = np.array(p) / (0.5 * np.array(p_ws) ** 3 * 1.225 * (RD / 2) ** 2 * np.pi)
    cp = cp.tolist()
    # Interpolate for one wind speed vector if they are not equal
    # note: could be removed with the updated turbine file
    if not p_ws == ct_ws:
        int_speeds = np.linspace(
            np.min(np.min([p_ws + ct_ws])),
            np.max(np.max([p_ws + ct_ws])),
            10000
        )
        cps_int = np.interp(int_speeds, p_ws, cp)
        cts_int = np.interp(int_speeds, ct_ws, ct)
        # convert to list
        cps_int = cps_int.tolist()
        cts_int = cts_int.tolist()
        int_speeds = int_speeds.tolist()
    else:
        int_speeds = p_ws
        cps_int = cp
        cts_int = ct
    # Dummy values for Floris Cp / Ct curves (necessary for interpolation)
    int_speeds[0:0] = [0,int_speeds[0]-0.00001]
    cps_int[0:0] = [0,0]
    cts_int[0:0] = [0,0]
    int_speeds.extend([int_speeds[-1]+0.00001, 100])
    cps_int.extend([0,0])
    cts_int.extend([0,0])
    # Create dict with input values
    dict_file = {
        'turbine_type':turbine_name,
        'generator_efficiency':1.0,
        'hub_height':HH,
        'pP':1.88,
        'pT':1.88,
        'rotor_diameter':RD,
        'TSR':7.0,
        'ref_density_cp_ct':1.225,
        'power_thrust_table':{'power':cps_int, 'thrust':cts_int, 'wind_speed':int_speeds}
    }
    # Add turbine name as single-quoted (work around to include single quotation in yaml writer)
    class SingleQuoted(str):
        pass
    def single_quoted_presenter(dumper, data):
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")
    yaml.add_representer(SingleQuoted, single_quoted_presenter)
    dict_file.update(turbine_type = SingleQuoted(turbine_name))
    # Write turbine yaml-file
    with open(tur_path, 'w') as file:
        yaml.dump(dict_file, file, default_flow_style=False,sort_keys=False)
        
# Load data
data = load_yaml(file_path)

# Extract information
x = data['wind_farm']['layouts']['initial_layout']['coordinates']['x']
y = data['wind_farm']['layouts']['initial_layout']['coordinates']['y']
turbine_name = data['wind_farm']['turbines']['name']
cut_in = data['wind_farm']['turbines']['performance']['cutin_wind_speed']
cut_out = data['wind_farm']['turbines']['performance']['cutout_wind_speed']
turbine_power = data['wind_farm']['turbines']['performance']['rated_power']
wb_scale = data['site']['energy_resource']['wind_resource']['weibull_a']['data']
wb_shape = data['site']['energy_resource']['wind_resource']['weibull_k']['data']
wb_wd_freq = data['site']['energy_resource']['wind_resource']['sector_probability']['data']
wb_wd = data['site']['energy_resource']['wind_resource']['wind_direction']

# Check if turbine already in library, otherwise write yaml file
path = os.path.dirname(inspect.getfile(FlorisInterface))
path = path.replace('tools','turbine_library/')
tur_path = path + turbine_name + '.yaml'
if write_turbine == 'on':
    WriteTurbine(turbine_name,data)

# Discretize Weibull for WindRose floris function
# a) Format and reshape
wb_scale = np.reshape(np.array(wb_scale),(-1,1))
wb_shape = np.reshape(np.array(wb_shape),(-1,1))
wb_wd_freq = np.reshape(np.array(wb_wd_freq),(-1,1))
# b) Wind speed discretization
wb_ws = np.arange(0, 51, ws_sw)     # make sure full frequency spectrum is covered
# c) Upper and lower boundaries of wind speed bins
ws_low = np.arange(np.min(wb_ws)-ws_sw/2,np.max(wb_ws)+ws_sw/2,ws_sw)
ws_high = ws_low + ws_sw
ws_low[ws_low<0] = 0
ws_high[ws_high<0] = 0
# d) Discretize distribution for each wind direction and store in list (Weibull CDF)
freq_grid_raw = wb_wd_freq * ((1 - np.exp(-(1 / wb_scale * ws_high) ** wb_shape)) -
              (1 - np.exp(-(1 / wb_scale * ws_low) ** wb_shape)))


# Initialize wind rose
wind_rose = WindRose()
wd_grid_raw, ws_grid_raw = np.meshgrid(wb_wd,wb_ws,indexing="ij")
wind_rose.make_wind_rose_from_user_dist(
    np.array([j for i in np.reshape(wd_grid_raw,(-1,1)).tolist() for j in i]),
    np.array([j for i in np.reshape(ws_grid_raw,(-1,1)).tolist() for j in i]),
    np.array([j for i in np.reshape(freq_grid_raw,(-1,1)).tolist() for j in i]),
    wd=np.array(wb_wd),
    ws=np.array(wb_ws),
)

# Interpolate for user-defined wind direction resolution
ws_int = wb_ws
wd_int = np.arange(0, 360, wd_sw)
wd_grid, ws_grid = np.meshgrid(wd_int,ws_int,indexing="ij")
freq_grid = wind_rose.interpolate(wd_grid,ws_grid)
freq_grid = freq_grid / np.sum(freq_grid)

# Initialize floris and update values
fi = FlorisInterface("floris_ex_input.yaml")
fi.reinitialize(layout_x = x, layout_y = y, turbine_type=[turbine_name])
fi.assign_hub_height_to_ref_height()

# Update Floris and evaluate AEP
fi.reinitialize(wind_directions=wd_int,wind_speeds=ws_int)
aep = fi.get_farm_AEP(freq=freq_grid, cut_in_wind_speed=cut_in, cut_out_wind_speed=cut_out)
print('aep is %.2f GWh' % (aep/1e9))
print('(%.2f capcacity factor)' % ( aep / (turbine_power * 8760 * len(x))))

# Quantify wake losses (slows down performance):
aep_nowake = fi.get_farm_AEP(freq=freq_grid, cut_in_wind_speed=cut_in, cut_out_wind_speed=cut_out, no_wake=True)
print('(%.2f%% wake losses)' % (100 - aep / aep_nowake * 100))