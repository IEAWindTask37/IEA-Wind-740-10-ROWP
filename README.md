# The IEA Wind 740-10-MW Reference Offshore Wind Plants (v0.1)
This repository provides the data defining the IEA Wind 740-10-MW ROWPs (v0.1) which are described in detail in [this report](https://www.nrel.gov/docs/fy24osti/87923.pdf).
The plants are located at the Borssele site approximately 40 km off the Dutch coast and aim to act as reference for future research projects on wind energy, representing modern offshore wind plants.
Seventy-four [IEA 10-MW Reference Wind Turbines](https://github.com/IEAWindTask37/IEA-10.0-198-RWT) are arranged in two layouts that are optimized for maximum annual energy production: one regular grid layout and one irregular layout.
For both layouts, collection system networks with minimized total cabling length are defined. 
The plants are described following the [WindIO](https://github.com/IEAWindTask37/windIO) schema, which allows for standardized representation of wind plant data.
Additional to the data, this repository contains example files to evaluate the optimized wind plant layouts for AEP using [FLORIS](https://github.com/NREL/floris) and [PyWake](https://topfarm.pages.windenergy.dtu.dk/PyWake/), the [TOPFARM](https://topfarm.pages.windenergy.dtu.dk/TopFarm2/index.html) script used to generate the irregular layout, and the code to plot the layouts.

To refer to the reference plants in your work, please cite:
> S. Kainz, J. Quick, M. Souza de Alencar, S. Sanchez Perez Moreno, K. Dykes, C. J. Bay, M. B. Zaaijer, and P. Bortolotti, *The IEA Wind 740-10-MW Reference Offshore Wind Plants*, 2024, IEA Wind Task 55, NREL Tech. Rep.

This repository provides the following main data:
* Plant:
    * Turbine coordinates for both layouts using the EPSG:25831 coordinate reference system.
    * Offshore substation coordinates using the EPSG:25831 coordinate reference system.
    * Electrical collection network (Route and cable type per section as well as cross section, capacity, and cost of the cable types)
* Turbine:
    * Rated power
    * Hub height
    * Rotor diameter
    * Rotor tilt
    * Cut-in and cut-out wind speed
    * Electric power curve
    * Thrust coefficient curve
* Site:
    * Bathymetry
    * Boundaries
    * Wind resource by means of the wind rose at hub height, TI, and shear

