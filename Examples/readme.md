# Example files
## Annual energy production (AEP) calculation
Example files are provided to calculate the annual energy production (AEP) of the IEA 740-10-MW ROWPs using [PyWake](https://topfarm.pages.windenergy.dtu.dk/PyWake/) or [FLORIS](https://github.com/NREL/floris). 
They can be run from an IDE or from the terminal (default).
From terminal and with PyWake, run the examples as:

```
python pywake_ex.py ../IEA37_Borssele_Regular_System.yaml 
python pywake_ex.py ../IEA37_Borssele_Irregular_System.yaml 
```

From terminal and with Floris, run the examples as:

```
python floris_ex.py ../IEA37_Borssele_Regular_System.yaml 
python floris_ex.py ../IEA37_Borssele_Irregular_System.yaml 
```

Differences in computed AEP between PyWake and Floris result from different
approaches for linear interpolation of the wind rose when calculating a
finer resolution for the frequency matrix.

## Irregular layout optimization
An example file is provided to generate the irregular layout optimized for AEP using the [Stochastic gradient descent for wind farm optimization](https://wes.copernicus.org/articles/8/1235/2023/wes-8-1235-2023.html) as implemented in [TOPFARM](https://topfarm.pages.windenergy.dtu.dk/TopFarm2/index.html).
Run the example from the terminal as:
```
python opt.py ../IEA37_Borssele_Irregular_System.yaml
```
 
## Layout plot
An example file is provided to plot the layouts of the regular and the irregular IEA 740-10-MW ROWPs.