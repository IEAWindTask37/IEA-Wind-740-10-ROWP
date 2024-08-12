import numpy as np
import matplotlib.pyplot as plt
from windIO.utils.yml_utils import load_yaml
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# Load data
regular_system = load_yaml('../ROWP_Regular_System.yaml')
irregular_system = load_yaml('../ROWP_Irregular_System.yaml')

# Extract site and wind farm
b = regular_system['site']
regular = regular_system['wind_farm']
irregular = irregular_system['wind_farm']
bx = b['boundaries']['polygons'][0]['x']
by = b['boundaries']['polygons'][0]['y']

# Extract bathymetry data
X = np.array(regular_system['site']['Bathymetry']['x'])
Y = np.flip(np.array(regular_system['site']['Bathymetry']['y']))
Z = np.array(regular_system['site']['Bathymetry']['depth']['data'])

# Extract turbine coordinates
regx = regular['layouts']['initial_layout']['coordinates']['x']
regy = regular['layouts']['initial_layout']['coordinates']['y']
irrgx = irregular['layouts']['initial_layout']['coordinates']['x']
irrgy = irregular['layouts']['initial_layout']['coordinates']['y']

# Cable and substation data
# ToDo: Save data in separate yaml file
CabReg = regular['electrical_collection_array']['edges']
CabIrr = irregular['electrical_collection_array']['edges']

# Extract the coordinates
x_subs_reg = regular['electrical_substations']['coordinates']['x'][0]
y_subs_reg = regular['electrical_substations']['coordinates']['y'][0]


Subs = [x_subs_reg / 1000, y_subs_reg / 1000]

CabCurrentReg = regular['electrical_collection_array']['cables']['current_capacity']

CabName = [f'Cable I_R={current}A' for current in CabCurrentReg]

# Transfer from m to km
regx = [i /1000 for i in regx]
regy = [i /1000 for i in regy]
irrgx = [i /1000 for i in irrgx]
irrgy = [i /1000 for i in irrgy]
X = [i /1000 for i in X]
Y = [i /1000 for i in Y]
bx = [i /1000 for i in bx]
by = [i / 1000 for i in by]

# Kick-out values outside of boundary
xi, yi = np.meshgrid(X,Y)
polygon = Polygon(list(zip(bx, by)))
for i in range(len(xi)):
    for j in range(len(xi[0])):
        if not polygon.contains(Point(xi[i,j],yi[i,j])):
            Z[i,j] = np.nan


# Create plot
fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.7), sharey=True)

# Plot bathymetry: filled contour plots
Z[np.isclose(Z, 70)] = np.nan
CS = ax[0].contourf(xi, yi, Z, 300, cmap=plt.colormaps.get_cmap('Blues'))
ax[1].contourf(xi, yi, Z, 300, cmap=plt.colormaps.get_cmap('Blues'))
cb_ax = fig.add_axes([0.916, 0.107, 0.02, 0.775])
cb = fig.colorbar(CS, cax=cb_ax)
cb.set_label('Depth [m]',fontsize=9)
cb.ax.invert_yaxis()
cb.set_ticks([20, 25, 30, 35, 40])
cb.ax.tick_params(labelsize=8)

# Plot boundaries
bx.append(bx[0])
by.append(by[0])
ax[0].plot(bx,by,color='k',linewidth=1,label='Boundary')
ax[1].plot(bx,by,color='k',linewidth=1)
        
# Plot turbines
ax[1].scatter(irrgx, irrgy, c='darkorange', marker='2', zorder=3, linewidth=1.5)
ax[0].scatter(regx, regy, c='darkorange', marker='2',
              zorder=3, label='Turbine', linewidth=1.5)

# Plot cabling
# a) Combine turbine + subsation coordinates
AllXreg = regx + [Subs[0]]
AllYreg = regy + [Subs[1]]
AllXirrg = irrgx + [Subs[0]]
AllYirrg = irrgy + [Subs[1]]
# b) helper for plot
lw = [0.5,1,1.7]    # line width of different cable types
plot1 = 'on'        # helper to plot cable in legend only once
plot2 = 0           # helper to plot legend of cable types only once
cabplot = [0,0,0]   #            - " -
# c) go through all turbines and plot connection
for i in range(len(regx)):
    # regular
    if plot1 == 'on' and CabReg[i][2] == 1:
        ax[0].plot([AllXreg[CabReg[i][0]],AllXreg[CabReg[i][1]]],[AllYreg[CabReg[i][0]],AllYreg[CabReg[i][1]]],color='firebrick',linewidth=lw[CabReg[i][2]],label='Power Cable')
        plot1 = 'off'
    else:
        ax[0].plot([AllXreg[CabReg[i][0]],AllXreg[CabReg[i][1]]],[AllYreg[CabReg[i][0]],AllYreg[CabReg[i][1]]],color='firebrick',linewidth=lw[CabReg[i][2]])
    # irregular
    if cabplot[CabIrr[i][2]] == 0 and CabIrr[i][2] == plot2:
        ax[1].plot([AllXirrg[CabIrr[i][0]],AllXirrg[CabIrr[i][1]]],[AllYirrg[CabIrr[i][0]],AllYirrg[CabIrr[i][1]]],color='firebrick',linewidth=lw[CabIrr[i][2]],label=CabName[CabIrr[i][2]])
        cabplot[CabIrr[i][2]] = 1
        plot2 = plot2 + 1
    else:
        ax[1].plot([AllXirrg[CabIrr[i][0]],AllXirrg[CabIrr[i][1]]],[AllYirrg[CabIrr[i][0]],AllYirrg[CabIrr[i][1]]],color='firebrick',linewidth=lw[CabIrr[i][2]])

# Plot Substation
ax[0].scatter([Subs[0]],[Subs[1]],marker='s',s=7,color='k', zorder=3,label='Substation')
ax[1].scatter([Subs[0]],[Subs[1]],marker='s',s=7,color='k', zorder=3)

# Legend + labels
ax[0].legend(prop={'size': 8},loc = 'lower left')
ax[1].legend(prop={'size': 8},loc = 'lower left')
ax[0].set_ylabel('Northing [km]',fontsize=9)
for ii in range(2):
    ax[ii].set_aspect(1)
    ax[ii].set_xlabel('Easting [km]',fontsize=9)
    
# Update ticklabel
ax[0].tick_params(axis='both', which='major', labelsize=8)
ax[1].tick_params(axis='both', which='major', labelsize=8)

# Set x + y lim
ax[0].set_xlim(483.5,503.6)
ax[1].set_xlim(483.5,503.6)
ax[0].set_ylim(5715.6,5738.2)
ax[1].set_ylim(5715.6,5738.2)

# Add grid
ax[0].grid(alpha=0.3)
ax[1].grid(alpha=0.3)

# Title
ax[0].set_title('Regular Layout\nAEP = 3385.51 GWh', pad=10, fontsize=10)
ax[1].set_title('Irregular Layout\nAEP = 3429.63 GWh', pad=10, fontsize=10)

# Draw and save
plt.draw()
plt.savefig('layouts.pdf', bbox_inches='tight')
plt.subplots_adjust(left=0.087, bottom=0.035, right=0.9, top=0.955, wspace=0.075, hspace=0.2)

plt.show()