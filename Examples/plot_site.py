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

# Cable and substation data
# ToDo: Save data in separate yaml file
CabReg = [(0, 2, 0), (1, 4, 0), (2, 6, 0), (3, 8, 0), (4, 5, 0), (5, 10, 0), (6, 7, 0), (7, 14, 1), (8, 9, 0), (9, 15, 0), (10, 16, 1), (11, 18, 0), (12, 13, 0), (13, 20, 0), (14, 22, 1), (15, 24, 1), (16, 17, 1), (17, 25, 2), (18, 19, 0), (19, 26, 0), (20, 21, 0), (21, 32, 1), (21, 30, 0), (22, 23, 2), (23, -1, 2), (24, 34, 1), (25, -1, 2), (26, 27, 1), (27, 36, 1), (28, 29, 0), (29, 38, 0), (29, 40, 0), (31, 44, 0), (31, 42, 0), (32, 33, 2), (33, -1, 2), (34, 46, 2), (35, -1, 2), (35, 36, 2), (37, 48, 1), (37, 38, 1), (39, 50, 1), (39, 52, 1), (41, 52, 0), (41, 54, 0), (43, 44, 0), (44, 45, 1), (45, -1, 1), (46, -1, 2), (47, -1, 2), (47, 48, 2), (49, -1, 2), (49, 50, 2), (51, 60, 1), (51, 62, 0), (53, 62, 0), (53, 64, 0), (54, 55, 0), (56, -1, 2), (56, 65, 1), (57, -1, 2), (57, 67, 2), (58, -1, 2), (58, 59, 2), (59, 60, 1), (61, 69, 0), (61, 70, 0), (63, 70, 0), (65, 66, 1), (66, 71, 0), (66, 73, 0), (67, 68, 1), (68, 69, 1), (71, 72, 0)]
CabIrr = [(0, 50, 0), (1, 34, 0), (2, 8, 1), (2, 7, 0), (3, 12, 1), (3, 15, 0), (4, 20, 0), (5, 32, 0), (5, 9, 0), (6, 35, 0), (7, 43, 0), (8, 42, 1), (9, 19, 0), (10, 24, 0), (11, 18, 0), (12, 45, 1), (13, 26, 0), (13, 14, 0), (15, 21, 0), (16, 29, 1), (16, 17, 0), (17, 24, 0), (18, 59, 0), (20, 37, 1), (20, 34, 0), (21, 66, 0), (22, 67, 1), (22, 44, 1), (23, 43, 0), (25, -1, 2), (25, 42, 2), (26, 31, 0), (27, 52, 0), (27, 38, 0), (28, -1, 2), (28, 53, 2), (29, 53, 1), (30, 49, 2), (30, 56, 1), (31, 56, 1), (32, 68, 1), (33, -1, 2), (33, 45, 2), (35, 36, 0), (36, 62, 0), (37, 51, 1), (39, -1, 2), (39, 63, 2), (40, 55, 0), (40, 57, 0), (41, -1, 2), (41, 51, 2), (44, 46, 0), (46, 50, 0), (47, 65, 0), (48, 65, 0), (49, -1, 2), (52, 59, 0), (54, -1, 2), (54, 73, 2), (55, -1, 1), (57, 72, 0), (58, 61, 1), (58, 65, 0), (59, 60, 2), (60, -1, 2), (61, 64, 1), (62, 69, 1), (63, 69, 1), (64, 71, 2), (67, 70, 2), (68, 73, 1), (70, -1, 2), (71, -1, 2)]
Subs = [497.6207, 5730.622] # km
CabName = ['Cable I$_{\mathrm{R}}$=300A','Cable I$_{\mathrm{R}}$=480A','Cable I$_{\mathrm{R}}$=655A']

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