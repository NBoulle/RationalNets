"""

@author: Nicolas Boulle
"""
# Code modified from https://github.com/maziarraissi/DeepHPMs written by Maziar Raissi

import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from plotting import newfig, savefig
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# Collect the results from ReLU, Sinusoid, Rational networks and plot the errors together

Exact_idn = np.loadtxt("Results/ReLU/exact_sol.csv", delimiter=',')
U_relu = np.loadtxt("Results/ReLU/idn_relu.csv", delimiter=',')
U_sine = np.loadtxt("Results/Sine/idn_sine.csv", delimiter=',')
U_rat = np.loadtxt("Results/Rational_3_2/idn_rat.csv", delimiter=',')

lb_idn = np.array([0.0, -20.0])
ub_idn = np.array([40.0, 20.0])

# Print mean square errors
E_relu = np.mean((Exact_idn-U_relu)**2)
E_sine = np.mean((Exact_idn-U_sine)**2)
E_rat = np.mean((Exact_idn-U_rat)**2)
print("Error ReLU = %e, Error Sine = %e, Error Rat = %e" % (E_relu, E_sine, E_rat))

# Plot the different solutions
fig, ax = newfig(0.6, 1.2)
ax.axis('off')
gs = gridspec.GridSpec(1, 1)
gs.update(top=0.9, bottom=0.1, left=0.1, right=0.9, wspace=0.7, hspace=0.5)

# ######## Exact solution #######################
ax = plt.subplot(gs[0, 0])
h = ax.imshow(Exact_idn, interpolation='nearest', cmap='jet', 
              extent=[lb_idn[0], ub_idn[0], lb_idn[1], ub_idn[1]],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

fig.colorbar(h, cax=cax)
savefig('Results/solution')

fig, ax = newfig(2.0, 0.5)
ax.axis('off')
gs = gridspec.GridSpec(1, 3)
gs.update(top=0.9, bottom=0.1, left=0.1, right=0.9, wspace=0.5)

######## ReLU error #######################
ax = plt.subplot(gs[0, 0])
h = ax.imshow(abs(Exact_idn-U_relu), interpolation='nearest', cmap='jet', 
              extent=[lb_idn[0], ub_idn[0], lb_idn[1], ub_idn[1]],
              origin='lower', aspect='auto', vmin=0.0, vmax=0.05)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

fig.colorbar(h, cax=cax)

######## Sine error #######################
ax = plt.subplot(gs[0, 1])
h = ax.imshow(abs(Exact_idn-U_sine), interpolation='nearest', cmap='jet', 
              extent=[lb_idn[0], ub_idn[0], lb_idn[1], ub_idn[1]],
              origin='lower', aspect='auto', vmin=0.0, vmax=0.01)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

fig.colorbar(h, cax=cax)

######## Rational error #######################
gs.update(top=0.8, bottom=0.2, left=0.1, right=0.9, wspace=0.5)
ax = plt.subplot(gs[0, 2])
h = ax.imshow(abs(Exact_idn-U_rat), interpolation='nearest', cmap='jet', 
              extent=[lb_idn[0], ub_idn[0], lb_idn[1], ub_idn[1]],
              origin='lower', aspect='auto', vmin=0.0, vmax=0.002)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

fig.colorbar(h, cax=cax)

# Save the figure
savefig('Results/solution_error')
