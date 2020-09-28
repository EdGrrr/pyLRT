'''Plots the vertically resolved longwave fluxes for a PI and 4xCO2 case'''
from pyLRT import RadTran, get_lrt_folder
import matplotlib.pyplot as plt
import numpy as np

tlrt = RadTran(get_lrt_folder())
tlrt.options['rte_solver'] = 'disort'
tlrt.options['source'] = 'thermal'
tlrt.options['output_user'] = 'p edir edn eup'
tlrt.options['wavelength'] = '4000 80000'
tlrt.options['output_process'] = 'integrate'

output_layers = [str(a) for a in np.arange(1010, 10, -25)]+['0.01']
tlrt.options['pressure_out'] = ' '.join(output_layers)
tlrt.options['atm_z_grid'] = ' '.join([str(a) for a in np.arange(0, 100, 0.25)])
tlrt.options['mixing_ratio'] = 'co2 280'
# Run the RT
tdata, tverb = tlrt.run(verbose=True)

tlrt.options['mixing_ratio'] = 'co2 1120'
# Run the RT with 4xCO2
tdata2, tverb = tlrt.run(verbose=True)

###################################
# Plot the height-resolved fluxes #
###################################

plt.plot(tdata[:, 2], tdata[:, 0], label='Downwelling', c='b')
plt.plot(tdata[:, 3], tdata[:, 0], label='Upwelling', c='r')
plt.plot(tdata[:, 3]-tdata[:, 2], tdata[:, 0], label='Net', c='g')
plt.legend()

plt.ylim(1000, 0)
plt.ylabel('Pressure')
plt.xlim(0, 400)
plt.xlabel(r'Irradiance (Wm$^{-2}$)')

fig = plt.gcf()
fig.set_size_inches(8, 4)
fig.savefig('output/longwave_fluxes.pdf', bbox_inches='tight')
fig.clf()
del(fig)

###############################
# Change in fluxes from 4xCO2 #
###############################

plt.plot(tdata[:, 2], tdata[:, 0], label='Downwelling', c='b')
plt.plot(tdata[:, 3], tdata[:, 0], label='Upwelling', c='r')
plt.plot(tdata[:, 3]-tdata[:, 2], tdata[:, 0], label='Net', c='g')
plt.plot(tdata2[:, 2], tdata2[:, 0], linestyle='--', c='b')
plt.plot(tdata2[:, 3], tdata2[:, 0], linestyle='--', c='r')
plt.plot(tdata2[:, 3]-tdata2[:, 2], tdata2[:, 0], linestyle='--', c='g')
plt.legend()

plt.ylim(1000, 0)
plt.ylabel('Pressure')
plt.xlim(0, 400)
plt.xlabel(r'Irradiance (Wm$^{-2}$)')

fig = plt.gcf()
fig.set_size_inches(8, 4)
fig.savefig('output/longwave_fluxes_4xco2.pdf', bbox_inches='tight')
