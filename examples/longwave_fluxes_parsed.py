'''Plots the vertically resolved longwave fluxes for a PI and 4xCO2 case'''
import matplotlib.pyplot as plt
import numpy as np
from pyLRT import RadTran, get_lrt_folder

LIBRADTRAN_FOLDER = get_lrt_folder()

tlrt = RadTran(LIBRADTRAN_FOLDER)
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
tdata, tverb = tlrt.run(verbose=True, parse=True, dims=['p'])

tlrt.options['mixing_ratio'] = 'co2 1120'
# Run the RT with 4xCO2
# Implicitly uses the parser, because the same RadTran object is used
tdata2, tverb = tlrt.run(verbose=True) 

###################################
# Plot the height-resolved fluxes #
###################################

tdata.edn.plot(y='p', label='Downwelling', c='b')
tdata.eup.plot(y='p', label='Upwelling', c='r')
(tdata.eup - tdata.edn).plot(y='p', label='Net', c='g')
plt.legend()

plt.ylim(1000, 0)
plt.ylabel('Pressure')
plt.xlim(0, 400)
plt.xlabel(r'Irradiance (Wm$^{-2}$)')

fig = plt.gcf()
fig.set_size_inches(8, 4)
# fig.savefig('output/longwave_fluxes.pdf', bbox_inches='tight')
plt.show()
fig.clf()
del fig

###############################
# Change in fluxes from 4xCO2 #
###############################

tdata.edn.plot(y='p', label='Downwelling', c='b')
tdata.eup.plot(y='p', label='Upwelling', c='r')
(tdata.eup - tdata.edn).plot(y='p', label='Net', c='g')
tdata2.edn.plot(y='p', linestyle='--', c='b')
tdata2.eup.plot(y='p', linestyle='--', c='r')
(tdata2.eup - tdata2.edn).plot(y='p', linestyle='--', c='g')
plt.legend()

plt.ylim(1000, 0)
plt.ylabel('Pressure')
plt.xlim(0, 400)
plt.xlabel(r'Irradiance (Wm$^{-2}$)')

fig = plt.gcf()
fig.set_size_inches(8, 4)
plt.show()
# fig.savefig('output/longwave_fluxes_4xco2.pdf', bbox_inches='tight')
