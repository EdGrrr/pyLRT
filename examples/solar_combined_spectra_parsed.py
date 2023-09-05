from pyLRT import RadTran, get_lrt_folder
from pyLRT.misc import planck_function
import matplotlib.pyplot as plt
import copy
import numpy as np
import scipy
import scipy.interpolate

LIBRADTRAN_FOLDER = get_lrt_folder()

slrt = RadTran(LIBRADTRAN_FOLDER)
slrt.options['rte_solver'] = 'disort'
slrt.options['source'] = 'solar'
slrt.options['wavelength'] = '200 2600'
slrt.options['output_user'] = 'lambda eglo eup edn edir'
slrt.options['zout'] = '0 5 TOA'
slrt.options['albedo'] = '0'
slrt.options['umu'] = '-1.0 1.0'
slrt.options['quiet'] = ''
slrt.options['sza'] = '0'

tlrt = copy.deepcopy(slrt)
tlrt.options['rte_solver'] = 'disort'
tlrt.options['source'] = 'thermal'
tlrt.options['output_user'] = 'lambda edir eup uu'
tlrt.options['wavelength'] = '2500 80000'
tlrt.options['mol_abs_param'] = 'reptran fine'
tlrt.options['sza'] = '0'

# Run the RT
print('Initial RT')
sdata, sverb = slrt.run(verbose=True, parse=True, dims=['lambda','zout'], zout=[0, 5, 120])
tdata, tverb = tlrt.run(verbose=True, parse=True, dims=['lambda','zout'], zout=[0, 5, 120])
print('Done RT')

sdata.edir.sel(zout=120).plot(label='TOA Incoming')
sdata.edir.sel(zout=0).plot(label='Surface')

plt.yticks([], [])
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'Spectral irradiance (Wm$^{-2}$$\mu$m$^{-1}$)')
plt.legend()
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches((8, 3))
# fig.savefig('output/solar_spectrum.png')

(1000*tdata).eup.sel(zout=0).plot(label='Black surface emission')
(1000*tdata).eup.sel(zout=120).plot(label='TOA Outgoing')
plt.legend()
plt.xscale('log')
fig = plt.gcf()
fig.set_size_inches((8, 3))
plt.show()
fig.clf()



pdata = planck_function(5800, wavelength=sdata.wvl*1e-9) # Solar radiance
pdata = 1e-6*pdata*np.pi*(6.9e5/1.5e8)**2

plt.plot(sdata.wvl, pdata, label='5800K')
sdata.edir.sel(zout=120).plot(label='TOA Incoming')
sdata.edir.sel(zout=0).plot(label='Surface')
plt.xscale('log')
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'Spectral irradiance (Wm$^{-2}$$\mu$m$^{-1}$)')
plt.legend()
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches((8, 3))
fig.savefig('output/solar_spectrum_toa.png')
