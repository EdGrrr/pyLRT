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
sdata, sverb = slrt.run(verbose=True)
tdata, tverb = tlrt.run(verbose=True)
print('Done RT')

wvlticks = [(list(range(200, 1000, 100))+
             list(range(1000, 10000, 1000))+
             list(range(10000, 71000, 10000))),
            (['0.2']+['']*7+
             ['1']+['']*8+
             ['10']+['']*5+['70'])]

solsurf = scipy.interpolate.interp1d(np.log(sdata[::3, 0]), sdata[::3, 1])
sol5km = scipy.interpolate.interp1d(np.log(sdata[1::3, 0]), sdata[1::3, 1])
soltoa = scipy.interpolate.interp1d(np.log(sdata[2::3, 0]), sdata[2::3, 1])
xlocs = np.linspace(np.log(sdata[0, 0]), np.log(sdata[-1, 0]), 1000)

# Solar spectrum
plt.plot(xlocs,
         soltoa(xlocs), label='TOA Incoming')
plt.plot(xlocs,
         solsurf(xlocs), label='Surface')
plt.xticks(np.log(wvlticks[0]), wvlticks[1])
plt.xlim(np.log(wvlticks[0][0]), np.log(wvlticks[0][-1]))
plt.yticks([], [])
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'Spectral irradiance (Wm$^{-2}$$\mu$m$^{-1}$)')
plt.legend()
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches((8, 3))
fig.savefig('output/solar_spectrum.png')

plt.plot(np.log(tdata[::3, 0]), 1000*tdata[::3, 2], label='Black surface emission')
plt.plot(np.log(tdata[2::3, 0]), 1000*tdata[2::3, 2], label='TOA Outgoing')
plt.legend()
fig = plt.gcf()
fig.set_size_inches((8, 3))
fig.savefig('output/combined_spectrum.png')            
fig.clf()


pdata = planck_function(5800, wavelength=np.exp(xlocs)*1e-9) # Solar radiance
pdata = 1e-6*pdata*np.pi*(6.9e5/1.5e8)**2

plt.plot(xlocs, pdata, label='5800K')
plt.plot(xlocs, solsurf(xlocs), label='Surface')
plt.plot(xlocs, soltoa(xlocs), label='TOA Incoming')
plt.xticks(np.log(wvlticks[0]), wvlticks[1])
plt.xlim(np.log(wvlticks[0][0]), np.log(2500))
plt.yticks([], [])
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'Spectral irradiance (Wm$^{-2}$$\mu$m$^{-1}$)')
plt.legend()
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches((8, 3))
fig.savefig('output/solar_spectrum_toa.png')
