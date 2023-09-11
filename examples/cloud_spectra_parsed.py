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

# Add in a cloud
slrt_cld = copy.deepcopy(slrt)
slrt_cld.cloud = {'z': np.array([4, 3.7]),
                  'lwc': np.array([0, 0.5]),
                  're': np.array([0, 20])}

tlrt_cld = copy.deepcopy(tlrt)
tlrt_cld.cloud = {'z': np.array([4, 3.7]),
                  'lwc': np.array([0, 0.5]),
                  're': np.array([0, 20])}

# Run the RT
print('Initial RT')
sdata, sverb = slrt.run(verbose=True, parse=True, dims=['lambda','zout'], zout=[0, 5, 120])
tdata, tverb = tlrt.run(verbose=True, parse=True, parser=slrt.parser)
print('Cloud RT')
tcdata, tcverb = tlrt_cld.run(verbose=True, parse=True, parser=slrt.parser)
scdata, scverb = slrt_cld.run(verbose=True, parse=True, parser=slrt.parser)
print('Done RT')

tclearsurf = scipy.interpolate.interp1d(np.log(tdata.sel(zout=0).wvl), tdata.sel(zout=0).eup)
tcleartoa = scipy.interpolate.interp1d(np.log(tdata.sel(zout=120).wvl), tdata.sel(zout=120).eup)
tcldtoa = scipy.interpolate.interp1d(np.log(tcdata.sel(zout=0).wvl), tcdata.sel(zout=120).eup)

xtlocs = np.linspace(np.log(tdata.wvl[0]), np.log(tdata.wvl[-1]), 1000)

wvlticks = [(list(range(200, 1000, 100))+
             list(range(1000, 10000, 1000))+
             list(range(10000, 71000, 10000))),
            (['0.2']+['']*7+
             ['1']+['']*8+
             ['10']+['']*5+['70'])]

plt.plot(xtlocs,
         tclearsurf(xtlocs)/np.pi, label='Surface (288K)')
plt.plot(xtlocs,
         tcleartoa(xtlocs)/np.pi, label='TOA (clear sky)')
plt.plot(xtlocs,
         tcldtoa(xtlocs)/np.pi, label='TOA (cloudy)')
for t in [300, 275, 250, 225, 200, 175]:
    plt.plot(xtlocs, 100*np.exp(2*xtlocs)*1e-18*
             planck_function(
                 t,
                 wavelength=np.exp(xtlocs)*1e-9), c='k', lw=0.5, zorder=-1)
    plwvl = wvlticks[0][-6]
    plt.text(np.log(plwvl), 100*plwvl**2*1e-18*
             planck_function(
                 t,
                 wavelength=plwvl*1e-9), str(t)+'K')

plt.xticks(np.log(wvlticks[0]), wvlticks[1])
plt.xlim(np.log(70000), np.log(4000))
plt.xlabel(r'Wavelength ($\mu m$)')
plt.ylabel(r'Spectral radiance (Wm$^{-2}$ sr$^{-1}$cm)')
plt.legend()

fig = plt.gcf()
fig.set_size_inches(8, 4.3)
plt.show()
# fig.savefig('output/cloud_temp.pdf', bbox_inches='tight')
