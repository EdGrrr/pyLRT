from pyLRT import RadTran, get_lrt_folder
from pyLRT.misc import planck_function
import matplotlib.pyplot as plt
import copy
import numpy as np
import scipy
import scipy.interpolate

def planck_wvl_plot(t, wvl, add_text=True, ax=None, unit=1e-9, **kwargs):
    """Plot the Planck function for a given temperature and wavelength range."""
    if ax is None:
        ax = plt.gca()
    radiances = 100 * (wvl*unit)**2 * planck_function(t, wavelength=wvl*unit)
    handle = ax.plot(wvl, radiances, **kwargs)
    
    if add_text:
        label_wvl = wvl[np.array(radiances).argmax()] if isinstance(add_text, bool) else add_text
        ax.text(
            label_wvl, 
            radiances.max(), 
            str(t)+'K',)
    
    return handle

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
tdata, tverb = tlrt.run(verbose=True, parser=slrt.parser)
print('Cloud RT')
tcdata, tcverb = tlrt_cld.run(verbose=True, parser=slrt.parser)
scdata, scverb = slrt_cld.run(verbose=True, parser=slrt.parser)
print('Done RT')

fig, ax = plt.subplots(figsize=(8, 4.3))
(tdata.sel(zout=0)/np.pi).eup.plot(label='Surface (288K)', xincrease=False)
(tdata.sel(zout=120)/np.pi).eup.plot(label='TOA (clear sky)')
(tcdata.sel(zout=120)/np.pi).eup.plot(label='TOA (cloudy)')

plt.xscale('log')
for t in [300, 275, 250, 225, 200, 175]:
    planck_wvl_plot(t, tdata.wvl, unit=1e-9, add_text=True, c='k', lw=0.5, zorder=-1)

ax.xaxis.set_minor_formatter(lambda x,_: f"{x/1e3:.0f}")
ax.xaxis.set_major_formatter(lambda x,_: f"{x/1e3:.0f}")
plt.tick_params(which='minor', labelsize=8)
plt.xlabel(r'Wavelength ($\mu m$)')
plt.ylabel(r'Spectral radiance (Wm$^{-2}$ sr$^{-1}$cm)')
plt.legend()

plt.show()
# fig.savefig('output/cloud_temp.pdf', bbox_inches='tight')
