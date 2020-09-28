from pyLRT import RadTran, get_lrt_folder
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

LIBRADTRAN_FOLDER = get_lrt_folder()

tlrt = RadTran(LIBRADTRAN_FOLDER)
tlrt.options['rte_solver'] = 'disort'
# A black surface
tlrt.options['albedo'] = '0'
# Only thermal radiation
tlrt.options['source'] = 'thermal'
# 2.6um to 80um range
tlrt.options['wavelength'] = '2600 80000'
# Ensure spectrally and height-resolved output
tlrt.options['output_user'] = 'lambda zout heat'
# Set some pressure output levels
output_layers = [str(a) for a in np.arange(1010, 10, -25)]+['0.01']
tlrt.options['pressure_out'] = ' '.join(output_layers)
# Centred differnce calculation for the heating rate
tlrt.options['heating_rate'] = 'layer_cd'
# Only use a simple molecular absorption parametrisation (for speed)
tlrt.options['mol_abs_param'] = 'reptran coarse'
# US standard atmosphere for the profile
tlrt.options['atmosphere_file'] = '../data/atmmod/afglus.dat'
# Make sure a suitably high resolution vertial grid
tlrt.options['atm_z_grid'] = ' '.join([str(a) for a in np.arange(0, 100, 0.25)])
# :-(
tlrt.options['mixing_ratio'] = 'co2 410'

# Run the thermal IR radiative transfer
tdata, tverb = tlrt.run(verbose=True)

###################################################################
# Plot the radaitive heating rates on a symmetric log colourscale #
###################################################################
toplot = tdata[:, 2].reshape((-1, len(output_layers))).transpose()[::-1]
plt.imshow(toplot, cmap=plt.get_cmap('RdBu_r'), vmin=-0.2, vmax=0.2,
           norm=matplotlib.colors.SymLogNorm(linthresh=0.0001),
           aspect='auto')
wvl = tdata[:, 0]
wvlticks = np.array([5000, 10000, 15000, 20000, 30000, 50000, 70000])
wvlinterp = scipy.interpolate.interp1d(wvl, np.arange(len(wvl))/len(output_layers))
plt.xticks(wvlinterp(wvlticks), wvlticks//1000)
yticklocs = np.linspace(0, len(output_layers)-1, 5)
plt.yticks(yticklocs, (np.array(output_layers)[yticklocs.astype('int')][::-1].astype('float')-10).astype('int'))
plt.ylim(len(output_layers), 1)
plt.xlim(*wvlinterp([4000, 70000]))
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel('Pressure (mb)')

plt.colorbar(label='Heating rate')

fig = plt.gcf()
fig.set_size_inches(8, 4)
fig.savefig('output/heating_rate.pdf', bbox_inches='tight')
fig.clf()
del(fig)

########################################
# Plot with GHG trasnmissivity spectra #
########################################

# Transmissivity ticks (for plots)
trans_ticks = [[0, 0.5, 1], [0, 50, 100]]

plt.subplot(211)
toplot = tdata[:, 2].reshape((-1, len(output_layers))).transpose()[::-1]
plt.imshow(toplot, cmap=plt.get_cmap('RdBu_r'), vmin=-0.2, vmax=0.2,
           norm=matplotlib.colors.SymLogNorm(linthresh=0.0001),
           aspect='auto')
wvl = tdata[:, 0]
wvlticks = np.array([5000, 10000, 15000, 20000, 30000, 50000, 70000])
wvlinterp = scipy.interpolate.interp1d(wvl, np.arange(len(wvl))/len(output_layers))
plt.xticks(wvlinterp(wvlticks), [])
yticklocs = np.linspace(0, len(output_layers)-1, 5)
plt.yticks(yticklocs, (np.array(output_layers)[yticklocs.astype('int')][::-1].astype('float')-10).astype('int'))
plt.ylim(len(output_layers), 1)
plt.xlim(*wvlinterp([4000, 70000]))
plt.ylabel('Pressure (mb)')

for v,var in enumerate([['co2', r'CO$_2$'],
                        ['o3', r'O$_3$'],
                        ['h2o', r'H$_2$O']]):
    plt.subplot(6, 1, v+4)
    ext = 1-np.exp(-tverb['gases'][var[0]][:, :].sum(axis=-1))
    plt.fill_between(range(len(ext)),
                     ext, color='grey')
    plt.plot(range(len(ext)),
             ext, c='k', lw=0.5)
    plt.xticks(wvlinterp(wvlticks), [])
    plt.xlim(*wvlinterp([4000, 70000]))
    plt.yticks(*trans_ticks)
    plt.ylim(0, 1)
    plt.xlabel(r'Wavelength ($\mu$m)')
    plt.ylabel(var[1])

fig = plt.gcf()
fig.set_size_inches(8, 4)
fig.savefig('output/heating_rate_species.pdf', bbox_inches='tight')
fig.clf()
del(fig)

