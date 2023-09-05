from matplotlib import colorbar
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
tdata, tverb = tlrt.run(verbose=True, parse=True, dims=['lambda','pressure_out'], pressure_out=np.array(output_layers).astype('float'))

###################################################################
# Plot the radaitive heating rates on a symmetric log colourscale #
###################################################################
wvlticks = np.array([5000, 10000, 15000, 20000, 30000, 50000, 70000])

tdata.heat.plot(
    x="wvl", y="pressure_out", yincrease=False,
    norm=matplotlib.colors.SymLogNorm(linthresh=0.0001, vmin=-0.2, vmax=0.2,),
    cmap="RdBu_r", cbar_kwargs={"label": "Heating rate"},
    size=3, aspect=3, 
)

# scale the x-axis to match wavelength grid
ax = plt.gca()
wvlinterp = scipy.interpolate.interp1d(tdata.wvl, np.arange(len(tdata.wvl)), fill_value='extrapolate')
invwvlinterp = scipy.interpolate.interp1d(np.arange(len(tdata.wvl)), tdata.wvl, fill_value='extrapolate')
plt.gca().set_xscale(
    "function", functions=(
        lambda x: wvlinterp(np.ma.filled(x, np.nan)), 
        lambda x: invwvlinterp(np.ma.filled(x, np.nan)),
    )
)

# configure the x-axis
plt.xlim(4000, 70000)
plt.xticks(wvlticks, wvlticks//10000)


plt.ylabel('Pressure (mb)')
plt.xlabel(r'Wavelength ($\mu$m)')

fig = plt.gcf()
# fig.savefig('output/heating_rate.pdf', bbox_inches='tight')
plt.show()
fig.clf()
del(fig)

########################################
# Plot with GHG trasnmissivity spectra #
########################################

# Transmissivity ticks (for plots)
trans_ticks = np.array([0, 0.5, 1])

ax = plt.subplot(211)
tdata.heat.plot(
    x="wvl", y="pressure_out", yincrease=False,
    norm=matplotlib.colors.SymLogNorm(linthresh=0.0001, vmin=-0.2, vmax=0.2,),
    cmap="RdBu_r",
    ax=ax, add_colorbar=False,
)

# scale the x-axis to match wavelength grid
ax = plt.gca()
plt.gca().set_xscale(
    "function", functions=(
        lambda x: wvlinterp(np.ma.filled(x, np.nan)), 
        lambda x: invwvlinterp(np.ma.filled(x, np.nan)),
    )
)

# configure the x-axis
plt.xlim(4000, 70000)
plt.xticks(wvlticks, [])

plt.ylabel('Pressure (mb)')

for v,var in enumerate([['co2', r'CO$_2$'],
                        ['o3', r'O$_3$'],
                        ['h2o', r'H$_2$O']]):
    plt.subplot(6, 1, v+4)
    ext = 1 - np.exp(-tverb['gases'][var[0]][:, :].sum(axis=-1))
    plt.fill_between(range(len(ext)),
                     ext, color='grey')
    plt.plot(range(len(ext)),
             ext, c='k', lw=0.5)
    plt.xticks(wvlinterp(wvlticks), [])
    plt.xlim(*wvlinterp([4000, 70000]))
    plt.yticks(trans_ticks, trans_ticks*100)
    plt.ylim(0, 1)
    plt.xlabel(r'Wavelength ($\mu$m)')
    plt.ylabel(var[1])

plt.xticks(wvlinterp(wvlticks), wvlticks//1000)

fig = plt.gcf()
fig.set_size_inches(8, 4)
# fig.savefig('output/heating_rate_species.pdf', bbox_inches='tight')
plt.show()
fig.clf()
del(fig)

