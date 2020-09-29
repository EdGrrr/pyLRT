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

##############
# Run the RT #
##############
print('Initial RT')
sdata, sverb = slrt.run(verbose=True)
tdata, tverb = tlrt.run(verbose=True)
print('Done RT')

###########################
# Setup some plot details #
###########################
wvlticks = [(list(range(200, 1000, 100))+
             list(range(1000, 10000, 1000))+
             list(range(10000, 71000, 10000))),
            (['0.2']+['']*7+
             ['1']+['']*8+
             ['10']+['']*5+['70'])]
trans_ticks = [[0, 0.25, 0.5, 0.75, 1], [0, 25, 50, 75, 100]]

tclearsurf = scipy.interpolate.interp1d(np.log(tdata[::3, 0]), tdata[::3, 2])
tcleartoa = scipy.interpolate.interp1d(np.log(tdata[2::3, 0]), tdata[2::3, 2])
xtlocs = np.linspace(np.log(tdata[0, 0]), np.log(tdata[-1, 0]), 1000)

vars = [['rayleigh_dtau', 'rayleigh', 'Rayleigh'],
        ['o3', 'o3', r'O$_3$'],
        ['o2', 'o2', r'O$_2$'],
        ['h2o', 'h2o', r'H$_2$O'],
        ['co2', 'co2', r'CO$_2$'],
        ['ch4', 'ch4', r'CH$_4$']]


##########################################################
# Total Extinction, Planck function and major components #
##########################################################
fig = plt.gcf()
toa = 0 # Top of atmosphere index
# Atmospheric transmittance
swvl = sverb['gases']['wvl'][::10]
swvl = np.concatenate((swvl, np.linspace(2500, 5000, 100)))

plt.subplot2grid((7, 1), (0, 0), rowspan=2)
# Extinction is calculated from optical depth (sum of molecular and rayleigh components)
plt.fill_between(np.log(sverb['gases']['wvl'][::10]),
                 1-np.exp((-sverb['gases']['mol_abs'] -
                          sverb['gases']['rayleigh_dtau'])[::10, toa:].sum(axis=-1)), color='grey')
plt.plot(np.log(sverb['gases']['wvl'][::10]),
         1-np.exp((-sverb['gases']['mol_abs'] -
                   sverb['gases']['rayleigh_dtau'])[::10, toa:].sum(axis=-1)), c='k', lw=0.5)
# Included rayleigh for the LW section, although the contribution is small!
plt.fill_between(np.log(tverb['gases']['wvl'][::10]),
                 1-np.exp((-tverb['gases']['mol_abs'] -
                           tverb['gases']['rayleigh_dtau'])[::10, toa:].sum(axis=-1)), color='grey')
plt.plot(np.log(tverb['gases']['wvl'][::10]),
         1-np.exp((-tverb['gases']['mol_abs'] -
                   tverb['gases']['rayleigh_dtau'])[::10, toa:].sum(axis=-1)), c='k', lw=0.5)

# Calculate some Planck functions at representative temperatures
planck5800 = planck_function(5800, wavelength=swvl*1e-9)
planck5800 = planck5800*swvl
plt.plot(np.log(swvl), planck5800/planck5800.max(), c='b', lw=2, label='5800K')
planck255 = planck_function(210, wavelength=tverb['gases']['wvl'][::10]*1e-9)
planck255 = planck255*tverb['gases']['wvl'][::10]
plt.plot(np.log(tverb['gases']['wvl'][::10]), planck255/planck255.max(), c='r', lw=2, label='210K')
planck255 = planck_function(255, wavelength=tverb['gases']['wvl'][::10]*1e-9)
planck255 = planck255*tverb['gases']['wvl'][::10]
plt.plot(np.log(tverb['gases']['wvl'][::10]), planck255/planck255.max(), c='C1', lw=2, label='255K')
planck255 = planck_function(310, wavelength=tverb['gases']['wvl'][::10]*1e-9)
planck255 = planck255*tverb['gases']['wvl'][::10]
plt.plot(np.log(tverb['gases']['wvl'][::10]), planck255/planck255.max(), c='y', lw=2, label='310K')

plt.legend()

plt.xticks(np.log(wvlticks[0]), ['']*len(wvlticks[1]))
plt.xlim(np.log(wvlticks[0][0]), np.log(wvlticks[0][-1]))
plt.yticks(*trans_ticks)
plt.ylim(0, 1)
plt.ylabel(r'Extinction')
ax2 = plt.gca().twinx()
plt.ylabel(r'$\lambda$B$_{\lambda}$ (normalised)')
plt.yticks([], [])

# Plot the extinction for a selection of important gases
for v, var in enumerate(vars[:-1]):
    plt.subplot2grid((7, 1), (v+2, 0))
    plt.fill_between(np.log(sverb['gases']['wvl'][::10]),
                     1-np.exp(-sverb['gases'][var[0]].sum(axis=-1))[::10], color='grey')
    sol, = plt.plot(np.log(sverb['gases']['wvl'][::10]),
                   1-np.exp(-sverb['gases'][var[0]].sum(axis=-1))[::10], c='k', lw=0.5)
    plt.fill_between(np.log(tverb['gases']['wvl'][::10]),
                     1-np.exp(-tverb['gases'][var[0]].sum(axis=-1))[::10], color='grey')
    plt.plot(np.log(tverb['gases']['wvl'][::10]),
            1-np.exp(-tverb['gases'][var[0]].sum(axis=-1))[::10], c='k', lw=0.5)
    plt.text(np.log(30000), 0.5, var[2], verticalalignment='center')
    plt.xticks([], [])
    plt.xlim(np.log(wvlticks[0][0]), np.log(wvlticks[0][-1]))
    plt.yticks(trans_ticks[0], ['', '', '', '', ''])
    plt.ylim(0, 1)

plt.xticks(np.log(wvlticks[0]), wvlticks[1])
plt.xlabel(r'Wavelength ($\mu$m)')
plt.tight_layout(h_pad=0.1)
fig.set_size_inches(6, 5)
fig.savefig('output/as_complete.png', bbox_inches='tight')
fig.clf()
del(fig)


#########################################
# Total extinction and Planck functions #
#########################################
fig = plt.gcf()
toa = 0
# Atmospheric transmittance
swvl = sverb['gases']['wvl'][::10]
swvl = np.concatenate((swvl, np.linspace(2500, 5000, 100)))
plt.fill_between(np.log(sverb['gases']['wvl'][::10]),
                 1-np.exp((-sverb['gases']['mol_abs'] -
                          sverb['gases']['rayleigh_dtau'])[::10, toa:].sum(axis=-1)), color='grey')
plt.plot(np.log(sverb['gases']['wvl'][::10]),
         1-np.exp((-sverb['gases']['mol_abs'] -
                   sverb['gases']['rayleigh_dtau'])[::10, toa:].sum(axis=-1)), c='k', lw=0.5)
plt.fill_between(np.log(tverb['gases']['wvl'][::10]),
                 1-np.exp((-tverb['gases']['mol_abs'] -
                           tverb['gases']['rayleigh_dtau'])[::10, toa:].sum(axis=-1)), color='grey')
plt.plot(np.log(tverb['gases']['wvl'][::10]),
         1-np.exp((-tverb['gases']['mol_abs'] -
                   tverb['gases']['rayleigh_dtau'])[::10, toa:].sum(axis=-1)), c='k', lw=0.5)

planck5800 = planck_function(5800, wavelength=swvl*1e-9)
planck5800 = planck5800*swvl
plt.plot(np.log(swvl), planck5800/planck5800.max(), c='b', lw=2, label='5800K')
planck255 = planck_function(210, wavelength=tverb['gases']['wvl'][::10]*1e-9)
planck255 = planck255*tverb['gases']['wvl'][::10]
plt.plot(np.log(tverb['gases']['wvl'][::10]), planck255/planck255.max(), c='r', lw=2, label='210K')
planck255 = planck_function(255, wavelength=tverb['gases']['wvl'][::10]*1e-9)
planck255 = planck255*tverb['gases']['wvl'][::10]
plt.plot(np.log(tverb['gases']['wvl'][::10]), planck255/planck255.max(), c='C1', lw=2, label='255K')
planck255 = planck_function(310, wavelength=tverb['gases']['wvl'][::10]*1e-9)
planck255 = planck255*tverb['gases']['wvl'][::10]
plt.plot(np.log(tverb['gases']['wvl'][::10]), planck255/planck255.max(), c='y', lw=2, label='310K')

plt.legend()

plt.xticks(np.log(wvlticks[0]), wvlticks[1])
plt.xlim(np.log(wvlticks[0][0]), np.log(wvlticks[0][-1]))
plt.yticks(*trans_ticks)
plt.ylim(0, 1)
plt.xlabel(r'Wavelength ($\mu$m)')
plt.ylabel(r'Extinction')
ax2 = plt.gca().twinx()
plt.ylabel(r'$\lambda$B$_{\lambda}$ (normalised)')
plt.yticks([], [])
plt.tight_layout(h_pad=0)
fig.set_size_inches((8, 3))
fig.savefig('output/as_total.png')
fig.clf()


#############################
# Extinction by height plot #
#############################
fig = plt.gcf()
for k, toa in enumerate([-1, -5, -10]):
    print(k, toa)
    plt.subplot(3, 1, k+1)
    plt.fill_between(np.log(sverb['gases']['wvl'][::10]),
                     1-np.exp((-sverb['gases']['mol_abs'] -
                               sverb['gases']['rayleigh_dtau'])[::10, :toa].sum(axis=-1)), color='grey')
    plt.plot(np.log(sverb['gases']['wvl'][::10]),
             1-np.exp((-sverb['gases']['mol_abs'] -
                       sverb['gases']['rayleigh_dtau'])[::10, :toa].sum(axis=-1)), c='k', lw=0.5)
    plt.fill_between(np.log(tverb['gases']['wvl'][::10]),
                     1-np.exp((-tverb['gases']['mol_abs'] -
                               tverb['gases']['rayleigh_dtau'])[::10, :toa].sum(axis=-1)), color='grey')
    plt.plot(np.log(tverb['gases']['wvl'][::10]),
             1-np.exp((-tverb['gases']['mol_abs'] -
                       tverb['gases']['rayleigh_dtau'])[::10, :toa].sum(axis=-1)), c='k', lw=0.5)
    if k == 2:
        plt.xlabel(r'Wavelength ($\mu$m)')
        plt.xticks(np.log(wvlticks[0]), wvlticks[1])
    else:
        plt.xlabel('')
        plt.xticks(np.log(wvlticks[0]), [''])

    plt.ylabel('Extinction\n'+['TOA-Surf', 'TOA-5km', 'TOA-10km'][k])
    plt.yticks(*trans_ticks)
    plt.xlim(np.log(wvlticks[0][0]), np.log(wvlticks[0][-1]))
    plt.ylim(0, 1)
    
plt.tight_layout(h_pad=0)
fig = plt.gcf()
fig.set_size_inches((8, 4))
fig.savefig('output/as_total_heights.png')
fig.clf()


#################################
# Extinction by component plots #
#################################
for var in vars:
    plt.fill_between(np.log(sverb['gases']['wvl'][::10]),
                     1-np.exp(-sverb['gases'][var[0]][::10, toa:].sum(axis=-1)), color='grey')
    plt.plot(np.log(sverb['gases']['wvl'][::10]),
                    1-np.exp(-sverb['gases'][var[0]][::10, toa:].sum(axis=-1)), c='k', lw=0.5)
    plt.fill_between(np.log(tverb['gases']['wvl'][::10]),
                     1-np.exp(-tverb['gases'][var[0]][::10, toa:].sum(axis=-1)), color='grey')
    plt.plot(np.log(tverb['gases']['wvl'][::10]),
             1-np.exp(-tverb['gases'][var[0]][::10, toa:].sum(axis=-1)), c='k', lw=0.5)
    plt.xticks(np.log(wvlticks[0]), wvlticks[1])
    plt.xlim(np.log(wvlticks[0][0]), np.log(wvlticks[0][-1]))
    plt.yticks(*trans_ticks)
    plt.ylim(0, 1)
    plt.xlabel(r'Wavelength ($\mu$m)')
    plt.ylabel(r'Extinction')
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches((8, 3))
    fig.savefig('output/as_{}.png'.format(var[1]))
    fig.clf()


##############################################
# Extinction and contribution to total plots #
##############################################
for var in vars:
    plt.subplot(211)
    plt.fill_between(np.log(sverb['gases']['wvl'][::10]),
                     1-np.exp(-sverb['gases']['mol_abs'].sum(axis=-1) -
                              sverb['gases']['rayleigh_dtau'].sum(axis=-1))[::10], color='r')
    plt.fill_between(np.log(tverb['gases']['wvl'][::10]),
                     1-np.exp(-tverb['gases']['mol_abs'].sum(axis=-1) -
                              tverb['gases']['rayleigh_dtau'].sum(axis=-1))[::10], color='r')

    plt.fill_between(np.log(sverb['gases']['wvl'][::10]),
                     1-np.exp(-sverb['gases']['mol_abs'].sum(axis=-1) -
                              sverb['gases']['rayleigh_dtau'].sum(axis=-1) +
                              sverb['gases'][var[0]].sum(axis=-1))[::10], color='grey')
    sol, = plt.plot(np.log(sverb['gases']['wvl'][::10]),
                   1-np.exp(-sverb['gases']['mol_abs'].sum(axis=-1) -
                            sverb['gases']['rayleigh_dtau'].sum(axis=-1) +
                            sverb['gases'][var[0]].sum(axis=-1))[::10], c='k', lw=0.5)
    plt.fill_between(np.log(tverb['gases']['wvl'][::10]),
                     1-np.exp(-tverb['gases']['mol_abs'].sum(axis=-1) -
                              tverb['gases']['rayleigh_dtau'].sum(axis=-1) +
                              tverb['gases'][var[0]].sum(axis=-1))[::10], color='grey')
    plt.plot(np.log(tverb['gases']['wvl'][::10]),
            1-np.exp(-tverb['gases']['mol_abs'].sum(axis=-1) -
                     tverb['gases']['rayleigh_dtau'].sum(axis=-1) +
                     tverb['gases'][var[0]].sum(axis=-1))[::10], c='k', lw=0.5)
    plt.xticks([], [])
    plt.xlim(np.log(wvlticks[0][0]), np.log(wvlticks[0][-1]))
    plt.yticks(*trans_ticks)
    plt.ylim(0, 1)
    plt.ylabel(r'Extinction')   
    
    plt.subplot(212)
    plt.fill_between(np.log(sverb['gases']['wvl'][::10]),
                     1-np.exp(-sverb['gases'][var[0]].sum(axis=-1))[::10], color='grey')
    sol, = plt.plot(np.log(sverb['gases']['wvl'][::10]),
                    1-np.exp(-sverb['gases'][var[0]].sum(axis=-1))[::10], c='k', lw=0.5)
    plt.fill_between(np.log(tverb['gases']['wvl'][::10]),
                     1-np.exp(-tverb['gases'][var[0]].sum(axis=-1))[::10], color='grey')
    plt.plot(np.log(tverb['gases']['wvl'][::10]),
             1-np.exp(-tverb['gases'][var[0]].sum(axis=-1))[::10], c='k', lw=0.5)
    plt.xticks(np.log(wvlticks[0]), wvlticks[1])
    plt.xlim(np.log(wvlticks[0][0]), np.log(wvlticks[0][-1]))
    plt.yticks(*trans_ticks)
    plt.ylim(0, 1)
    plt.xlabel(r'Wavelength ($\mu$m)')
    plt.ylabel(r'Extinction')
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches((8, 3))
    fig.savefig('output/as2_{}.png'.format(var[1]))
    fig.clf()
