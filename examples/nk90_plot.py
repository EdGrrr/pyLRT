'''An example plotting a Nakajima-King style warped grid of radiances for a liquid water cloud.
Note that the water cloud properties are calcualted using the default libradtran water cloud
properties, which are not suitable for radiances. If you want to do this properly, make sure you
calculate the optical properties using 'mie' '''
import numpy as np
from pyLRT import RadTran, get_lrt_folder
from multiprocessing.pool import Pool
from netCDF4 import Dataset
import matplotlib.pyplot as plt

#  Note that you can specify a location here too
LIBRADTRAN_FOLDER = get_lrt_folder()

vals = {'re': np.array([4, 8, 16, 32]),  # The max Deff value for effective radius is 26um (when using wc_property mie)
        'tau': np.array([0, 2, 4, 8, 16, 32, 64]),
        'solz': np.array([26.0]),
        'satz': np.array([np.cos(np.deg2rad(40))]),
        'raz': np.array([42.0])}

#Filter file locations are relative to UVSPEC
banddata = [['1940 2220', '../data/filter/modis/modis_aqua_b07', '0.045'],
            ['660 990', '../data/filter/modis/modis_aqua_b02', '0.060']]

def create_rtm(bdata):
    lrt21 = RadTran(LIBRADTRAN_FOLDER)
    lrt21.options['rte_solver'] = 'disort'
    lrt21.options['source'] = 'solar  ../data/solar_flux/kurudz_0.1nm.dat per_nm'
    lrt21.options['wavelength'] = bdata[0]
    lrt21.options['filter_function_file'] = bdata[1]
    # Set water vapour to zero
    lrt21.options['mol_modify h2o'] = '0 mm'
    #  Get reflectivities at TOA
    lrt21.options['output_user'] = 'uu'
    lrt21.options['output_quantity'] = 'reflectivity'
    lrt21.options['zout'] = 'TOA'
    lrt21.options['albedo'] = bdata[2]
    # View zenith angles are calcualted within DISTORT
    lrt21.options['umu'] = ' '.join([str(a)[:5] for a in vals['satz']])
    lrt21.options['phi'] = ' '.join([str(a) for a in vals['raz']])
    lrt21.options['phi0'] = 0
    #  Integrate over the wavebands using the filter functions
    lrt21.options['output_process'] = 'integrate'
    lrt21.options['quiet'] = ''
    return lrt21


def get_refl(inputs):
    bands, ndind, lwpind, solzind, cloud, solz, totalind = inputs
    outputs = [ndind, lwpind, solzind]
    print(totalind)
    for band in bands:
        rtm = create_rtm(band)
        rtm.cloud = cloud
        rtm.options['sza'] = str(solz)[:5]
        outputs.append(rtm.run())
    return outputs

        
grid_axes = ['re', 'tau', 'satz', 'solz', 'raz']
bands = banddata
output = np.zeros([len(bands)]+[len(vals[n]) for n in grid_axes])*np.nan

# Cloud constants
ctt = 280
dz = 500
f_ad = 1
rho_w = 1e6  # Density of water in gm-3
const_A = 2/3 * rho_w * f_ad * 1e-6
const_B = 1.67e4 * (0.0192*ctt-4.293)

total = len(vals['re'])*len(vals['tau'])*len(vals['solz'])

# Build the input array
tasks = []
for reind, re in enumerate(vals['re']):
    print('Re: {:6.2f}'.format(re))
    for tauind, tau in enumerate(vals['tau']):
        # We are working in re-tau space so have to calculate lwc
        # Using a constant LWC profile
        lwc = const_A * tau * re/dz
        if ((re>60) or (re<2.5)) and (lwc>0):
            continue
        cloud = {'z':   [1.0, 1-dz/1000],
                 'lwc': [0.0, lwc],
                 're':  [re, re]}
        for solzind, solz in enumerate(vals['solz']):
            totalind = (reind*len(vals['tau'])*len(vals['solz'])+
                        tauind*len(vals['solz']) +
                        solzind)
            tasks.append([bands, reind, tauind, solzind, cloud, solz, totalind])

# Run the radiative transfer
pool = Pool(8)
opmap = pool.map(get_refl, tasks, chunksize=1)

# Reshape the output to match the input data arrays
for op in opmap:
    output[0, op[0], op[1], :, op[2]] = op[3].reshape((len(vals['satz']), len(vals['raz'])))
    output[1, op[0], op[1], :, op[2]] = op[4].reshape((len(vals['satz']), len(vals['raz'])))

# Plot the reflection values with warped grid
for i in range(output.shape[1]):
    plt.plot(output[1, i].ravel(), output[0, i].ravel(), c='k')
    plt.text(output[1, i, -1, 0, 0]+0.02, output[0, i, -1, 0, 0], r'r$_e$='+'{}'.format(vals['re'][i])+r'$\mu$m', va='center')

for i in range(output.shape[2]):
    plt.plot(output[1, :, i].ravel(), output[0, :, i].ravel(), c='k')
    plt.text(output[1, -1, i, 0, 0], output[0, -1, i, 0, 0]-0.01, r'$\tau_c$='+'{}'.format(vals['tau'][i]), ha='center', va='top', rotation=90)
    
plt.xlabel('Reflection Function MODIS Ch2 (0.86$\mu$m)')
plt.ylabel('Reflection Function MODIS Ch7 (2.1$\mu$m)')
plt.xlim(0, 1.1)
plt.ylim(-0.06, 0.6)

plt.text(0.1, 0.4,
         r'$\theta_0$'+' = {:.0f}'.format(vals['solz'][0])+r'$^{\circ}$'+'\n'+
         r'$\theta$'+' = {:.0f}'.format(np.rad2deg(vals['satz'][0]))+r'$^{\circ}$'+'\n'+
         r'$\Delta\phi$'+' = {:.0f}'.format(vals['raz'][0])+r'$^{\circ}$')

fig = plt.gcf()
fig.set_size_inches(5, 4)
plt.show()
