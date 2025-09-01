'''Plots the complex refractive index of water as a function of wavelength.
Requires the libradtran optical properties to be downloaded'''
from pyLRT import RadTran, get_lrt_folder
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from pathlib import Path

LIBRADTRAN_FOLDER = get_lrt_folder()

try:
    ncdf = Dataset(Path(LIBRADTRAN_FOLDER) /'data/wc/mie/wc.sol.mie.cdf')
except FileNotFoundError:
    raise FileNotFoundError('This program requires the libRadtran water cloud optical properties.')
wvl = ncdf.variables['wavelen'][:]
refim = ncdf.variables['refim'][:]
refre = ncdf.variables['refre'][:]
ncdf.close()

ncdf = Dataset(LIBRADTRAN_FOLDER+'/data/wc/mie/wc.trm.mie.cdf')
wvl = np.concatenate([wvl, ncdf.variables['wavelen'][:]])
refim = np.concatenate([refim, ncdf.variables['refim'][:]])
refre = np.concatenate([refre, ncdf.variables['refre'][:]])
ncdf.close()

plt.plot(np.log(wvl), refim, c='k', label = 'Imaginary part (Absorption)')
plt.plot(np.log(wvl), refre, c='k', ls='--', label='Real part (Refractive)')

xticks = [0.5, 1, 3, 10, 30, 70]
plt.xticks(np.log(xticks), xticks)
plt.xlabel(r'Wavelength ($\mu$m)')
plt.xlim(np.log(0.24), np.log(70))

plt.yscale('log')
plt.ylabel('Complex Refractive Index')
plt.legend()

fig = plt.gcf()
fig.set_size_inches(4, 4)
plt.tight_layout()
fig.savefig('output/refind.pdf', bbox_inches='tight')
