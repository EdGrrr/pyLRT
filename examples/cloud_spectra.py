"""
Cloud spectra example.
======================

This example shows how to calculate the spectral radiance of a cloudy atmosphere
using the pyLRT wrapper for libRadtran, and plots the upwelling thermal radiance
at the surface and top of the atmosphere for a clear and cloudy atmosphere with 
Planck functions for different temperatures. This demonstrates the the 
thermalisation of the upwelling radiation in different spectral regions.
"""

# %% Package imports and helper functions
from pyLRT import RadTran, get_lrt_folder
from pyLRT.misc import planck_function
import matplotlib.pyplot as plt
import copy
import numpy as np


def planck_wvl_plot(t, wvl, add_text=True, ax=None, unit=1e-9, **kwargs):
    """Plot the Planck function for a given temperature and wavelength range."""
    if ax is None:
        ax = plt.gca()
    radiances = 100 * (wvl * unit) ** 2 * planck_function(t, wavelength=wvl * unit)
    handle = ax.plot(wvl, radiances, **kwargs)

    if add_text:
        label_wvl = (
            wvl[np.unravel_index(np.array(radiances).argmax(), wvl.shape)]
            if isinstance(add_text, bool)
            else add_text
        )
        ax.text(
            label_wvl,
            radiances.max(),
            str(t) + "K",
        )

    return handle


# %% Set up the radiative transfer models
LIBRADTRAN_FOLDER = get_lrt_folder()

slrt = RadTran(LIBRADTRAN_FOLDER)
slrt.options["rte_solver"] = "disort"
slrt.options["source"] = "solar"
slrt.options["wavelength"] = "200 2600"
slrt.options["output_user"] = "lambda eglo eup edn edir"
slrt.options["zout"] = "0 5 TOA"
slrt.options["albedo"] = "0"
slrt.options["umu"] = "-1.0 1.0"
slrt.options["quiet"] = ""
slrt.options["sza"] = "0"

tlrt = copy.deepcopy(slrt)
tlrt.options["source"] = "thermal"
tlrt.options["output_user"] = "lambda edir eup uu"
tlrt.options["wavelength"] = "2500 80000"
tlrt.options["mol_abs_param"] = "reptran fine"

# Add in a cloud
slrt_cld = copy.deepcopy(slrt)
slrt_cld.cloud = {
    "z": np.array([4, 3.7]),
    "lwc": np.array([0, 0.5]),
    "re": np.array([0, 20]),
}

tlrt_cld = copy.deepcopy(tlrt)
tlrt_cld.cloud = {
    "z": np.array([4, 3.7]),
    "lwc": np.array([0, 0.5]),
    "re": np.array([0, 20]),
}

# %% Run the radiative transfer models
print("Initial RT")
sdata = slrt.run(parse=True)
tdata = tlrt.run(parse=True)
print("Cloud RT")
tcdata = tlrt_cld.run(parse=True)
scdata = slrt_cld.run(parse=True)
print("Done RT")

# %% Plot the results
fig, ax = plt.subplots(figsize=(8, 4.3), constrained_layout=True)

# The upwelling thermal radiance at the surface and TOA (in clear and cloudy conditions)
(tdata.sel(zout="0") / np.pi).eup.plot(label="Surface (288K)", xincrease=False)
(tdata.sel(zout="TOA") / np.pi).eup.plot(label="TOA (clear sky)")
(tcdata.sel(zout="TOA") / np.pi).eup.plot(label="TOA (cloudy)")

# Planck functions for different temperatures
for t in [300, 275, 250, 225, 200, 175]:
    planck_wvl_plot(t, tdata.wvl, unit=1e-9, add_text=True, c="k", lw=0.5, zorder=-1)

# Fine tune the plot
plt.xscale("log")
ax.xaxis.set_minor_formatter(lambda x, _: f"{x/1e3:.0f}")
ax.xaxis.set_major_formatter(lambda x, _: f"{x/1e3:.0f}")
plt.tick_params(which="minor", labelsize=8)
plt.xlabel(r"Wavelength ($\mu m$)")
plt.ylabel(r"Spectral radiance (Wm$^{-2}$ sr$^{-1}$cm)")
plt.legend()

# Show and save the plot
plt.show()
fig.savefig("output/cloud_temp.pdf")
