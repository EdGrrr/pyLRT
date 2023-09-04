from numpy.typing import ArrayLike
import pandas as pd
from pyLRT import RadTran
import xarray as xr
import numpy as np

default_output = {
    "disort": "lambda edir edn eup uavgdir, uavgdn, uavgup",
}

def parse_output(output: ArrayLike, rt:RadTran, dims=["lambda"], **dim_specs):
    """
    Parse the output of a libRadtran run into an xarray dataset

    Parameters
    ----------
    output : ArrayLike
        The output of a libRadtran run
    rt : RadTran
        The RadTran object used to run the simulation
    dims : list, optional
        The dimensions to unstack the output along, by default ["lambda"].
        Note that "lambda" is renamed to "wvl" in the output dataset.
    dim_specs : dict, optional
        The values of the dimensions to unstack the output along who are
        not included in the output of the libRadtran run, by default {}
    
    Returns
    -------
    xr.Dataset
        The output of the libRadtran run as an xarray dataset
    """


    output_cols = _get_columns(rt)
    output, dim_values = _unstack_dims(output, output_cols, dims, **dim_specs)

    # avoid calling wavelength "lambda", as it is a reserved word in python
    if "lambda" in dim_values:
        dim_values["wvl"] = dim_values["lambda"]
        del(dim_values["lambda"])

    output_dataset = xr.DataArray(output, coords={**dim_values, "variable":output_cols}, dims=["variable"]+dims)
    output_dataset = output_dataset.to_dataset("variable")

    # Check for directional quantities (radiances)
    if np.any(["uu" in col for col in output_dataset.data_vars]):
        output_dataset = _promote_uu_directions(output_dataset, rt)

    return output_dataset.squeeze()


def _get_columns(rt: RadTran):
    """Get the column names for the output of a libRadtran run"""
    try:
        solver = rt.options.get("rte_solver", "disort")
        output_cols = rt.options.get("output_user", default_output[rt.options["rte_solver"]]).split()
    except KeyError:
        raise NotImplementedError(f"rte_solver {solver} output parsing is implemented")

    # Add extra columns for directional radiances
    # get the number of umu and phi
    n_umu = len(rt.options.get("umu", "1.0").split())
    n_phi = len(rt.options.get("phi", "0.0").split())
    if "uu" in output_cols:
        # expand uu into uu(umu(0), phi(0)) ... uu(umu(0), phi(m)) ... uu(umu(n), phi(m))

        # get the index of uu
        uu_index = np.argwhere(np.array(output_cols) == "uu")[0, 0]
        
        # generate column names
        uu_cols = [f"uu(umu({i_umu}), phi({i_phi}))" for i_umu in range(n_umu) for i_phi in range(n_phi)]
        output_cols = output_cols[:uu_index] + uu_cols + output_cols[uu_index+1:]

    return output_cols


def _unstack_dims(output, output_cols, dims, **dim_specs):
    """Reshape the output of a libRadtran run and extract the 
    coordiantes for each dimension"""

    # Transpose the output such that the first index is the data_var
    output = np.array(output).T 

    # Identify the coordinates for each dim.
    dim_values = {}
    i_dim = 1
    for dim in dims:
        if dim not in output_cols:
            try:
                dim_values[dim] = np.array(dim_specs[dim])
            except KeyError:
                raise ValueError(f"Dimension {dim} not in output columns and no dim_spec provided.")
            i_dim += 1
            continue
        elif dim in dim_specs:
            print(f"Warning: dimension {dim} is in output columns and dim_spec. Using the output values.")
        dim_index = np.argwhere(np.array(output_cols) == dim)[0, 0]
        dim_values[dim] = np.unique(output[dim_index])
        i_dim += 1

    # reshape the output
    output = output.reshape((-1, *[len(np.unique(dim_values[dim])) for dim in dims]))
    
    # check the reshape is correct
    for dim in dims:
        if dim not in output_cols: # are dim values in output?
            print(f"Warning: dimension {dim} values are not in output; cannot check for consistency.")
            continue

        # check coordinates don't vary along non-dim axis
        dim_index = np.argwhere(np.array(output_cols) == dim)[0, 0]
        i_dim_axis = np.argwhere(np.array(dims) == dim)[0, 0]
        for i_axis in range(len(output.shape)-1):
            if i_axis != i_dim_axis:
                assert np.all(np.diff(output[dim_index], axis=i_axis) == 0), f"Dimension {dim} is not constant along axis {i_axis}."
            else:
                # check there aren't repeated values along the dim-axis
                assert np.unique(output[i_axis]).size == output[i_axis].shape[i_axis], f"Dimension {dim} has repeated values; is there another stacked dimension not specified in dims?"

    return output, dim_values

def _promote_uu_directions(output_dataset, rt):
    """Combine uu to a data_var and promote umu and phi to dims."""
    # Promoting phi to a dimension
    uu_cols = [col for col in output_dataset.data_vars if "uu" in col]
    n_umu = len(rt.options.get("umu", "1.0").split())
    for i_umu in range(n_umu):
        umu_cols = [col for col in uu_cols if f"umu({i_umu})" in col]
        da = xr.concat([output_dataset[col] for col in umu_cols], "phi")
        da.name = f"uu(umu({i_umu}))"
        output_dataset = xr.merge([output_dataset, da])
    output_dataset = output_dataset.drop_vars(uu_cols)
    # Promoting umu to a dimension
    uu_cols = [col for col in output_dataset.data_vars if "uu" in col]
    da = xr.concat([output_dataset[col] for col in uu_cols], "umu")
    da.name = "uu"
    output_dataset = xr.merge([output_dataset, da])
    output_dataset = output_dataset.drop_vars(uu_cols)

    # Add umu and phi coordinates based on input
    if "umu" in rt.options:
        umu_vals = np.array(rt.options["umu"].split(), dtype=float)
        output_dataset = output_dataset.assign_coords(umu=umu_vals)
    if "phi" in rt.options:
        phi_vals = np.array(rt.options["phi"].split(), dtype=float)
        output_dataset = output_dataset.assign_coords(phi=phi_vals)

    return output_dataset