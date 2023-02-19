import openmc
import openmc_geometry_plot
from pathlib import Path
import numpy as np
import pytest


def test_is_geometry_dagmc():
    bound_dag_univ = openmc.DAGMCUniverse(
        filename="two_disconnected_cubes.h5m"
    ).bounded_universe()
    my_geometry = openmc.Geometry(root=bound_dag_univ)
    assert my_geometry.is_geometry_dagmc() is True

    # same as above but no bounding cell
    bound_dag_univ = openmc.DAGMCUniverse(filename="two_disconnected_cubes.h5m")
    my_geometry = openmc.Geometry(root=bound_dag_univ)
    assert my_geometry.is_geometry_dagmc() is True


def test_get_dagmc_filepath():
    bound_dag_univ = openmc.DAGMCUniverse(filename="two_disconnected_cubes.h5m")
    my_geometry = openmc.Geometry(root=bound_dag_univ)
    assert (
        my_geometry.get_dagmc_filepath()
        == Path(__file__).parent / "two_disconnected_cubes.h5m"
    )


def test_get_dagmc_universe():
    bound_dag_univ = openmc.DAGMCUniverse(filename="two_disconnected_cubes.h5m")
    my_geometry = openmc.Geometry(root=bound_dag_univ)
    assert isinstance(my_geometry.get_dagmc_universe(), openmc.DAGMCUniverse)


def test_slice_material_dagmc_file():
    bound_dag_univ = openmc.DAGMCUniverse(
        filename="two_disconnected_cubes.h5m"
    ).bounded_universe()
    my_geometry = openmc.Geometry(root=bound_dag_univ)

    data_slice = my_geometry.get_slice_of_material_ids(
        view_direction="x", slice_value=1, pixels_across=10
    )

    assert np.array(data_slice).shape == (10, 10)


def test_slice_cell_dagmc_file():
    bound_dag_univ = openmc.DAGMCUniverse(
        filename="two_disconnected_cubes.h5m"
    ).bounded_universe()
    my_geometry = openmc.Geometry(root=bound_dag_univ)

    with pytest.raises(ValueError):
        data_slice = my_geometry.get_slice_of_cell_ids(
            view_direction="x", slice_value=1, pixels_across=10
        )

    assert np.array(data_slice).shape == (10, 10)
