import openmc
import openmc_geometry_plot
import numpy as np


def test_get_slice_of_material_ids():

    openmc_material = openmc.Material()
    openmc_material.id = 1

    # my_materials = openmc.Materials([openmc_material])

    surf_1 = openmc.Sphere(r=10)
    surf_2 = openmc.Sphere(r=20)
    surf_3 = openmc.Sphere(r=30, boundary_type = "vacuum")

    region_1 = -surf_1
    region_2 = +surf_1 & -surf_2
    region_3 = +surf_2 & -surf_3

    cell1 = openmc.Cell(fill=openmc_material, region=region_1)
    cell1.id = 10
    cell2 = openmc.Cell(region=region_2)
    cell2.id = 20
    cell3 = openmc.Cell(fill=openmc_material, region=region_3)
    cell3.id = 40
    
    my_geometry = openmc.Geometry([cell1,cell2,cell3])

    slice_data = my_geometry.get_slice_of_cell_ids(view_direction="x")

    assert np.array(slice_data).shape == (500, 500)
    for rows in slice_data:
        for value in rows:
            assert value in [0, 10, 20, 40]
