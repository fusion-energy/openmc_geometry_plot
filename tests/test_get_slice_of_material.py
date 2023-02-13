import openmc
import openmc_geometry_plot
import numpy as np

def test_get_slice_of_material_ids():
    
    openmc_material = openmc.Material()
    openmc_material.id = 1

    # my_materials = openmc.Materials([openmc_material])

    spheres = [openmc.Sphere(r=r) for r in [10, 20, 30, 40, 50]]

    spheres[-1].boundary_type = "vacuum"

    regions = openmc.model.subdivide(spheres)

    cells = [openmc.Cell(fill=openmc_material, region=r) for r in regions[:-1]]
    
    my_geometry = openmc.Geometry(cells)
    
    slice_data = my_geometry.get_slice_of_material_ids(view_direction='x')

    assert np.array(slice_data).shape == (500, 500)
    for rows in slice_data:
        for value in rows:
            assert value in [1, 0]