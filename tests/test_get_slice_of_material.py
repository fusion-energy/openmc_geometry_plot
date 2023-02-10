import openmc
import openmc_geometry_plot

def test_get_slice_of_material_ids():
    
    openmc_material = openmc.Material()
    openmc_material.id = 1

    # my_materials = openmc.Materials([openmc_material])

    spheres = [openmc.Sphere(r=r) for r in [10, 20, 30, 40, 50]]

    spheres[-1].boundary_type = "vacuum"

    regions = openmc.model.subdivide(spheres)

    cells = [openmc.Cell(fill=openmc_material, region=r) for r in regions[:-1]]
    
    my_geometry = openmc.Geometry(cells)
    
    slice_data = my_geometry.get_slice_of_material_ids()
    
    my_geometry.view_direction = 'x'
    assert slice_data.shape == (500, 500)