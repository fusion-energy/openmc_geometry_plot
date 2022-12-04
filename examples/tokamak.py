

import openmc
import matplotlib.pyplot as plt

# surfaces
central_column_surface = openmc.ZCylinder(r=100) # note the new surface type
inner_sphere_surface = openmc.Sphere(r=480)
middle_sphere_surface = openmc.Sphere(r=500) 
outer_sphere_surface = openmc.Sphere(r=600, boundary_type='vacuum')

# regions
# the center column region is cut at the top and bottom using the -outer_sphere_surface
central_column_region = -central_column_surface & -outer_sphere_surface
firstwall_region = -middle_sphere_surface & +inner_sphere_surface & +central_column_surface
blanket_region = +middle_sphere_surface & -outer_sphere_surface & +central_column_surface
inner_vessel_region = +central_column_surface & -inner_sphere_surface

# cells
firstwall_cell = openmc.Cell(region=firstwall_region)
central_column_cell = openmc.Cell(region=central_column_region)
blanket_cell = openmc.Cell(region=blanket_region)
inner_vessel_cell = openmc.Cell(region=inner_vessel_region)

universe = openmc.Universe(cells=[central_column_cell, firstwall_cell,
                                  blanket_cell, inner_vessel_cell])

my_geometry = openmc.Geometry(universe)


from openmc_geometry_plot import plot_axis_slice

plot = plot_axis_slice(
    geometry=my_geometry,
    view_direction='x',
)

plot.show()