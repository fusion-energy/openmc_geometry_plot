# plots a geometry with both backends (matplotlib and plotly)
# A couple of zoomed in plots are plotted
# All view angles (x, y, z) for the whole geometry are plotted


import openmc
import openmc_geometry_plot  # adds plot_axis_slice to openmc.Geometry


breeder_material = openmc.Material()
breeder_material.add_element("Li", 1.0)
breeder_material.set_density("g/cm3", 0.01)

copper_material = openmc.Material()
copper_material.set_density("g/cm3", 0.01)
copper_material.add_element("Cu", 1.0)

eurofer_material = openmc.Material()
eurofer_material.set_density("g/cm3", 0.01)
eurofer_material.add_element("Fe", 1)

# surfaces
central_sol_surface = openmc.ZCylinder(r=100)
central_shield_outer_surface = openmc.ZCylinder(r=110)
port_hole = openmc.Sphere(r=60, x0=500)
upper_port_hole = openmc.Sphere(r=100, z0=500)
vessel_inner_surface = openmc.Sphere(r=500)
first_wall_outer_surface = openmc.Sphere(r=510)
breeder_blanket_outer_surface = openmc.Sphere(r=610, boundary_type="vacuum")

# cells
central_sol_region = (
    -central_sol_surface & -breeder_blanket_outer_surface & +upper_port_hole
)
central_sol_cell = openmc.Cell(region=central_sol_region)
central_sol_cell.fill = copper_material

central_shield_region = (
    +central_sol_surface
    & -central_shield_outer_surface
    & -breeder_blanket_outer_surface
    & +upper_port_hole
)
central_shield_cell = openmc.Cell(region=central_shield_region)
central_shield_cell.fill = eurofer_material

inner_vessel_region = (
    -vessel_inner_surface
    & +central_shield_outer_surface
    & +port_hole
    & +upper_port_hole
)
inner_vessel_cell = openmc.Cell(region=inner_vessel_region)
# no material set as default is vacuum

upper_port_hole_region = -upper_port_hole
upper_port_hole_cell = openmc.Cell(region=upper_port_hole_region)

port_hole_region = -port_hole
port_hole_cell = openmc.Cell(region=port_hole_region)
# no material set as default is vacuum

first_wall_region = (
    -first_wall_outer_surface & +vessel_inner_surface & +port_hole & +upper_port_hole
)
first_wall_cell = openmc.Cell(region=first_wall_region)
first_wall_cell.fill = eurofer_material

breeder_blanket_region = (
    +first_wall_outer_surface
    & -breeder_blanket_outer_surface
    & +central_shield_outer_surface
    & +port_hole
    & +upper_port_hole
)
breeder_blanket_cell = openmc.Cell(region=breeder_blanket_region)
breeder_blanket_cell.fill = breeder_material

my_geometry = openmc.Geometry(
    [
        central_sol_cell,
        central_shield_cell,
        inner_vessel_cell,
        first_wall_cell,
        breeder_blanket_cell,
        port_hole_cell,
        upper_port_hole_cell,
    ]
)


# example code for plotting materials of the geometry with an outline

import numpy as np
import matplotlib.pylab as plt



import time
for x in range(2):
    start_time = time.time()
    data_slice = my_geometry.get_slice_of_material_ids(view_direction='x')
    print("--- %s seconds ---" % (time.time() - start_time))


# data_slice = my_geometry.get_slice_of_material_ids(view_direction='x')
# xlabel, ylabel = my_geometry.get_axis_labels(view_direction='x')
# plt.xlabel(xlabel)
# plt.ylabel(ylabel)

# plot_extent = my_geometry.get_mpl_plot_extent(view_direction='x')

# plt.imshow(
#     data_slice,
#     extent=plot_extent,
#     interpolation="none",
# )

# # gets unique levels for outlines contour plot
# levels = np.unique([item for sublist in data_slice for item in sublist])

# plt.contour(
#     data_slice,
#     origin="upper",
#     colors="k",
#     linestyles="solid",
#     levels=levels,
#     linewidths=0.5,
#     extent=plot_extent,
# )

# plt.show()
