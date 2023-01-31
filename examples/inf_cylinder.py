import openmc


material_1 = openmc.Material()
material_1.set_density("g/cm3", 0.01)
material_1.add_element("Ag", 1)

material_2 = openmc.Material()
material_2.set_density("g/cm3", 0.01)
material_2.add_element("Li", 1)

# surfaces
surface_1 = openmc.ZCylinder(r=50)
surface_2 = openmc.ZCylinder(r=100)

region_1 = -surface_1
region_2 = -surface_2 & +surface_1

cell_1 = openmc.Cell(region=region_1)
cell_1.fill = material_1

cell_2 = openmc.Cell(region=region_2)
cell_2.fill = material_2

universe = openmc.Universe(cells=[cell_1, cell_2])
my_geometry = openmc.Geometry(universe)


# example code for plotting materials of the geometry with an outline

import numpy as np
import openmc_geometry_plot  # adds data slicing functions to openmc.Geometry
import matplotlib.pylab as plt

my_geometry.view_direction = "x"
plot_left = -100
plot_right = 100
plot_bottom = 200
plot_top = -200
data_slice = my_geometry.get_slice_of_material_ids(
    plot_left=plot_left,
    plot_right=plot_right,
    plot_bottom=plot_bottom,
    plot_top=plot_top,
)

xlabel, ylabel = my_geometry.get_axis_labels()
plt.xlabel(xlabel)
plt.ylabel(ylabel)

plt.imshow(
    data_slice,
    extent=(plot_left, plot_right, plot_bottom, plot_top),
    interpolation="none",
)

# gets unique levels for outlines contour plot
levels = np.unique([item for sublist in data_slice for item in sublist])

plt.contour(
    data_slice,
    origin="upper",
    colors="k",
    linestyles="solid",
    levels=levels,
    linewidths=0.5,
    extent=(plot_left, plot_right, plot_bottom, plot_top),
)

plt.show()
