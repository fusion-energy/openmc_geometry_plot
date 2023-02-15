import numpy as np
import matplotlib.pylab as plt
import openmc
import openmc_geometry_plot  # adds functions to openmc.Geometry
from matplotlib import colors

# MATERIALS

mat1 = openmc.Material()
mat1.set_density("g/cm3", 1)
mat1.id = 1
mat1.add_element("Pb", 1)

mat2 = openmc.Material()
mat2.id = 2
mat2.set_density("g/cm3", 1)
mat2.add_element("Fe", 1)

mats = openmc.Materials([mat1, mat2])

# GEOMETRY

# surfaces
vessel_inner = openmc.Sphere(r=100)
first_wall_outer_surface = openmc.Sphere(r=200)
breeder_blanket_outer_surface = openmc.Sphere(r=300, boundary_type="vacuum")

# cells
inner_vessel_region = -vessel_inner
inner_vessel_cell = openmc.Cell(region=inner_vessel_region)

first_wall_region = -first_wall_outer_surface & +vessel_inner
first_wall_cell = openmc.Cell(region=first_wall_region)
first_wall_cell.fill = mat2

breeder_blanket_region = +first_wall_outer_surface & -breeder_blanket_outer_surface
breeder_blanket_cell = openmc.Cell(region=breeder_blanket_region)
breeder_blanket_cell.fill = mat1

my_geometry = openmc.Geometry(
    [inner_vessel_cell, first_wall_cell, breeder_blanket_cell]
)


# this part plots the geometry with colors

data_slice = my_geometry.get_slice_of_material_ids(view_direction="z")

xlabel, ylabel = my_geometry.get_axis_labels(view_direction="z")
plt.xlabel(xlabel)
plt.ylabel(ylabel)

# gets unique levels for outlines contour plot and for the color scale
levels = np.unique([item for sublist in data_slice for item in sublist])

cmap = colors.ListedColormap(["white", "red", "blue"])
# our material ids are set to be 1 and 2. void space is 0
bounds = [0, 1, 2, 3]
norm = colors.BoundaryNorm(bounds, cmap.N)

plot_extent = my_geometry.get_mpl_plot_extent(view_direction="z")
plt.imshow(
    data_slice,
    extent=plot_extent,
    interpolation="none",
    norm=norm,  # needed for colors
    cmap=cmap,  # needed for colors
)


plt.contour(
    data_slice,
    origin="upper",
    colors="k",
    linestyles="solid",
    levels=levels,
    linewidths=0.5,
    extent=plot_extent,
)

plt.show()
