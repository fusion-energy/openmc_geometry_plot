# plots a geometry with both backends (matplotlib and plotly)
# A couple of zoomed in plots are plotted
# All view angles (x, y, z) for the whole geometry are plotted


import openmc

# dimentions are in cm
height = 150
outer_radius = 50
thickness = 10

outer_cylinder = openmc.ZCylinder(r=outer_radius, boundary_type="vacuum")
inner_cylinder = openmc.ZCylinder(r=outer_radius - thickness, boundary_type="vacuum")
inner_top = openmc.ZPlane(z0=height * 0.5, boundary_type="vacuum")
inner_bottom = openmc.ZPlane(z0=-height * 0.5, boundary_type="vacuum")
outer_top = openmc.ZPlane(z0=(height * 0.5) + thickness, boundary_type="vacuum")
outer_bottom = openmc.ZPlane(z0=(-height * 0.5) - thickness, boundary_type="vacuum")

steel = openmc.Material()
steel.set_density("g/cm3", 7.75)
steel.add_element("Fe", 0.95, percent_type="wo")
steel.add_element("C", 0.05, percent_type="wo")

mats = openmc.Materials([steel])

cylinder_region = -outer_cylinder & +inner_cylinder & -inner_top & +inner_bottom
cylinder_cell = openmc.Cell(region=cylinder_region)
cylinder_cell.fill = steel

top_cap_region = -outer_top & +inner_top & -outer_cylinder
top_cap_cell = openmc.Cell(region=top_cap_region)
top_cap_cell.fill = steel

bottom_cap_region = +outer_bottom & -inner_bottom & -outer_cylinder
bottom_cap_cell = openmc.Cell(region=bottom_cap_region)
bottom_cap_cell.fill = steel

inner_void_region = -inner_cylinder & -inner_top & +inner_bottom
inner_void_cell = openmc.Cell(region=inner_void_region)

my_geometry = openmc.Geometry(
    [inner_void_cell, cylinder_cell, top_cap_cell, bottom_cap_cell]
)


# example code for plotting materials of the geometry with an outline

import numpy as np
import matplotlib.pylab as plt
import openmc_geometry_plot  # adds plot_axis_slice to openmc.Geometry


my_geometry.view_direction = "x"
data_slice = my_geometry.get_slice_of_cell_ids()

plt.imshow(
    data_slice,
    extent=my_geometry.get_mpl_plot_extent(),
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
    extent=my_geometry.get_mpl_plot_extent(),
)

plt.show()
