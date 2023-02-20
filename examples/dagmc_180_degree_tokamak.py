from pathlib import Path

import matplotlib.pylab as plt
import numpy as np
import openmc
import openmc_geometry_plot  # adds plot_axis_slice to openmc.Geometry

# minimal openmc.Geoemtry made just from a single DAGMC file with a bounding surface
bound_dag_univ = openmc.DAGMCUniverse(
    filename="dagmc_180_tokamak.h5m"
).bounded_universe()
my_geometry = openmc.Geometry(root=bound_dag_univ)


# example code for plotting materials of the geometry with an outline

data_slice = my_geometry.get_slice_of_material_ids(
    view_direction="y", slice_value=1, pixels_across=1000
)

xlabel, ylabel = my_geometry.get_axis_labels(view_direction="y")
plt.xlabel(xlabel)
plt.ylabel(ylabel)

plot_extent = my_geometry.get_mpl_plot_extent(view_direction="y")

# shows the materials in random colors
plt.imshow(
    data_slice,
    extent=plot_extent,
    interpolation="none",
)

# gets unique levels for outlines contour plot
levels = np.unique([item for sublist in data_slice for item in sublist])

# shows the material outlines
plt.contour(
    data_slice,
    origin="upper",
    colors="k",
    linestyles="solid",
    levels=levels,
    linewidths=2,
    extent=plot_extent,
)

plt.show()
