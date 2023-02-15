# plots a geometry with both backends (matplotlib and plotly)
# A couple of zoomed in plots are plotted
# All view angles (x, y, z) for the whole geometry are plotted


import openmc
import openmc_geometry_plot  # adds plot_axis_slice to openmc.Geometry
from pathlib import Path

breeder_material = openmc.Material()
breeder_material.add_element("Li", 1.0)
breeder_material.set_density("g/cm3", 0.01)

copper_material = openmc.Material()
copper_material.set_density("g/cm3", 0.01)
copper_material.add_element("Cu", 1.0)

eurofer_material = openmc.Material()
eurofer_material.set_density("g/cm3", 0.01)
eurofer_material.add_element("Fe", 1)

bound_dag_univ = openmc.DAGMCUniverse(filename='dagmc.h5m').bounded_universe()
my_geometry = openmc.Geometry(root=bound_dag_univ)

# example code for plotting materials of the geometry with an outline

import numpy as np
import matplotlib.pylab as plt


data_slice = my_geometry.get_slice_of_material_ids(
    view_direction="x",
    slice_value=1,
    plot_left=-500

)
xlabel, ylabel = my_geometry.get_axis_labels(view_direction="x")
plt.xlabel(xlabel)
plt.ylabel(ylabel)

plot_extent = my_geometry.get_mpl_plot_extent(view_direction="x")

plt.imshow(
    data_slice,
    extent=plot_extent,
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
    extent=plot_extent,
)

plt.show()
