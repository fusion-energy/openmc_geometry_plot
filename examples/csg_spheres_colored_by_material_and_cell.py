import matplotlib.patches as mpatches
import matplotlib.pylab as plt
import numpy as np
import openmc
import openmc_geometry_plot  # adds functions to openmc.Geometry
from matplotlib import colors

# MATERIALS

mat = openmc.Material()
mat.set_density("g/cm3", 1)
mat.id = 1
mat.name = "lead"
mat.add_element("Pb", 1)

mats = openmc.Materials([mat])

# GEOMETRY

# surfaces
surf1 = openmc.Sphere(r=100)
surf2 = openmc.Sphere(r=200)
surf3 = openmc.Sphere(r=300, boundary_type="vacuum")

# regions
region1 = -surf1
region2 = +surf1 & -surf2
region3 = +surf2 & -surf3

# cells
cell1 = openmc.Cell(region=region1)
cell1.id = 1
cell1.name = "cell 1"
# cell 1 is not filled with any material
cell2 = openmc.Cell(region=region2)
cell2.id = 2
cell2.fill = mat
cell2.name = "cell 2"
cell3 = openmc.Cell(region=region3)
cell3.fill = mat
cell3.id = 3
cell3.name = "cell 3"

my_geometry = openmc.Geometry([cell1, cell2, cell3])

fig, (ax1, ax2) = plt.subplots(1, 2)

# these three lines are functionality added by the openmc_geometry_plot
xlabel, ylabel = my_geometry.get_axis_labels(view_direction="x")

ax1.set_xlabel(xlabel)
ax1.set_ylabel(ylabel)
ax1.set_title("material ids")
ax2.set_xlabel(xlabel)
ax2.set_ylabel(ylabel)
ax2.set_title("cell ids")

mat_id_slice = my_geometry.get_slice_of_material_ids(
    view_direction="x", pixels_across=600
)
cell_id_slice = my_geometry.get_slice_of_cell_ids(view_direction="x", pixels_across=600)

# gets unique levels for outlines contour plot and for the color scale
mat_levels = np.unique([item for sublist in mat_id_slice for item in sublist])
cell_levels = np.unique([item for sublist in cell_id_slice for item in sublist])

mat_cmap = colors.ListedColormap(["white", "red"])
# our material ids are set to be 1 and 2. void space is 0
mat_bounds = [0, 1, 2]
mat_norm = colors.BoundaryNorm(mat_bounds, mat_cmap.N)

cell_cmap = colors.ListedColormap(["white", "blue", "green", "yellow"])
# our cell ids are set to be 1, 2 and 3. void space or undefined is 0
cell_bounds = [0, 1, 2, 3, 4]
cell_norm = colors.BoundaryNorm(cell_bounds, cell_cmap.N)

# gets the extent of the geometry to pass to matplotlib
plot_extent = my_geometry.get_mpl_plot_extent(view_direction="x")

# shows the mat ids with selected colors
im_mat = ax1.imshow(
    mat_id_slice,
    extent=plot_extent,
    interpolation="none",
    norm=mat_norm,  # needed for colors
    cmap=mat_cmap,  # needed for colors
)

# shows the cell ids with selected colors
im_cell = ax2.imshow(
    cell_id_slice,
    extent=plot_extent,
    interpolation="none",
    norm=cell_norm,  # needed for colors
    cmap=cell_cmap,  # needed for colors
)

# adds contour lines on material boundaries
ax1.contour(
    mat_id_slice,
    origin="upper",
    colors="k",
    linestyles="solid",
    levels=mat_levels,
    linewidths=1,
    extent=plot_extent,
)
# adds contour lines on cell boundaries
ax2.contour(
    cell_id_slice,
    origin="upper",
    colors="k",
    linestyles="solid",
    levels=cell_levels,
    linewidths=1,
    extent=plot_extent,
)

# code from here onwards adds a legend for both materials and cells

# gets the colors of the values according to the color map used by imshow
colors_mat = [im_mat.cmap(im_mat.norm(value)) for value in mat_levels]
colors_cells = [im_cell.cmap(im_cell.norm(value)) for value in cell_levels]

# these will be the labels used for each legend entry
mat_labels = {1: "mat1 lead"}
cell_labels = {1: "cell 1", 2: "cell 2", 3: "cell 3"}

mat_patches = [
    mpatches.Patch(color=colors_mat[i], label=mat_labels[mat_levels[i]])
    for i in range(len(mat_levels))[1:]  # skips the first 0 value
]
cell_patches = [
    mpatches.Patch(color=colors_cells[i], label=cell_labels[cell_levels[i]])
    for i in range(len(cell_levels))[1:]  # skips the first 0 value
]

ax1.legend(handles=mat_patches, bbox_to_anchor=(0.5, -0.55), loc="lower center")
ax2.legend(handles=cell_patches, bbox_to_anchor=(0.5, -0.55), loc="lower center")

plt.show()
