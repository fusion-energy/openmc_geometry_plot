
import openmc
from openmc_geometry_plot import plot_axis_slice



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

universe = openmc.Universe(
    cells=[
        cell_1, cell_2
    ]
)
my_geometry = openmc.Geometry(universe)


for backend in ["matplotlib", "plotly"]:

    plot = plot_axis_slice(
        geometry=my_geometry, view_direction="z", backend=backend
    )

    plot.show()

for backend in ["matplotlib", "plotly"]:

    for view_direction in ["x", "y"]:

        # a single zoomed in plot, ranges must be specified as the geometry is not fully bound
        plot = plot_axis_slice(
            geometry=my_geometry,
            plot_left=-100,
            plot_right=100,
            plot_top=-200,
            plot_bottom=200,
            view_direction=view_direction,
            backend=backend,
        )

        plot.show()
