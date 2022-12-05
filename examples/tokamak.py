# plots a geometry with both backends (matplotlib and plotly)
# A couple of zoomed in plots are plotted
# All view angles (x, y, z) for the whole geometry are plotted


import openmc
from openmc_geometry_plot import plot_axis_slice


breeder_material = openmc.Material()
breeder_material.add_element("Li", 1.0)
breeder_material.set_density("g/cm3", 0.01)

copper_material = openmc.Material()
copper_material.set_density("g/cm3", 0.01)
copper_material.add_element("Li", 1.0)

eurofer_material = openmc.Material()
eurofer_material.set_density("g/cm3", 0.01)
eurofer_material.add_element("Li", 1)

# surfaces
central_sol_surface = openmc.ZCylinder(r=100)
central_shield_outer_surface = openmc.ZCylinder(r=110)
# plasma_surface = openmc.ZTorus(x0=200,y0=200)
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

universe = openmc.Universe(
    cells=[
        central_sol_cell,
        central_shield_cell,
        inner_vessel_cell,
        first_wall_cell,
        breeder_blanket_cell,
        port_hole_cell,
        upper_port_hole_cell,
    ]
)
my_geometry = openmc.Geometry(universe)


for backend in ["matplotlib", "plotly"]:

    for view_direction in ["z", "x", "y"]:
        # plots the geometry
        plot = plot_axis_slice(
            geometry=my_geometry, view_direction=view_direction, backend=backend
        )

        plot.show()

    # a single zoomed in plot
    plot = plot_axis_slice(
        geometry=my_geometry,
        plot_left=400,
        plot_right=600,
        plot_top=-200,
        plot_bottom=100,
        view_direction="y",
        backend=backend,
    )

    plot.show()
