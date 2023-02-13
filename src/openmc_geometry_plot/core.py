import openmc
import numpy as np
import typing
import openmc.lib


def get_side_extent(self, side: str, view_direction:str, bounding_box=None):

    if bounding_box is None:
        bounding_box = self.bounding_box

    avail_extents = {}
    avail_extents[("left", "x")] = bounding_box[0][1]
    avail_extents[("right", "x")] = bounding_box[1][1]
    avail_extents[("top", "x")] = bounding_box[1][2]
    avail_extents[("bottom", "x")] = bounding_box[0][2]
    avail_extents[("left", "y")] = bounding_box[0][0]
    avail_extents[("right", "y")] = bounding_box[1][0]
    avail_extents[("top", "y")] = bounding_box[1][2]
    avail_extents[("bottom", "y")] = bounding_box[0][2]
    avail_extents[("left", "z")] = bounding_box[0][0]
    avail_extents[("right", "z")] = bounding_box[1][0]
    avail_extents[("top", "z")] = bounding_box[1][1]
    avail_extents[("bottom", "z")] = bounding_box[0][1]
    return avail_extents[(side, view_direction)]


def get_mpl_plot_extent(self, view_direction):
    """Returns the (x_min, x_max, y_min, y_max) of the bounding box. The
    view_direction is taken into account and can be set using
    openmc.Geometry.view_direction property is taken into account and can be
    set to 'x', 'y' or 'z'."""

    bb = self.bounding_box

    x_min = self.get_side_extent(side="left", view_direction=view_direction, bounding_box=bb)
    x_max = self.get_side_extent(side="right", view_direction=view_direction, bounding_box=bb)
    y_min = self.get_side_extent(side="bottom", view_direction=view_direction, bounding_box=bb)
    y_max = self.get_side_extent(side="top", view_direction=view_direction, bounding_box=bb)

    return (x_min, x_max, y_min, y_max)


def get_mid_slice_value(self, view_direction, bounding_box=None):
    """Returns the position of the center of the mesh. The view_direction is
    taken into account and can be set using openmc.Geometry.view_direction
    property is taken into account and can be set to 'x', 'y' or 'z'."""

    if bounding_box is None:
        bounding_box = self.bounding_box

    if view_direction == "x":
        plot_edge = (bounding_box[0][0] + bounding_box[1][0]) / 2
    elif view_direction == "y":
        plot_edge = (bounding_box[0][1] + bounding_box[1][1]) / 2
    elif view_direction == "z":
        plot_edge = (bounding_box[0][2] + bounding_box[1][2]) / 2
    else:
        msg = f'view_direction must be "x", "y" or "z" not {view_direction}'
        raise ValueError(msg)

    if np.isinf(plot_edge):
        msg = f"Mid slice value can't be obtained from the bounding box as boundary is at inf. Slice value must be specified by user"
        raise ValueError(msg)

    return plot_edge


def get_axis_labels(self, view_direction):
    """Returns two axis label values for the x and y value. Takes
    view_direction into account."""

    if view_direction == "x":
        xlabel = "Y [cm]"
        ylabel = "Z [cm]"
    if view_direction == "y":
        xlabel = "X [cm]"
        ylabel = "Z [cm]"
    if view_direction == "z":
        xlabel = "X [cm]"
        ylabel = "Y [cm]"
    return xlabel, ylabel


def find_cell_id(self, inputs):
    plot_x, plot_y, slice_value = inputs
    return self.find((plot_x, plot_y, slice_value))


def get_slice_of_material_ids(
    self,
    view_direction: str,
    slice_value: typing.Optional[float]=None,
    plot_top:typing.Optional[float]=None,
    plot_bottom:typing.Optional[float]=None,
    plot_left:typing.Optional[float]=None,
    plot_right:typing.Optional[float]=None,
    pixels_across: int=500,
):
    """Returns a grid of material IDs for each mesh voxel on the slice. This
    can be passed directly to plotting functions like Matplotlib imshow.

    Args:
        slice_value:
        plot_top:
        plot_bottom:
        plot_left:
        plot_right:
        pixels_across:
    """

    # if any of the plot_ are None then this needs calculating
    bb = self.bounding_box

    if view_direction not in ["x", "y", "z"]:
        raise ValueError('view_direction must be "x", "y" or "z"')

    if plot_left is None:
        plot_left = self.get_side_extent(side="left", bounding_box=bb, view_direction=view_direction)

    if plot_right is None:
        plot_right = self.get_side_extent(side="right", bounding_box=bb, view_direction=view_direction)

    if plot_bottom is None:
        plot_bottom = self.get_side_extent(side="bottom", bounding_box=bb, view_direction=view_direction)

    if plot_top is None:
        plot_top = self.get_side_extent(side="top", bounding_box=bb, view_direction=view_direction)

    if slice_value is None:
        slice_value = self.get_mid_slice_value(bounding_box=bb, view_direction=view_direction)

    plot_width = abs(plot_left - plot_right)
    plot_height = abs(plot_bottom - plot_top)

    aspect_ratio = plot_height / plot_width
    pixels_up = int(pixels_across * aspect_ratio)

    materials = self.get_all_materials()
    mat_ids = materials.keys()

    all_materials = []
    for i in mat_ids:
        print(i, mat_ids)
        n = openmc.Material()
        n.id = i
        n.add_nuclide('He4', 1)
        all_materials.append(n)
    nn = openmc.Materials(all_materials)
    nn.export_to_xml()
    self.export_to_xml()

    my_settings = openmc.Settings()
    my_settings.output = {'summary': False, 'tallies': False}
    my_settings.particles=1
    my_settings.batches=1
    my_settings.batches=1
    my_settings.run_mode = 'fixed source'
    my_settings.export_to_xml()

    my_plot = openmc.Plot()

    my_plot.basis = 'xz'
    my_plot.origin = (5.0, 2.0, 3.0)
    my_plot.width = (50., 50.)
    my_plot.pixels = (400, 400)

    # my_plot.colors = {
    #     water: (0, 0, 255),
    #     clad: (0, 0, 0)
    # }
    my_plots = openmc.Plots([my_plot])
    my_plots.export_to_xml()
    openmc.plot_geometry()

    # material_ids = []
    # for plot_y in np.linspace(plot_top, plot_bottom, pixels_up):
    #     row_material_ids = []
    #     for plot_x in np.linspace(plot_left, plot_right, pixels_across):

    #         try:
                    
    #             if view_direction == "z":
    #                 found = openmc.lib.find_material((plot_x, plot_y, slice_value))
    #                 # found = self.find((plot_x, plot_y, slice_value))
    #             if view_direction == "x":
    #                 found = openmc.lib.find_material((slice_value, plot_x, plot_y))
    #                 # found = self.find((slice_value, plot_x, plot_y))
    #             if view_direction == "y":
    #                 found = openmc.lib.find_material((plot_x, slice_value, plot_y))
    #                 # found = self.find((plot_x, slice_value, plot_y))

    #             if found == None:
    #                 found = 0
    #             else:
    #                 found = found.id
    #         except openmc.exceptions.GeometryError:
    #             found = 0
    #         #     print(found)
    #         #     print(found.id)
    #         #     input()
    #         row_material_ids.append(found)
    #         # if len(found) >= 2:
    #         #     if found[1].fill is not None:
    #         #         mat = found[1].fill
    #         #         row_material_ids.append(mat.id)
    #         #     else:
    #         #         row_material_ids.append(0)  # when material is "void"
    #         # else:
    #         #     row_material_ids.append(0)  # when material is "void"
    #     material_ids.append(row_material_ids)
    
    # openmc.lib.finalize()
    # return material_ids


def get_slice_of_cell_ids(
    self,
    view_direction: str,
    slice_value=None,
    plot_top=None,
    plot_bottom=None,
    plot_left=None,
    plot_right=None,
    pixels_across=500,
):
    """Returns a grid of cell IDs for each mesh voxel on the slice. This
    can be passed directly to plotting functions like Matplotlib imshow.

    Args:
        slice_value:
        plot_top:
        plot_bottom:
        plot_left:
        plot_right:
        pixels_across:
    """

    bb = self.bounding_box

    if view_direction not in ["x", "y", "z"]:
        raise ValueError('view_direction must be "x", "y" or "z"')

    if plot_left is None:
        plot_left = self.get_side_extent(side="left", bounding_box=bb, view_direction=view_direction)

    if plot_right is None:
        plot_right = self.get_side_extent(side="right", bounding_box=bb, view_direction=view_direction)

    if plot_bottom is None:
        plot_bottom = self.get_side_extent(side="bottom", bounding_box=bb, view_direction=view_direction)

    if plot_top is None:
        plot_top = self.get_side_extent(side="top", bounding_box=bb, view_direction=view_direction)

    if slice_value is None:
        slice_value = self.get_mid_slice_value(bounding_box=bb, view_direction=view_direction)

    plot_width = abs(plot_left - plot_right)
    plot_height = abs(plot_bottom - plot_top)

    aspect_ratio = plot_height / plot_width
    pixels_up = int(pixels_across * aspect_ratio)

    cell_ids = []
    for plot_y in np.linspace(plot_top, plot_bottom, pixels_up):
        row_cell_ids = []
        for plot_x in np.linspace(plot_left, plot_right, pixels_across):

            if view_direction == "z":
                found = self.find((plot_x, plot_y, slice_value))
            if view_direction == "x":
                found = self.find((slice_value, plot_x, plot_y))
            if view_direction == "y":
                found = self.find((plot_x, slice_value, plot_y))

            if len(found) >= 2:
                id = found[1].id
                row_cell_ids.append(id)
            else:
                row_cell_ids.append(0)
        cell_ids.append(row_cell_ids)
    return cell_ids


# patching openmc

openmc.Geometry.get_side_extent = get_side_extent
openmc.Geometry.get_mpl_plot_extent = get_mpl_plot_extent
openmc.Geometry.get_mid_slice_value = get_mid_slice_value
openmc.Geometry.get_axis_labels = get_axis_labels
openmc.Geometry.get_slice_of_material_ids = get_slice_of_material_ids
openmc.Geometry.get_slice_of_cell_ids = get_slice_of_cell_ids
openmc.Geometry.find_cell_id = find_cell_id

# setting default view direction
openmc.Geometry.viewdirection = "x"
