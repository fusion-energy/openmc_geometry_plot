import openmc
import numpy as np


def get_side_extent(self, side, bb=None):

    if bb is None:
        bb = self.bounding_box

    avail_extents = {}
    avail_extents[("left", "x")] = bb[0][1]
    avail_extents[("right", "x")] = bb[1][1]
    avail_extents[("top", "x")] = bb[1][2]
    avail_extents[("bottom", "x")] = bb[0][2]
    avail_extents[("left", "y")] = bb[0][0]
    avail_extents[("right", "y")] = bb[1][0]
    avail_extents[("top", "y")] = bb[1][2]
    avail_extents[("bottom", "y")] = bb[0][2]
    avail_extents[("left", "z")] = bb[0][0]
    avail_extents[("right", "z")] = bb[1][0]
    avail_extents[("top", "z")] = bb[1][1]
    avail_extents[("bottom", "z")] = bb[0][1]
    return avail_extents[(side, self.view_direction)]


def get_mpl_plot_extent(self, bb=None):

    if bb is None:
        bb = self.bounding_box

    x_min = self.get_side_extent("left", bb)
    x_max = self.get_side_extent("right", bb)
    y_min = self.get_side_extent("bottom", bb)
    y_max = self.get_side_extent("top", bb)

    return (x_min, x_max, y_min, y_max)


def get_mid_slice_value(self, bb=None):

    if bb is None:
        bb = self.bounding_box

    if self.view_direction == "x":
        plot_edge = (bb[0][0] + bb[1][0]) / 2
    elif self.view_direction == "y":
        plot_edge = (bb[0][1] + bb[1][1]) / 2
    elif self.view_direction == "z":
        plot_edge = (bb[0][2] + bb[1][2]) / 2
    else:
        msg = f'view_direction must be "x", "y" or "z" not {self.view_direction}'
        raise ValueError(msg)

    if np.isinf(plot_edge):
        msg = f"{var_name} can't be obtained from the bounding box as boundary is at inf. {var_name} value must be specified by user"

        raise ValueError(msg)

    return plot_edge


def get_axis_labels(self):
    if self.view_direction == "x":
        xlabel = "Y [cm]"
        ylabel = "Z [cm]"
    if self.view_direction == "y":
        xlabel = "X [cm]"
        ylabel = "Z [cm]"
    if self.view_direction == "z":
        xlabel = "X [cm]"
        ylabel = "Y [cm]"
    return xlabel, ylabel


def get_slice_of_material_ids(
    self,
    slice_value=None,
    plot_top=None,
    plot_bottom=None,
    plot_left=None,
    plot_right=None,
    pixels_across=500,
):
    # if any of the plot_ are None then this needs calculating
    bb = self.bounding_box

    if self.view_direction not in ["x", "y", "z"]:
        raise ValueError('view_direction must be "x", "y" or "z"')

    if plot_left is None:
        plot_left = self.get_side_extent("left", bb)

    if plot_right is None:
        plot_right = self.get_side_extent("right", bb)

    if plot_bottom is None:
        plot_bottom = self.get_side_extent("bottom", bb)

    if plot_top is None:
        plot_top = self.get_side_extent("top", bb)

    if slice_value is None:
        slice_value = self.get_mid_slice_value(bb)

    plot_width = abs(plot_left - plot_right)
    plot_height = abs(plot_bottom - plot_top)

    aspect_ratio = plot_height / plot_width
    pixels_up = int(pixels_across * aspect_ratio)

    # todo look into parrallel version of this
    # import multiprocessing.pool
    # global pool
    # pool = multiprocessing.Pool(4)
    # pool = multiprocessing.Semaphore(multiprocessing.cpu_count() -1)
    # out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))

    material_ids = []
    for plot_y in np.linspace(plot_top, plot_bottom, pixels_up):
        row_material_ids = []
        for plot_x in np.linspace(plot_left, plot_right, pixels_across):

            if self.view_direction == "z":
                found = self.find((plot_x, plot_y, slice_value))
            if self.view_direction == "x":
                found = self.find((slice_value, plot_x, plot_y))
            if self.view_direction == "y":
                found = self.find((plot_x, slice_value, plot_y))

            if len(found) >= 2:
                if found[1].fill is not None:
                    mat = found[1].fill
                    row_material_ids.append(mat.id)
                else:
                    row_material_ids.append(0)  # when material is "void"
            else:
                row_material_ids.append(0)  # when material is "void"
        material_ids.append(row_material_ids)
    return material_ids


def get_slice_of_cell_ids(
    self,
    slice_value=None,
    plot_top=None,
    plot_bottom=None,
    plot_left=None,
    plot_right=None,
    pixels_across=500,
):

    bb = self.bounding_box

    if self.view_direction not in ["x", "y", "z"]:
        raise ValueError('view_direction must be "x", "y" or "z"')

    if plot_left is None:
        plot_left = self.get_side_extent("left", bb)

    if plot_right is None:
        plot_right = self.get_side_extent("right", bb)

    if plot_bottom is None:
        plot_bottom = self.get_side_extent("bottom", bb)

    if plot_top is None:
        plot_top = self.get_side_extent("top", bb)

    if slice_value is None:
        slice_value = self.get_mid_slice_value(bb)

    plot_width = abs(plot_left - plot_right)
    plot_height = abs(plot_bottom - plot_top)

    aspect_ratio = plot_height / plot_width
    pixels_up = int(pixels_across * aspect_ratio)

    cell_ids = []
    for plot_y in np.linspace(plot_top, plot_bottom, pixels_up):
        row_cell_ids = []
        for plot_x in np.linspace(plot_left, plot_right, pixels_across):

            if self.view_direction == "z":
                found = self.find((plot_x, plot_y, slice_value))
            if self.view_direction == "x":
                found = self.find((slice_value, plot_x, plot_y))
            if self.view_direction == "y":
                found = self.find((plot_x, slice_value, plot_y))

            if len(found) >= 2:
                id = found[1].id
                row_cell_ids.append(id)
            else:
                row_cell_ids.append(0)
        cell_ids.append(row_cell_ids)
    return cell_ids


openmc.Geometry.get_side_extent = get_side_extent
openmc.Geometry.get_mpl_plot_extent = get_mpl_plot_extent
openmc.Geometry.get_mid_slice_value = get_mid_slice_value
openmc.Geometry.get_axis_labels = get_axis_labels
openmc.Geometry.get_slice_of_material_ids = get_slice_of_material_ids
openmc.Geometry.get_slice_of_cell_ids = get_slice_of_cell_ids

openmc.Geometry.viewdirection = "x"
