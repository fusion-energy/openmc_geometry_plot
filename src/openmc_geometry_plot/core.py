import matplotlib.pyplot as plt
import openmc
import numpy as np
import plotly.graph_objects as go

def check_for_inf_value(var_name, view_direction):
    if np.isinf(var_name):
        # print(f'view direction {view_direction}\n')
        msg = f"{var_name} can't be obtained from the bounding box as boundary is at inf. {var_name} value must be specified by user"

        raise ValueError(msg)


def get_side_extent(bb, side, view_direction):

    avail_extents = {}
    avail_extents[('left', 'x')]= bb[0][1]
    avail_extents[('right', 'x')]= bb[1][1]
    avail_extents[('top', 'x')]= bb[1][2]
    avail_extents[('bottom', 'x')]= bb[0][2]
    avail_extents[('left', 'y')]= bb[0][0]
    avail_extents[('right', 'y')]= bb[1][0]
    avail_extents[('top', 'y')]= bb[1][2]
    avail_extents[('bottom', 'y')]= bb[0][2]
    avail_extents[('left', 'z')]= bb[0][0]
    avail_extents[('right', 'z')]= bb[1][0]
    avail_extents[('top', 'z')]= bb[1][1]
    avail_extents[('bottom', 'z')]= bb[0][1]
    return avail_extents[(side,view_direction)]


def get_mpl_plot_extent(bb, view_direction):
    x_min = get_side_extent(bb, 'left', view_direction)
    x_max = get_side_extent(bb, 'right', view_direction)
    y_min = get_side_extent(bb, 'bottom', view_direction)
    y_max = get_side_extent(bb, 'top', view_direction)

    return (x_min, x_max, y_min, y_max)
    


def get_mid_slice_value(bb, view_direction):

    if view_direction == "x":
        plot_edge = (bb[0][0] + bb[1][0]) / 2
    elif view_direction == "y":
        plot_edge = (bb[0][1] + bb[1][1]) / 2
    elif view_direction == "z":
        plot_edge = (bb[0][2] + bb[1][2]) / 2
    else:
        raise ValueError('view_direction must be "x", "y" or "z"')

    check_for_inf_value(plot_edge, view_direction)
    return plot_edge


def get_axis_labels(view_direction):
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


class Geometry(openmc.Geometry):

    def get_slice_of_material_ids(
        self,
        view_direction,
        slice_value=None,
        plot_top=None,
        plot_bottom=None,
        plot_left=None,
        plot_right=None,
        pixels_across=200,
    ):
        bb = self.bounding_box

        if view_direction not in ["x", "y", "z"]:
            raise ValueError('view_direction must be "x", "y" or "z"')

        if plot_left is None:
            plot_left = get_side_extent(bb, 'left', view_direction)

        if plot_right is None:
            plot_right = get_side_extent(bb, 'right', view_direction)

        if plot_bottom is None:
            plot_bottom = get_side_extent(bb, 'bottom', view_direction)

        if plot_top is None:
            plot_top = get_side_extent(bb, 'top', view_direction)

        if slice_value is None:
            slice_value = get_mid_slice_value(bb, view_direction)

        plot_width = abs(plot_left - plot_right)
        plot_height = abs(plot_bottom - plot_top)

        aspect_ratio = plot_height / plot_width
        pixels_up = int(pixels_across * aspect_ratio)

        material_ids = []
        for plot_y in np.linspace(plot_top, plot_bottom, pixels_up):
            row_material_ids = []
            for plot_x in np.linspace(plot_left, plot_right, pixels_across):

                if view_direction == "z":
                    found = self.find((plot_x, plot_y, slice_value))
                if view_direction == "x":
                    found = self.find((slice_value, plot_x, plot_y))
                if view_direction == "y":
                    found = self.find((plot_x, slice_value, plot_y))

                if len(found) >= 2:
                    if found[1].fill is not None:
                        mat = found[1].fill
                        row_material_ids.append(mat.id)
                    else:
                        row_material_ids.append(0)  # "void")
                else:
                    row_material_ids.append(0)  # "void")
            material_ids.append(row_material_ids)
        return material_ids

    def get_slice_of_cell_ids(self,view_direction, slice_value, plot_top, plot_bottom, pixels_up, plot_left, plot_right, pixels_across):
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

    def plot_axis_slice(
        self,
        view_direction="x",
        plot_left=None,
        plot_right=None,
        plot_top=None,
        plot_bottom=None,
        slice_value=None,
        pixels_across=200,
        backend="plotly",
        title=None,
        color_by="cells",
        outline='cells'
    ):

        bb = self.bounding_box

        if plot_left is None:
            plot_left = get_side_extent(bb, 'left', view_direction)

        if plot_right is None:
            plot_right = get_side_extent(bb, 'right', view_direction)

        if plot_bottom is None:
            plot_bottom = get_side_extent(bb, 'bottom', view_direction)

        if plot_top is None:
            plot_top = get_side_extent(bb, 'top', view_direction)

        if slice_value is None:
            slice_value = get_mid_slice_value(bb, view_direction)

        xlabel, ylabel = get_axis_labels(view_direction)

        plot_width = abs(plot_left - plot_right)
        plot_height = abs(plot_bottom - plot_top)

        aspect_ratio = plot_height / plot_width
        pixels_up = int(pixels_across * aspect_ratio)

        if 'materials' in [color_by, outline]:
            material_ids = self.get_slice_of_material_ids(view_direction, slice_value, plot_top, plot_bottom, pixels_up, plot_left, plot_right, pixels_across)
        if 'cells' in [color_by, outline]:
            cell_ids = self.get_slice_of_cell_ids(view_direction, slice_value, plot_top, plot_bottom, pixels_up, plot_left, plot_right, pixels_across)

        if title is None:
            # consider adding a link, does not work well in mpl
            # title = 'Made with <a href="https://github.com/fusion-energy/openmc_geometry_plot/">openmc-geometry-plot</a>'
            title = f"Slice through OpenMC geometry with view direction of {view_direction}"

        if backend == "matplotlib":

            # TODO color picker not working for 2 colors
            # from matplotlib import colors
            # cmap = plt.cm.jet
            # cmaplist = [cmap(i) for i in range(cmap.N)]
            # cmaplist.append((1., 1., 1., 0.))
            # cmap = colors.LinearSegmentedColormap.from_list(
            #     'Custom cmap', cmaplist, cmap.N
            # )
            # bounds = list(geometry.get_all_cells().keys())  + [0]
            # print(bounds)
            # bounds.sort()
            # print(bounds)
            # norm = colors.BoundaryNorm(bounds, cmap.N)

            # outline_levels = np.unique(geometry.ids)

            if color_by == "cells":
                plot_data = cell_ids
            elif color_by == "materials":
                plot_data = material_ids
            else:
                msg = f"only materials or cells are acceptable values for color_by, not {color_by}"
                raise ValueError(msg)

            plot = plt.imshow(
                plot_data,
                extent=(plot_left, plot_right, plot_bottom, plot_top),
                interpolation="none",
                # origin='lower', # this flips the axis incorrectly
                # cmap=cmap,
                # norm=norm,
            )
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)

            if outline is not None:

                if outline == "cells":
                    outline_data = cell_ids
                if outline == "materials":
                    outline_data = material_ids
                self.get_outline_contour(
                    outline, outline_data,
                    plot_left, plot_right, plot_bottom, plot_top)

            return plot

        elif backend == "plotly":

            plot = go.Figure(
                data=go.Heatmap(
                    z=cell_ids,
                    colorscale="viridis",
                    x0=plot_left,
                    dx=abs(plot_left - plot_right) / (len(cell_ids[0]) - 1),
                    y0=plot_bottom,
                    dy=abs(plot_bottom - plot_top) / (len(cell_ids) - 1),
                    # colorbar=dict(title=dict(side="right", text=cbar_label)),
                    # text = material_ids,
                    hovertemplate=
                    # 'material ID = %{z}<br>'+
                    "Cell ID = %{z}<br>" +
                    # '<br>%{text}<br>'+
                    xlabel[:2].title()
                    + ": %{x} cm<br>"
                    + ylabel[:2].title()
                    + ": %{y} cm<br>",
                ),
            )

            plot.update_layout(
                xaxis={"title": xlabel},
                # reversed autorange is required to avoid image needing rotation/flipping in plotly
                yaxis={"title": ylabel, "autorange": "reversed"},
                title=title,
                autosize=False,
                height=800,
            )
            plot.update_yaxes(
                scaleanchor="x",
                scaleratio=1,
            )
            return plot

        else:
            raise ValueError(
                f"Supported backend are 'plotly' and 'matplotlib', not {backend}"
            )


    def get_outline_contour(
        self,
        outline_data,
        view_direction,
        plot_left=None,
        plot_right=None,
        plot_bottom=None,
        plot_top=None
    ):

        bb = self.bounding_box

        if plot_left is None:
            plot_left = get_side_extent(bb, 'left', view_direction)

        if plot_right is None:
            plot_right = get_side_extent(bb, 'right', view_direction)

        if plot_bottom is None:
            plot_bottom = get_side_extent(bb, 'bottom', view_direction)

        if plot_top is None:
            plot_top = get_side_extent(bb, 'top', view_direction)

        levels = np.unique([item for sublist in outline_data for item in sublist])

        plot = plt.contour(
            outline_data,
            origin="upper",
            colors="k",
            linestyles="solid",
            levels=levels,
            linewidths=0.5,
            extent=(plot_left, plot_right, plot_bottom, plot_top),
        )
        return plot


openmc.Geometry = Geometry

