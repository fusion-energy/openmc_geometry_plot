import matplotlib.pyplot as plt
import openmc
import numpy as np
import plotly.graph_objects as go


def check_for_inf_value(var_name):
    if np.isinf(var_name):
        msg = f"{var_name} can't be obtained from the bounding box as boundary is at inf. {var_name} value must be specified by user"
        raise ValueError(msg)


def plot_axis_slice(
    geometry,
    view_direction="x",
    plot_left=None,
    plot_right=None,
    plot_top=None,
    plot_bottom=None,
    slice_value=None,
    pixels_across=500,
    backend="plotly",
    title=None,
):

    bb = geometry.bounding_box

    if view_direction == "x":
        # need plot_left, plot_right, plot_top, plot_bottom

        if plot_left is None:
            plot_left = bb[0][1]
            check_for_inf_value(plot_left)

        if plot_right is None:
            plot_right = bb[1][1]
            check_for_inf_value(plot_right)

        if plot_top is None:
            plot_top = bb[1][2]
            check_for_inf_value(plot_top)

        if plot_bottom is None:
            plot_bottom = bb[0][2]
            check_for_inf_value(plot_bottom)

        if slice_value is None:
            slice_value = (bb[0][0] + bb[1][0]) / 2
            check_for_inf_value(slice_value)

        xlabel = "Y [cm]"
        ylabel = "Z [cm]"

    if view_direction == "y":
        # need plot_left, plot_right, plot_top, plot_bottom

        if plot_left is None:
            plot_left = bb[0][0]
            check_for_inf_value(plot_left)

        if plot_right is None:
            plot_right = bb[1][0]
            check_for_inf_value(plot_right)

        if plot_top is None:
            plot_top = bb[1][2]
            check_for_inf_value(plot_top)

        if plot_bottom is None:
            plot_bottom = bb[0][2]
            check_for_inf_value(plot_bottom)

        if slice_value is None:
            slice_value = (bb[0][1] + bb[1][1]) / 2
            check_for_inf_value(slice_value)

        xlabel = "X [cm]"
        ylabel = "Z [cm]"

    if view_direction == "z":
        # need plot_left, plot_right, plot_top, plot_bottom

        if plot_left is None:
            plot_left = bb[0][0]
            check_for_inf_value(plot_left)

        if plot_right is None:
            plot_right = bb[1][0]
            check_for_inf_value(plot_right)

        if plot_top is None:
            plot_top = bb[1][1]
            check_for_inf_value(plot_top)

        if plot_bottom is None:
            plot_bottom = bb[0][1]
            check_for_inf_value(plot_bottom)

        if slice_value is None:
            slice_value = (bb[0][2] + bb[1][2]) / 2
            check_for_inf_value(slice_value)

        xlabel = "X [cm]"
        ylabel = "Y [cm]"

    plot_width = abs(plot_left - plot_right)
    plot_height = abs(plot_bottom - plot_top)

    aspect_ratio = plot_height / plot_width
    pixels_up = int(pixels_across * aspect_ratio)

    cell_ids = []
    material_ids = []
    for plot_y in np.linspace(plot_top, plot_bottom, pixels_up):
        row_cell_ids = []
        for plot_x in np.linspace(plot_left, plot_right, pixels_across):

            if view_direction == "z":
                found = geometry.find((plot_x, plot_y, slice_value))
            if view_direction == "x":
                found = geometry.find((slice_value, plot_x, plot_y))
            if view_direction == "y":
                found = geometry.find((plot_x, slice_value, plot_y))

            if len(found) >= 2:
                id = found[1].id
                row_cell_ids.append(id)
                if found[1].fill is not None:
                    mat = found[1].fill
                    material_ids.append(str(mat.id))
                else:
                    material_ids.append("void")
            else:
                row_cell_ids.append(0)
                material_ids.append("void")
        cell_ids.append(row_cell_ids)

    if title is None:
        # consider adding a link, does not work well in mpl
        # title = 'Made with <a href="https://github.com/fusion-energy/openmc_geometry_plot/">openmc-geometry-plot</a>'
        title = (
            f"Slice through OpenMC geometry with view direction of {view_direction}"
        )

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

        plot = plt.imshow(
            cell_ids,
            extent=(plot_left, plot_right, plot_bottom, plot_top),
            interpolation='none',
            # origin='lower', # this flips the axis incorrectly
            # cmap=cmap,
            # norm=norm,
        )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
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

