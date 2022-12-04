import matplotlib.pyplot as plt
import openmc
import numpy as np
import plotly.graph_objects as go


def check_for_inf_value(var_name):
    if np.isinf(var_name):
        msg=f"{var_name} can't be obtained from the bounding box as boundary is at inf. {var_name} value must be specified by user"
        raise ValueError(msg)

def plot_axis_slice(
    geometry,
    view_direction,
    plot_left,
    plot_right,
    plot_top,
    plot_bottom,
    slice_value,
    pixels_across=200,
    backend='plotly'
):

    bb = geometry.bounding_box
    
    if view_direction == "z":
        # need plot_left, plot_right, plot_top, plot_bottom
    
        if plot_left is None:
            plot_left = bb[0][0]
            check_for_inf_value(plot_left)
    
        if plot_right is None:
            plot_right = bb[1][0]
            check_for_inf_value(plot_right)
    
        if plot_top is None:
            plot_top = bb[0][1]
            check_for_inf_value(plot_top)
    
        if plot_bottom is None:
            plot_bottom = bb[1][1]
            check_for_inf_value(plot_bottom)

        xlabel = "X [cm]"
        ylabel = "Z [cm]"

    plot_width = abs(plot_left-plot_right)
    plot_height = abs(plot_bottom-plot_top)

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
            else:
                row_cell_ids.append(0)
        cell_ids.append(row_cell_ids)

    if backend == 'matplotlib':

        plot = plt.imshow(cell_ids, extent=(plot_left, plot_right, plot_bottom, plot_top))
        plt.show()
        return plot
#     elif backend=='plotly':

#             figure = go.Figure(
#                 data=go.Heatmap(
#                     z=cell_ids,
#                     colorscale='viridis',
#                     x0 =left,
#                     dx=abs(left-right)/(len(cell_ids[0])-1),
#                     y0 =bottom,
#                     dy=abs(bottom-top)/(len(cell_ids)-1),
#                     # colorbar=dict(title=dict(side="right", text=cbar_label)),
#                     ),
#                 )
            

#             # figure.update_layout(
#             #     xaxis={"title": x_label},
#             #     yaxis={"title": y_label},
#             #     autosize=False,
#             #     height=800,
#             # )
#             # figure.update_yaxes(
#             #     scaleanchor = "x",
#             #     scaleratio = 1,
#             # )

#             figure.write_html('openmc_plot_regularmesh_image.html')

        


# s1 = openmc.Sphere(r=10)
# s2 = openmc.Sphere(r=20)
# s3 = openmc.XPlane(x0=1)

# r1 = -s1 & -s3
# r2 = -s2 & +s1 & -s3

# c1 = openmc.Cell(region=r1)
# c2 = openmc.Cell(region=r2)

# u1 = openmc.Universe(cells=[c1, c2])

# # test_no_corners_xz_axis
# plot_axis_slice(
#     geometry=u1,
#     axis='yz'
# )

# # test_one_corners_xz_axis
# plot_axis_slice(
#     geometry=u1,
#     upper_right=(-10,10),
#     axis='yz'
# )

# # test_two_corners_xz_axis
# plot_axis_slice(
#     geometry=u1,
#     upper_right=(-5,5),
#     lower_left=(-10, 0),
#     axis='yz'
# )
