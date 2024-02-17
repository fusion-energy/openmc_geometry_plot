import xml.etree.ElementTree as ET
from pathlib import Path
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import openmc
import streamlit as st
from matplotlib import colors
from pylab import cm, colormaps
import numpy as np

import openmc_geometry_plot  # adds extra functions to openmc.Geometry


def save_uploadedfile(uploadedfile):
    with open(uploadedfile.name, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success(f"Saved File to {uploadedfile.name}")


def header():
    """This section writes out the page header common to all tabs"""

    st.set_page_config(
        page_title="OpenMC Geometry Plot",
        page_icon="⚛",
        layout="wide",
    )

    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {
                    visibility: hidden;
                    }
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.write(
        """
            # OpenMC geometry plot

            ### ⚛ A geometry plotting user interface for OpenMC.

            🐍 Run this app locally with Python ```pip install openmc_geometry_plot``` then run with ```openmc_geometry_plot```

            ⚙ Produce MatPlotLib or Plotly plots in batch with the 🐍 [Python API](https://github.com/fusion-energy/openmc_geometry_plot/tree/master/examples)

            💾 Raise a feature request, report and issue or make a contribution on [GitHub](https://github.com/fusion-energy/openmc_geometry_plot)

            📧 Email feedback to mail@jshimwell.com

            🔗 This package forms part of a more [comprehensive openmc plotting](https://github.com/fusion-energy/openmc_plot) package where geometry, tallies, slices, etc can be plotted and is hosted on [xsplot.com](https://www.xsplot.com/) .
        """
    )
    st.write("<br>", unsafe_allow_html=True)


def main():

    file_label_col1, file_label_col2 = st.columns([1, 1])
    file_label_col1.write(
        """
            👉 Create your ```openmc.Geometry()``` and export the geometry xml file using ```export_to_xml()```.

            Not got a geometry.xml file handy, right mouse 🖱️ click and save these links 
            [ example 1 ](https://fusion-energy.github.io/openmc_geometry_plot/examples/csg_tokamak/geometry.xml),
            [ example 2 ](https://fusion-energy.github.io/openmc_geometry_plot/examples/csg_cylinder_box/geometry.xml)

        """
    )
    file_label_col2.write(
        """
            👉 Create your DAGMC h5m file using tools like [CAD-to-h5m](https://github.com/fusion-energy/cad_to_dagmc), [STL-to_h5m](https://github.com/fusion-energy/stl_to_h5m) [vertices-to-h5m](https://github.com/fusion-energy/vertices_to_h5m), [Brep-to-h5m](https://github.com/fusion-energy/brep_to_h5m) or the [Cubit](https://coreform.com/products/coreform-cubit/) [Plugin](https://github.com/svalinn/Cubit-plugin)
            
            Not got a DAGMC h5m file handy, right mouse 🖱️ click and save these links 
            [ example 1 ](https://fusion-energy.github.io/openmc_geometry_plot/examples/dagmc_tokamak/dagmc_180_tokamak.h5m)

        """
    )

    file_col1, file_col2 = st.columns([1, 1])
    geometry_xml_file = file_col1.file_uploader(
        "Upload your geometry.xml", type=["xml"]
    )
    dagmc_file = file_col2.file_uploader("Upload your DAGMC h5m", type=["h5m"])

    my_geometry = None

    if dagmc_file is None and geometry_xml_file is None:
        new_title = '<center><p style="font-family:sans-serif; color:Red; font-size: 30px;">Upload your geometry.xml or DAGMC h5m file</p></center>'
        st.markdown(new_title, unsafe_allow_html=True)

        sub_title = '<center><p> Not got geometry files handy? Download an example <a href="https://raw.githubusercontent.com/fusion-energy/openmc_plot/main/examples/tokamak/geometry.xml" download>geometry.xml</a> or DAGMC h5m file</p></center>'
        st.markdown(sub_title, unsafe_allow_html=True)

    # DAGMC route
    elif dagmc_file is not None and geometry_xml_file is not None:
        save_uploadedfile(dagmc_file)
        save_uploadedfile(geometry_xml_file)

        bound_dag_univ = openmc.DAGMCUniverse(
            filename=dagmc_file.name
        ).bounded_universe()
        my_geometry = openmc.Geometry(root=bound_dag_univ)

        dag_universe = my_geometry.get_dagmc_universe()

        mat_ids = range(0, len(dag_universe.material_names) + 1)
        # mat_names = dag_universe.material_names

        if len(mat_ids) >= 1:
            set_mat_ids = set(mat_ids)
        else:
            set_mat_ids = ()

        # set_cell_ids = set(di.get_volumes_from_h5m(dagmc_file.name))
        set_cell_ids = list(range(1, dag_universe.n_cells + 1))
        set_mat_names = set(dag_universe.material_names)
        all_cell_names = set_cell_ids
        set_cell_names = set(all_cell_names)

    elif dagmc_file is not None and geometry_xml_file is None:
        save_uploadedfile(dagmc_file)

        # make a basic openmc geometry
        bound_dag_univ = openmc.DAGMCUniverse(
            filename=dagmc_file.name
        ).bounded_universe()
        my_geometry = openmc.Geometry(root=bound_dag_univ)

        dag_universe = my_geometry.get_dagmc_universe()

        # find all material names
        mat_ids = range(0, len(dag_universe.material_names) + 1)

        if len(mat_ids) >= 1:
            set_mat_ids = set(mat_ids)
        else:
            set_mat_ids = ()

        # set_cell_ids = set(di.get_volumes_from_h5m(dagmc_file.name))
        set_cell_ids = list(range(1, dag_universe.n_cells + 1))
        set_mat_names = set(dag_universe.material_names)
        all_cell_names = set_cell_ids
        set_cell_names = set(all_cell_names)

    # CSG route
    elif dagmc_file is None and geometry_xml_file is not None:

        save_uploadedfile(geometry_xml_file)

        tree = ET.parse(geometry_xml_file.name)

        root = tree.getroot()
        all_cells = root.findall("cell")
        mat_ids = []

        for cell in all_cells:
            if "material" in cell.keys():
                if cell.get("material") == "void":
                    mat_ids.append(0)
                    print(f"material for cell {cell} is void")
                else:
                    mat_ids.append(int(cell.get("material")))

        if len(mat_ids) >= 1:
            set_mat_ids = set(mat_ids)
        else:
            set_mat_ids = ()

        set_mat_names = set_mat_ids  # can't find material names in CSG with just the geometry xml as we don't have material names

        my_mats = openmc.Materials()
        for mat_id in set_mat_ids:
            new_mat = openmc.Material()
            new_mat.id = mat_id
            new_mat.add_nuclide("He4", 1)
            # adds a single nuclide that is in minimal cross section xml to avoid material failing
            my_mats.append(new_mat)

        my_geometry = openmc.Geometry.from_xml(
            path=geometry_xml_file.name, materials=my_mats
        )
        all_cells = my_geometry.get_all_cells()

    if my_geometry:
        print("geometry is set to something so attempting to plot")
        bb = my_geometry.bounding_box

        basis = st.sidebar.selectbox(
            label="basis",
            options=("xy", "xz", "yz"),
            index=0,
            key="basis",
            help="Setting the direction of view automatically sets the horizontal and vertical axis used for the plot.",
        )
        backend = st.sidebar.selectbox(
            label="Ploting backend",
            options=("matplotlib", "plotly"),
            index=0,
            key="geometry_ploting_backend",
            help="Create png images with MatPlotLib or HTML plots with Plotly",
        )
        outline = st.sidebar.selectbox(
            label="Outline",
            options=("materials", "cells", None),
            index=0,
            key="outline",
            help="Allows an outline to be drawn around the cells or materials, select None for no outline",
        )
        legend = st.sidebar.selectbox(
            label="Outline",
            options=(True, False),
            index=0,
            key="legend",
            help="Allows a plot legend to be added",
        )
        axis_units = st.sidebar.selectbox(
            label="Axis units",
            options=('km', 'm', 'cm', 'mm'),
            index=3,
            key="axis_units",
            help="Select the units used on the X and Y axis",
        )
        pixels = st.sidebar.number_input(
            label="Number of pixels",
            value=40000,
            help="Increasing this value increases the image resolution but also requires longer to create the image",
        )

        plot_left, plot_right = None, None
        plot_bottom, plot_top = None, None
        x_min, x_max = None, None
        y_min, y_max = None, None

        x_index = {'x':0,'y':1,'z':2}[basis[0]]
        y_index = {'x':0,'y':1,'z':2}[basis[1]]
        slice_index = {"xy": 2, "xz": 1, "yz": 0}[basis]
        slice_axis = {"xy": 'Z', "xz": 'Y', "yz": 'X'}[basis]

        if np.isinf(bb[0][x_index]) or np.isinf(bb[1][x_index]):
            x_min = st.sidebar.number_input(
                label="minimum vertical axis value", key="x_min"
            )
            x_max = st.sidebar.number_input(
                label="maximum vertical axis value", key="x_max"
            )
        else:
            x_min = float(bb[0][x_index])
            x_max = float(bb[1][x_index])

        # y axis is y values
        if np.isinf(bb[0][y_index]) or np.isinf(bb[1][y_index]):
            y_min = st.sidebar.number_input(
                label="minimum vertical axis value", key="y_min"
            )
            y_max = st.sidebar.number_input(
                label="maximum vertical axis value", key="y_max"
            )
        else:
            y_min = float(bb[0][y_index])
            y_max = float(bb[1][y_index])

        # slice axis is z
        if np.isinf(bb[0][slice_index]) or np.isinf(bb[1][slice_index]):
            slice_min = st.sidebar.number_input(
                label="minimum slice value", key="slice_min"
            )
            slice_max = st.sidebar.number_input(
                label="maximum slice value", key="slice_max"
            )
        else:
            slice_min = float(bb[0][slice_index])
            slice_max = float(bb[1][slice_index])

        if isinstance(x_min, float) and isinstance(x_max, float):
            plot_right, plot_left = st.sidebar.slider(
                label="Left and right values for the horizontal axis",
                min_value=x_min,
                max_value=x_max,
                value=(x_min, x_max),
                key="left_right_slider",
                help="Set the lowest visible value and highest visible value on the horizontal axis",
            )

        if isinstance(y_min, float) and isinstance(y_max, float):
            plot_bottom, plot_top = st.sidebar.slider(
                label="Bottom and top values for the vertical axis",
                min_value=y_min,
                max_value=y_max,
                value=(y_min, y_max),
                key="bottom_top_slider",
                help="Set the lowest visible value and highest visible value on the vertical axis",
            )
        if isinstance(slice_min, float) and isinstance(slice_max, float):
            slice_value = st.sidebar.slider(
                label="Slice value",
                min_value=slice_min,
                max_value=slice_max,
                value=(slice_min + slice_max) / 2,
                key="slice_slider",
                help="Set the value of the slice axis",
            )

        color_by = st.sidebar.selectbox(
            label="Color by",
            options=("materials", "cells"),
            index=0,
            key="color_by",
            help="Should the plot be colored by material or by cell",
        )

        selected_color_map = st.sidebar.selectbox(
            label="Color map", options=colormaps(), index=82
        )  # index 82 is tab20c

        if color_by == "materials":
            cmap = cm.get_cmap(selected_color_map, len(set_mat_ids))
            initial_hex_color = []
            for i in range(cmap.N):
                rgba = cmap(i)
                # rgb2hex accepts rgb or rgba
                initial_hex_color.append(colors.rgb2hex(rgba))

            for c, id in enumerate(set_mat_ids):
                st.sidebar.color_picker(
                    f"Color of material with id {id}",
                    key=f"mat_{id}",
                    value=initial_hex_color[c],
                )

            my_colors = {}
            for id in set_mat_ids:
                hex_color = st.session_state[f"mat_{id}"].lstrip("#")
                RGB = tuple(int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))
                my_colors[id] = RGB

        elif color_by == "cells":
            cmap = cm.get_cmap(selected_color_map, len(all_cells))
            initial_hex_color = []
            for i in range(cmap.N):
                rgba = cmap(i)
                # rgb2hex accepts rgb or rgba
                initial_hex_color.append(colors.rgb2hex(rgba))

            for c, cell in enumerate(all_cells.values()):
                if cell.name in ["", None]:
                    desc = f"Color of cell id {cell.id}"
                else:
                    desc = f"Color of cell id {cell.id}, cell name {cell.name}"

                st.sidebar.color_picker(
                    desc,
                    key=f"cell_{cell.id}",
                    value=initial_hex_color[c],
                )

            my_colors = {}  # adding entry for void cells
            for id, value in all_cells.items():
                hex_color = st.session_state[f"cell_{id}"].lstrip("#")
                RGB = tuple(int(hex_color[i : i + 2], 16)  for i in (0, 2, 4))
                print('RGB',RGB)
                my_colors[value] =  RGB

        print(my_colors)
        title = st.sidebar.text_input(
            "Plot title",
            help="Optionally set your own title for the plot",
            value=f"Slice through OpenMC geometry at {slice_axis}={slice_value}",
        )

        if (
            isinstance(plot_left, float)
            and isinstance(plot_right, float)
            and isinstance(plot_top, float)
            and isinstance(plot_bottom, float)
        ):

            if backend == "matplotlib":

                origin=(
                    (plot_left+plot_right)/2,
                    (plot_top+plot_bottom)/2,
                    (slice_value)
                )
                width_x=plot_left-plot_right
                width_y=plot_top-plot_bottom
                print('origin',origin)

                plot = my_geometry.plot(
                    origin=origin,
                    width=[width_x,width_y],
                    pixels=pixels,
                    basis='xy',
                    color_by='cell',
                    colors=my_colors,
                    legend=legend,
                    axis_units=axis_units,
                    outline=outline,
                )

                plt.title(title)
                plt.savefig("openmc_plot_geometry_image.png")
                st.pyplot(plt)
                # st.image("openmc_plot_geometry_image.png", use_column_width="always")

                with open("openmc_plot_geometry_image.png", "rb") as file:
                    st.sidebar.download_button(
                        label="Download image",
                        data=file,
                        file_name="openmc_plot_geometry_image.png",
                        mime="image/png",
                    )
            else:
                data = [
                    go.Heatmap(
                        z=color_data_slice,
                        showscale=False,
                        colorscale="viridis",
                        x0=plot_left,
                        dx=abs(plot_left - plot_right) / (len(color_data_slice[0]) - 1),
                        y0=plot_bottom,
                        dy=abs(plot_bottom - plot_top) / (len(color_data_slice) - 1),
                        # colorbar=dict(title=dict(side="right", text=cbar_label)),
                        # text = material_ids,
                        # hovertemplate=
                        # # 'material ID = %{z}<br>'+
                        # "Cell ID = %{z}<br>" +
                        # # '<br>%{text}<br>'+
                        # xlabel[:2].title()
                        # + ": %{x} cm<br>"
                        # + ylabel[:2].title()
                        # + ": %{y} cm<br>",
                    )
                ]

                if outline is not None:
                    data.append(
                        go.Contour(
                            z=outline_data_slice,
                            x0=plot_left,
                            dx=abs(plot_left - plot_right)
                            / (len(outline_data_slice[0]) - 1),
                            y0=plot_bottom,
                            dy=abs(plot_bottom - plot_top)
                            / (len(outline_data_slice) - 1),
                            contours_coloring="lines",
                            line_width=1,
                            colorscale=[[0, "rgb(0, 0, 0)"], [1.0, "rgb(0, 0, 0)"]],
                            showscale=False,
                        )
                    )

                plot = go.Figure(data=data)

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

                plot.write_html("openmc_plot_geometry_image.html")

                with open("openmc_plot_geometry_image.html", "rb") as file:
                    st.sidebar.download_button(
                        label="Download image",
                        data=file,
                        file_name="openmc_plot_geometry_image.html",
                        mime=None,
                    )
                st.plotly_chart(plot, use_container_width=True)

            st.write("Model info")
            st.write(f"origin {origin}")
            st.write(f"Material IDS found {set_mat_ids}")
            st.write(f"Material names found {set_mat_names}")
            st.write(f"Cell IDS found {all_cells.keys()}")
            # st.write(f"Cell names found {all_cells.}")
            st.write(f"Bounding box lower left x={bb[0][0]} y={bb[0][1]} z={bb[0][2]}")
            st.write(f"Bounding box upper right x={bb[1][0]} y={bb[1][1]} z={bb[1][2]}")


if __name__ == "__main__":
    header()
    main()
