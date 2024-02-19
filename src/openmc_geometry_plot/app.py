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

    st.write("<br>", unsafe_allow_html=True)


def main():


    file_col1, file_col2 = st.sidebar.columns([1, 1])
    geometry_xml_file = file_col1.file_uploader(
        "Upload your geometry.xml", type=["xml"]
    )
    dagmc_file = file_col2.file_uploader("Optionally upload your DAGMC h5m", type=["h5m"])
 
    my_geometry = None

    if dagmc_file is None and geometry_xml_file is None:
        title_1 = '<center><p style="font-family:sans-serif; font-size: 30px;">Upload your geometry.xml file</p></center>'
        st.markdown(title_1, unsafe_allow_html=True)

        title_2 = '<center><p style="font-family:sans-serif; font-size: 30px;">Create an openmc.Geometry() and export the geometry xml file using export_to_xml().</p></center>'
        st.markdown(title_2, unsafe_allow_html=True)

        title_3 = '<center><p> Not got geometry files handy? Right mouse 🖱️ click and save this link <a href="https://raw.githubusercontent.com/fusion-energy/openmc_geometry_plot/31be0556f3f34c102cab3de094df08f48acad5ca/examples/csg_tokamak/geometry.xml" download>geometry.xml</a></p></center>'
        st.markdown(title_3, unsafe_allow_html=True)

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
        all_cells_ids = range(0, dag_universe.n_cells + 1)
        all_cells ={}
        for cell_id in all_cells_ids:
            all_cells[cell_id] =openmc.Cell(cell_id=cell_id)

        if len(mat_ids) >= 1:
            set_mat_ids = set(mat_ids)
        else:
            set_mat_ids = ()

        set_mat_names = set(dag_universe.material_names)

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

        all_cells_ids = range(0, dag_universe.n_cells + 1)
        all_cells ={}
        for cell_id in all_cells_ids:
            all_cells[cell_id] =openmc.Cell(cell_id=cell_id)

        if len(mat_ids) >= 1:
            set_mat_ids = set(mat_ids)
        else:
            set_mat_ids = ()

        set_mat_names = set(dag_universe.material_names)

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
                    pass
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
            path=geometry_xml_file.name,
            materials=my_mats
        )
        all_cells = my_geometry.get_all_cells()

    if my_geometry:
        print("geometry is set to something so attempting to plot")
        bb = my_geometry.bounding_box
        print(f'bounding box {bb}')

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
            key="geometry_plotting_backend",
            help="Create png images with MatPlotLib or HTML plots with Plotly",
        )
        legend = st.sidebar.selectbox(
            label="Legend",
            options=(True, False),
            index=0,
            key="legend",
            help="Allows a plot legend to be added",
        )
        axis_units = st.sidebar.selectbox(
            label="Axis units",
            options=('km', 'm', 'cm', 'mm'),
            index=2,
            key="axis_units",
            help="Select the units used on the X and Y axis",
        )
        pixels = st.sidebar.number_input(
            label="Number of pixels",
            value=200000,
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

        # x axis values
        if np.isinf(bb[0][x_index]) or np.isinf(bb[1][x_index]):
            plot_left = st.sidebar.number_input(
                value = -2000., label="minimum vertical axis value", key="x_min"
            )
            plot_right = st.sidebar.number_input(
                value=2000., label="maximum vertical axis value", key="x_max"
            )
        else:
            x_min = float(bb[0][x_index])
            x_max = float(bb[1][x_index])
            plot_right, plot_left = st.sidebar.slider(
                label="Left and right values for the horizontal axis",
                min_value=x_min,
                max_value=x_max,
                value=(x_min, x_max),
                key="left_right_slider",
                help="Set the lowest visible value and highest visible value on the horizontal axis",
            )


        # y axis values
        if np.isinf(bb[0][y_index]) or np.isinf(bb[1][y_index]):
            plot_bottom = st.sidebar.number_input(
                value=-2000., label="minimum vertical axis value", key="y_min"
            )
            plot_top = st.sidebar.number_input(
                value=2000., label="maximum vertical axis value", key="y_max"
            )
        else:
            y_min = float(bb[0][y_index])
            y_max = float(bb[1][y_index])
            plot_bottom, plot_top = st.sidebar.slider(
                label="Bottom and top values for the vertical axis",
                min_value=y_min,
                max_value=y_max,
                value=(y_min, y_max),
                key="bottom_top_slider",
                help="Set the lowest visible value and highest visible value on the vertical axis",
            )

        # slice axis is z
        if np.isinf(bb[0][slice_index]) or np.isinf(bb[1][slice_index]):
            slice_value = st.sidebar.number_input(
                value=0, label=f"Slice value on slice axis", key="slice_slider"
            )
        else:
            slice_min = float(bb[0][slice_index])
            slice_max = float(bb[1][slice_index])
            slice_value = st.sidebar.slider(
                label=f"Slice value on slice axis",
                min_value=slice_min,
                max_value=slice_max,
                value=(slice_min + slice_max) / 2,
                key="slice_slider",
                help="Set the value of the slice axis",
            )

        color_by = st.sidebar.selectbox(
            label="Color by",
            options=("cell", "material"),
            index=0,
            key="color_by",
            help="Should the plot be colored by material or by cell",
        )
        outline = st.sidebar.selectbox(
            label="Outline",
            options=(True, False),
            index=0,
            key="outline",
            help="Allows an outline to be drawn around the cells or materials, select None for no outline",
        )
        selected_color_map = st.sidebar.selectbox(
            label="Color map", options=colormaps(), index=82
        )  # index 82 is tab20c

        if color_by == "material":
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
                RGB = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
                mat = my_geometry.get_all_materials()[id]
                my_colors[mat] = RGB

        elif color_by == "cell":
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
                my_colors[value] =  RGB

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

            if basis == 'xy':
                origin=(
                    (plot_left+plot_right)/2,
                    (plot_top+plot_bottom)/2,
                    slice_value,
                )
            elif basis == 'yz':
                origin=(
                    slice_value,
                    (plot_left+plot_right)/2,
                    (plot_top+plot_bottom)/2,
                )
            elif basis == 'xz':
                origin=(
                    (plot_left+plot_right)/2,
                    slice_value,
                    (plot_top+plot_bottom)/2,
                )

            if backend == "matplotlib":
                print('plotting with matplotlib')

                width_x=plot_left-plot_right
                width_y=plot_top-plot_bottom

                plot = my_geometry.plot(
                    origin=origin,
                    width=[width_x,width_y],
                    pixels=pixels,
                    basis=basis,
                    color_by=color_by,
                    colors=my_colors,
                    legend=legend,
                    axis_units=axis_units,
                    outline=outline,
                )

                plt.title(title)
                plt.savefig("openmc_plot_geometry_image.png")
                st.pyplot(plt)

                with open("openmc_plot_geometry_image.png", "rb") as file:
                    st.sidebar.download_button(
                        label="Download image",
                        data=file,
                        file_name="openmc_plot_geometry_image.png",
                        mime="image/png",
                    )
            else:
                from openmc_geometry_plot import plot_plotly
                print('plotting with plotly')
                plot = plot_plotly(
                    my_geometry,
                    origin=origin,
                    # width=[width_x,width_y],
                    pixels=pixels,
                    basis=basis,
                    color_by=color_by,
                    colors=my_colors,
                    legend=legend,
                    axis_units=axis_units,
                    outline=outline,      
                    title=title         
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
        else:
            print(plot_left,plot_right, plot_top, plot_bottom)


if __name__ == "__main__":
    header()
    main()
