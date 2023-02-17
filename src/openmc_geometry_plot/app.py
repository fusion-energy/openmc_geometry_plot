import xml.etree.ElementTree as ET
from pathlib import Path
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import openmc
import streamlit as st
from pylab import *


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
    header()

    file_label_col1, file_label_col2 = st.columns([1, 1])
    file_label_col1.write(
        """
            👉 Create your ```openmc.Geometry()``` and export the geometry xml file using ```export_to_xml()```.
        """
    )
    file_label_col2.write(
        """
            👉 Create your DAGMC h5m file using tools like [CAD-to-h5m](https://github.com/fusion-energy/cad_to_dagmc), [STL-to_h5m](https://github.com/fusion-energy/stl_to_h5m) [vertices-to-h5m](https://github.com/fusion-energy/vertices_to_h5m), [Brep-to-h5m](https://github.com/fusion-energy/brep_to_h5m) or the [Cubit](https://coreform.com/products/coreform-cubit/) [Plugin](https://github.com/svalinn/Cubit-plugin)
        """
    )
    file_col1, file_col2 = st.columns([1, 1])
    geometry_xml_file = file_col1.file_uploader(
        "Upload your geometry.xml", type=["xml"]
    )
    dagmc_file = file_col2.file_uploader("Upload your DAGMC h5m", type=["h5m"])

    my_geometry = None

    if dagmc_file == None and geometry_xml_file == None:
        new_title = '<center><p style="font-family:sans-serif; color:Red; font-size: 30px;">Upload your geometry.xml or DAGMC h5m file</p></center>'
        st.markdown(new_title, unsafe_allow_html=True)

        sub_title = '<center><p> Not got geometry files handy? Download an example <a href="https://raw.githubusercontent.com/fusion-energy/openmc_plot/main/examples/tokamak/geometry.xml" download>geometry.xml</a> or DAGMC h5m file</p></center>'
        st.markdown(sub_title, unsafe_allow_html=True)

    # DAGMC route
    elif dagmc_file != None and geometry_xml_file != None:
        st.markdown("DAGMC geometry not yet supported, work in progress")
        save_uploadedfile(dagmc_file)
        save_uploadedfile(geometry_xml_file)

        bound_dag_univ = openmc.DAGMCUniverse(filename=dagmc_file.name).bounded_universe()
        my_geometry = openmc.Geometry(root=bound_dag_univ)
        
        mat_ids = range(0, len(my_geometry.material_names)+1)

    elif dagmc_file != None and geometry_xml_file == None:
        st.markdown("DAGMC geometry not yet supported, work in progress")
        save_uploadedfile(dagmc_file)

        # make a basic openmc geometry
        bound_dag_univ = openmc.DAGMCUniverse(filename=dagmc_file.name).bounded_universe()
        my_geometry = openmc.Geometry(root=bound_dag_univ)

        dag_universe = my_geometry.get_dagmc_universe()

        # find all material names
        mat_ids = range(0, len(dag_universe.material_names)+1)
        

        # make a pretend material for each one

    # CSG route
    elif dagmc_file == None and geometry_xml_file != None:
        save_uploadedfile(geometry_xml_file)

        tree = ET.parse(geometry_xml_file.name)

        root = tree.getroot()
        all_cells = root.findall("cell")
        mat_ids = []
        for cell in all_cells:
            if "material" in cell.keys():
                if cell.get("material") == "void":
                    print(f"material for cell {cell} is void")
                else:
                    mat_ids.append(int(cell.get("material")))

        if len(mat_ids) >= 1:
            set_mat_ids = set(mat_ids)
        else:
            set_mat_ids = ()

        my_mats = openmc.Materials()
        for mat_id in set_mat_ids:
            new_mat = openmc.Material()
            new_mat.id = mat_id
            new_mat.add_nuclide("Li6", 1)
            # adds a single nuclide that is in minimal cross section xml to avoid material failing
            my_mats.append(new_mat)

        my_geometry = openmc.Geometry.from_xml(
            path=geometry_xml_file.name, materials=my_mats
        )

    if my_geometry:
        print('geometry is something so plotting')
        bb = my_geometry.bounding_box
        print('bb', bb)

        col1, col2 = st.columns([1, 3])

        view_direction = col1.selectbox(
            label="View direction",
            options=("z", "x", "y"),
            index=0,
            key="geometry_view_direction",
            help="Setting the direction of view automatically sets the horizontal and vertical axis used for the plot.",
        )
        backend = col1.selectbox(
            label="Ploting backend",
            options=("matplotlib", "plotly"),
            index=0,
            key="geometry_ploting_backend",
            help="Create png images with MatPlotLib or HTML plots with Plotly",
        )
        outline = col1.selectbox(
            label="Outline",
            options=("cells", "materials", None),
            index=0,
            key="outline",
            help="Allows an outline to be drawn around the cells or materials, select None for no outline",
        )
        color_by = col1.selectbox(
            label="Color by",
            options=("cells", "materials"),
            index=0,
            key="color_by",
            help="Should the plot be colored by material or by cell",
        )
        plot_left, plot_right = None, None
        plot_bottom, plot_top = None, None
        x_min, x_max = None, None
        y_min, y_max = None, None

        if view_direction in ["z"]:
            # x axis is x values
            if np.isinf(bb[0][0]) or np.isinf(bb[1][0]):
                x_min = col1.number_input(
                    label="minimum vertical axis value", key="x_min"
                )
                x_max = col1.number_input(
                    label="maximum vertical axis value", key="x_max"
                )
            else:
                x_min = float(bb[0][0])
                x_max = float(bb[1][0])
            print('x_min', x_min)
            print('x_max', x_max)

            # y axis is y values
            if np.isinf(bb[0][1]) or np.isinf(bb[1][1]):
                y_min = col1.number_input(
                    label="minimum vertical axis value", key="y_min"
                )
                y_max = col1.number_input(
                    label="maximum vertical axis value", key="y_max"
                )
            else:
                y_min = float(bb[0][1])
                y_max = float(bb[1][1])
            print('y_min', y_min)
            print('y_max', y_max)

        if view_direction in ["y"]:
            # x axis is x values
            if np.isinf(bb[0][0]) or np.isinf(bb[1][0]):
                x_min = col1.number_input(
                    label="minimum horizontal axis value", key="x_min"
                )
                x_max = col1.number_input(
                    label="maximum horizontal axis value", key="x_max"
                )
            else:
                x_min = float(bb[0][0])
                x_max = float(bb[1][0])

            # y axis is z values
            if np.isinf(bb[0][2]) or np.isinf(bb[1][2]):
                y_min = col1.number_input(
                    label="minimum vertical axis value", key="y_min"
                )
                y_max = col1.number_input(
                    label="maximum vertical axis value", key="y_max"
                )
            else:
                y_min = float(bb[0][2])
                y_max = float(bb[1][2])

        if view_direction in ["x"]:
            # x axis is y values
            if np.isinf(bb[0][1]) or np.isinf(bb[1][1]):
                x_min = col1.number_input(label="minimum vertical axis value")
                x_max = col1.number_input(label="maximum vertical axis value")
            else:
                x_min = float(bb[0][1])
                x_max = float(bb[1][1])

            # y axis is z values
            if np.isinf(bb[0][2]) or np.isinf(bb[1][2]):
                y_min = col1.number_input(
                    label="minimum vertical axis value", key="y_min"
                )
                y_max = col1.number_input(
                    label="maximum vertical axis value", key="y_max"
                )
            else:
                y_min = float(bb[0][1])
                y_max = float(bb[1][1])

        if isinstance(x_min, float) and isinstance(x_max, float):
            print('slider')
            plot_left, plot_right = col1.slider(
                label="Left and right values for the horizontal axis",
                min_value=x_min,
                max_value=x_max,
                value=(x_min, x_max),
                key="left_right_slider",
                help="Set the lowest visible value and highest visible value on the horizontal axis",
            )

        if isinstance(y_min, float) and isinstance(y_max, float):
            plot_bottom, plot_top = col1.slider(
                label="Bottom and top values for the vertical axis",
                min_value=y_min,
                max_value=y_max,
                value=(y_min, y_max),
                key="bottom_top_slider",
                help="Set the lowest visible value and highest visible value on the vertical axis",
            )

        pixels_across = col1.number_input(
            label="Number of horizontal pixels",
            value=500,
            help="Increasing this value increases the image resolution but also requires longer to create the image",
        )

        title = col1.text_input(
            "Plot title",
            help="Optionally set your own title for the plot",
            value=f"Slice through OpenMC geometry with view direction {view_direction}",
        )
        print(plot_left, plot_right, plot_top, plot_bottom)
        if isinstance(plot_left, float) and isinstance(plot_right, float) and isinstance(plot_top, float) and isinstance(plot_bottom, float):
            if color_by == "cells":
                print('getting cell id slice')
                data_slice = my_geometry.get_slice_of_cell_ids(
                    view_direction=view_direction,
                    plot_left=plot_left,
                    plot_right=plot_right,
                    plot_top=plot_top,
                    plot_bottom=plot_bottom,
                    pixels_across=pixels_across,
                )
            elif color_by == "materials":
                data_slice = my_geometry.get_slice_of_material_ids(
                    view_direction=view_direction,
                    plot_left=plot_left,
                    plot_right=plot_right,
                    plot_top=plot_top,
                    plot_bottom=plot_bottom,
                    pixels_across=pixels_across,
                )

            (xlabel, ylabel) = my_geometry.get_axis_labels(
                view_direction=view_direction
            )
            if backend == "matplotlib":
                plt.imshow(
                    data_slice,
                    extent=my_geometry.get_mpl_plot_extent(
                        view_direction=view_direction
                    ),
                    interpolation="none",
                )

                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(title)

                if outline is not None:
                    # gets unique levels for outlines contour plot
                    levels = np.unique(
                        [item for sublist in data_slice for item in sublist]
                    )
                    plt.contour(
                        data_slice,
                        origin="upper",
                        colors="k",
                        linestyles="solid",
                        levels=levels,
                        linewidths=0.5,
                        extent=my_geometry.get_mpl_plot_extent(
                            view_direction=view_direction
                        ),
                    )

                plt.savefig("openmc_plot_geometry_image.png")
                col2.pyplot(plt)
                # col2.image("openmc_plot_geometry_image.png", use_column_width="always")

                with open("openmc_plot_geometry_image.png", "rb") as file:
                    col1.download_button(
                        label="Download image",
                        data=file,
                        file_name="openmc_plot_geometry_image.png",
                        mime="image/png",
                    )
            else:
                plot = go.Figure(
                    data=[
                        go.Heatmap(
                            z=data_slice,
                            showscale=False,
                            colorscale="viridis",
                            x0=plot_left,
                            dx=abs(plot_left - plot_right) / (len(data_slice[0]) - 1),
                            y0=plot_bottom,
                            dy=abs(plot_bottom - plot_top) / (len(data_slice) - 1),
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
                        ),
                        # outline
                        go.Contour(
                            z=data_slice,
                            x0=plot_left,
                            dx=abs(plot_left - plot_right) / (len(data_slice[0]) - 1),
                            y0=plot_bottom,
                            dy=abs(plot_bottom - plot_top) / (len(data_slice) - 1),
                            contours_coloring="lines",
                            line_width=1,
                            colorscale=[[0, "rgb(0, 0, 0)"], [1.0, "rgb(0, 0, 0)"]],
                            showscale=False,
                        ),
                    ]
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

                plot.write_html("openmc_plot_geometry_image.html")

                with open("openmc_plot_geometry_image.html", "rb") as file:
                    col1.download_button(
                        label="Download image",
                        data=file,
                        file_name="openmc_plot_geometry_image.html",
                        mime=None,
                    )
                col2.plotly_chart(plot, use_container_width=True)


if __name__ == "__main__":
    main()
