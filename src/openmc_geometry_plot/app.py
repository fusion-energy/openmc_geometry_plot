import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import openmc
import streamlit as st
from pylab import *


from openmc_geometry_plot import plot_axis_slice


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

    st.write(
        """
            👉 Create your ```openmc.Geometry()``` and export the geometry xml file using ```export_to_xml()```.
        """
    )
    geometry_xml_file = st.file_uploader("Upload your geometry.xml", type=["xml"])

    if geometry_xml_file == None:
        new_title = '<p style="font-family:sans-serif; color:Red; font-size: 30px;">Upload your geometry.xml</p>'
        st.markdown(new_title, unsafe_allow_html=True)

        st.markdown(
            'Not got xml files handy? Download sample [geometry.xml](https://raw.githubusercontent.com/fusion-energy/openmc_plot/main/examples/tokamak/geometry.xml "download")'
        )

    else:

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

        my_mats = []
        for mat_id in set_mat_ids:
            new_mat = openmc.Material()
            new_mat.id = mat_id
            new_mat.add_nuclide("Li6", 1)
            # adds a single nuclide that is in minimal cross section xml to avoid material failing
            my_mats.append(new_mat)

        my_geometry = openmc.Geometry.from_xml(
            path=geometry_xml_file.name, materials=my_mats
        )

        my_universe = my_geometry.root_universe

        bb = my_universe.bounding_box

        col1, col2 = st.columns([1, 3])

        view_direction = col1.selectbox(
            label="View direction",
            options=("z", "x", "y"),
            index=0,
            key='geometry_view_direction',
            help='Setting the direction of view automatically sets the horizontal and vertical axis used for the plot.'
        )
        backend = col1.selectbox(
            label="Ploting backend",
            options=("matplotlib", "plotly"),
            index=0,
            key='geometry_ploting_backend',
            help='Create png images with MatPlotLib or HTML plots with Plotly'
        )
        plot_left, plot_right = None, None
        plot_bottom, plot_top = None, None
        x_min, x_max = None, None
        y_min, y_max = None, None

        if view_direction in ['z']:

            # x axis is x values
            if np.isinf(bb[0][0]) or np.isinf(bb[1][0]):
                x_min = col1.number_input(label='minimum vertical axis value', key='x_min')
                x_max = col1.number_input(label='maximum vertical axis value', key='x_max')
            else:
                x_min = float(bb[0][0])
                x_max = float(bb[1][0])
            # if :
            # else:

            # y axis is y values
            if np.isinf(bb[0][1]) or np.isinf(bb[1][1]):
                y_min = col1.number_input(label='minimum vertical axis value', key='y_min')
                y_max = col1.number_input(label='maximum vertical axis value', key='y_max')
            else:
                y_min = float(bb[0][1])
                y_max = float(bb[1][1])
            # if :
            # else:

        if view_direction in ['y']:

            # x axis is x values
            if np.isinf(bb[0][0]) or np.isinf(bb[1][0]):
                x_min = col1.number_input(label='minimum horizontal axis value', key='x_min')
                x_max = col1.number_input(label='maximum horizontal axis value', key='x_max')
            else:
                x_min = float(bb[0][0])
                x_max = float(bb[1][0])
            # if :
            # else:

            # y axis is z values
            if np.isinf(bb[0][2]) or np.isinf(bb[1][2]):
                y_min = col1.number_input(label='minimum vertical axis value', key='y_min')
                y_max = col1.number_input(label='maximum vertical axis value', key='y_max')
            else:
                y_min = float(bb[0][2])
                y_max = float(bb[1][2])
            # if :
            # else:

        if view_direction in ['x']:

            # x axis is y values
            if np.isinf(bb[0][1]) or np.isinf(bb[1][1]):
                x_min = col1.number_input(label='minimum vertical axis value')
                x_max = col1.number_input(label='maximum vertical axis value')
            else:
                x_min = float(bb[0][1])
                x_max = float(bb[1][1])
            # if :
            # else:

            # y axis is z values
            if np.isinf(bb[0][2]) or np.isinf(bb[1][2]):
                y_min = col1.number_input(label='minimum vertical axis value', key='y_min')
                y_max = col1.number_input(label='maximum vertical axis value', key='y_max')
            else:
                y_min = float(bb[0][1])
                y_max = float(bb[1][1])
            # if :
            # else:

        if x_min and x_max:
            plot_left, plot_right = col1.slider(
                label="Left and right values for the horizontal axis",
                min_value=x_min,
                max_value=x_max,
                value=(x_min, x_max),
                key='left_right_slider',
                help='Set the lowest visible value and highest visible value on the horizontal axis'
            )

        if y_min and y_max:
            plot_bottom, plot_top = col1.slider(
                label="Bottom and top values for the vertical axis",
                min_value=y_min,
                max_value=y_max,
                value=(y_min, y_max),
                key='bottom_top_slider',
                help='Set the lowest visible value and highest visible value on the vertical axis'
            )

        pixels_across = col1.number_input(
            label='Number of horizontal pixels',
            value=200,
            help='Increasing this value increases the image resolution but also requires longer to create the image'
        )

        title = col1.text_input('Plot title',
            help='Optionally set your own title for the plot',
            value=f'Slice through OpenMC geometry with view direction {view_direction}'
        )

        if plot_left and plot_right and plot_top and plot_bottom:
            geom_plt = plot_axis_slice(
                geometry=my_geometry,
                plot_left=plot_left,
                plot_right=plot_right,
                plot_top=plot_top,
                plot_bottom=plot_bottom,
                view_direction=view_direction,
                pixels_across=pixels_across,
                backend=backend,
                title=title
            )

            if backend == 'matplotlib':

                geom_plt.figure.savefig("openmc_plot_geometry_image.png")
                col2.pyplot(plt)
                # col2.image("openmc_plot_geometry_image.png", use_column_width="always")

                with open("openmc_plot_geometry_image.png", "rb") as file:
                    col1.download_button(
                        label="Download image",
                        data=file,
                        file_name="openmc_plot_geometry_image.png",
                        mime="image/png"
                    )
            else:

                geom_plt.write_html('openmc_plot_geometry_image.html')

                with open("openmc_plot_geometry_image.html", "rb") as file:
                    col1.download_button(
                        label="Download image",
                        data=file,
                        file_name="openmc_plot_geometry_image.html",
                        mime=None
                    )
                col2.plotly_chart(geom_plt, use_container_width=True)


if __name__ == "__main__":
    main()
