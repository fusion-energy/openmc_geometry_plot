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


def main():


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

        view_direction = col1.selectbox(label="view_direction", options=("z", "x", "y"), index=0)
        backend = col1.selectbox(label="backend", options=("matplotlib", "plotly"), index=0)

        if view_direction == 'x':
             x_offset = col1.slider(
                label="left and right values",
                min_value=float(bb[0][0]),
                max_value=float(bb[1][0]),
                value=(float((bb[0][0] + bb[1][0]) / 3), float((bb[0][0] + bb[1][0]) / 2)),
            )

        # # # bb may have -inf or inf values in, these break the slider bar automatic scaling
        # if np.isinf(bb[0][0]) or np.isinf(bb[1][0]):
        #     msg = "Infinity value found in X axis, axis length can't be automatically found. Input desired Z axis length"
        #     x_width = col1.number_input(msg, value=1.0)
        #     x_offset = col1.number_input("X axis offset")
        # else:
        #     x_width = abs(bb[0][0] - bb[1][0])
        #     x_offset = col1.slider(
        #         label="X axis offset",
        #         min_value=float(bb[0][0]),
        #         max_value=float(bb[1][0]),
        #         value=float((bb[0][0] + bb[1][0]) / 2),
        #     )

        # if np.isinf(bb[0][1]) or np.isinf(bb[1][1]):
        #     msg = "Infinity value found in Y axis, axis length can't be automatically found. Input desired Z axis length"
        #     y_width = col1.number_input(msg, value=1.0)
        #     y_offset = col1.number_input("Y axis offset")
        # else:
        #     y_width = abs(bb[0][1] - bb[1][1])
        #     y_offset = col1.slider(
        #         label="Y axis offset",
        #         min_value=float(bb[0][1]),
        #         max_value=float(bb[1][1]),
        #         value=float((bb[0][1] + bb[1][1]) / 2),
        #     )

        # if np.isinf(bb[0][2]) or np.isinf(bb[1][2]):
        #     msg = "Infinity value found in Z axis, axis length can't be automatically found. Input desired Z axis length"
        #     z_width = col1.number_input(msg, value=1.0)
        #     z_offset = col1.number_input("Z axis offset")
        # else:
        #     z_width = abs(bb[0][2] - bb[1][2])
        #     z_offset = col1.slider(
        #         label="Z axis offset",
        #         min_value=float(bb[0][2]),
        #         max_value=float(bb[1][2]),
        #         value=float((bb[0][2] + bb[1][2]) / 2),
        #     )
        # left = 
        
        pixels_across = int(col1.number_input('pixels_across', value=200))

        geom_plt = plot_axis_slice(
            geometry=my_geometry,
            plot_left=-100,
            plot_right=100,
            plot_top=-200,
            plot_bottom=200,
            view_direction=view_direction,
            pixels_across=pixels_across,
            backend=backend,
        )

        if backend == 'matplotlib':

            geom_plt.figure.savefig("openmc_plot_geometry_image.png")

            col2.image("openmc_plot_geometry_image.png", use_column_width="always")

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
            col1.plotly_chart(figure, use_container_width=True, height=800)




if __name__ == "__main__":
    main()
