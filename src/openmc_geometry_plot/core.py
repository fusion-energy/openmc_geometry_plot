import os
import openmc
import numpy as np
import typing
from tempfile import mkdtemp
from pathlib import Path
import matplotlib.pyplot as plt
import math
from tempfile import TemporaryDirectory
import warnings
from PIL import Image
import matplotlib.image as mpimg
# import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from numpy import asarray


def get_rgb_from_int(value: int) -> typing.Tuple[int, int, int]:
    blue = value & 255
    green = (value >> 8) & 255
    red = (value >> 16) & 255
    return red, green, blue


def get_int_from_rgb(rgb: typing.Tuple[int, int, int]) -> int:
    red = rgb[0]
    green = rgb[1]
    blue = rgb[2]
    return (red << 16) + (green << 8) + blue

def get_hover_text_from_id(id, color_by):
    return f'{color_by} {id}'


def get_plot_extent(
    self, plot_left, plot_right, plot_bottom, plot_top, slice_value, bb, view_direction
):
    if view_direction not in ["x", "y", "z"]:
        raise ValueError('view_direction must be "x", "y" or "z"')

    if plot_left is None:
        plot_left = self.get_side_extent(
            side="left", bounding_box=bb, view_direction=view_direction
        )

    if plot_right is None:
        plot_right = self.get_side_extent(
            side="right", bounding_box=bb, view_direction=view_direction
        )

    if plot_bottom is None:
        plot_bottom = self.get_side_extent(
            side="bottom", bounding_box=bb, view_direction=view_direction
        )

    if plot_top is None:
        plot_top = self.get_side_extent(
            side="top", bounding_box=bb, view_direction=view_direction
        )

    if slice_value is None:
        slice_value = self.get_mid_slice_value(
            bounding_box=bb, view_direction=view_direction
        )
    return plot_left, plot_right, plot_bottom, plot_top, slice_value


def get_side_extent(self, side: str, view_direction: str, bounding_box=None):
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

    x_min = self.get_side_extent(
        side="left", view_direction=view_direction, bounding_box=bb
    )
    x_max = self.get_side_extent(
        side="right", view_direction=view_direction, bounding_box=bb
    )
    y_min = self.get_side_extent(
        side="bottom", view_direction=view_direction, bounding_box=bb
    )
    y_max = self.get_side_extent(
        side="top", view_direction=view_direction, bounding_box=bb
    )

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


def get_axis_labels(self, view_direction, axis_units):
    """Returns two axis label values for the x and y value. Takes
    view_direction into account."""

    if view_direction == "x":
        xlabel = f"Y [{axis_units}]"
        ylabel = f"Z [{axis_units}]"
    if view_direction == "y":
        xlabel = f"X [{axis_units}]"
        ylabel = f"Z [{axis_units}]"
    if view_direction == "z":
        xlabel = f"X [{axis_units}]"
        ylabel = f"Y [{axis_units}]"
    return xlabel, ylabel


def find_cell_id(self, inputs):
    plot_x, plot_y, slice_value = inputs
    return self.find((plot_x, plot_y, slice_value))


def is_geometry_dagmc(self):
    for univ in self.get_all_universes().values():
        if isinstance(univ, openmc.DAGMCUniverse):
            return True
    return False


def get_dagmc_filepath(self):
    "absolute path"
    for univ in self.get_all_universes().values():
        if isinstance(univ, openmc.DAGMCUniverse):
            return Path(univ.filename).absolute()
            # could try relative paths for more complex filenames
            # return Path(univ.filename).relative_to(Path(__file__).parent)


def get_dagmc_universe(self):
    for univ in self.get_all_universes().values():
        if isinstance(univ, openmc.DAGMCUniverse):
            return univ


def get_slice_of_material_ids(
    self,
    view_direction: str,
    slice_value: typing.Optional[float] = None,
    plot_top: typing.Optional[float] = None,
    plot_bottom: typing.Optional[float] = None,
    plot_left: typing.Optional[float] = None,
    plot_right: typing.Optional[float] = None,
    pixels_across: int = 500,
    verbose: bool = False,
):
    """Returns a grid of material IDs for each mesh voxel on the slice. This
    can be passed directly to plotting functions like Matplotlib imshow. 0
    values represent void space or undefined space.

    Args:
        slice_value:
        plot_top:
        plot_bottom:
        plot_left:
        plot_right:
        pixels_across:
        verbose:
    """

    tmp_folder = mkdtemp(prefix="openmc_geometry_plotter_tmp_files_")
    if verbose:
        print(f"writing files to {tmp_folder}")

    if self.is_geometry_dagmc():
        dagmc_abs_filepath = self.get_dagmc_filepath()

        os.system(f"cp {dagmc_abs_filepath} {tmp_folder}")

        dag_universe = self.get_dagmc_universe()

        if str(Path(dag_universe.filename).name) != str(Path(dag_universe.filename)):
            msg = (
                "Paths for dagmc files that contain folders are not currently "
                "supported. Try setting your DAGMCUniverse.filename "
                f"to {Path(dag_universe.filename).name} instead of "
                f"{Path(dag_universe.filename)}"
            )
            raise IsADirectoryError(msg)
        # dag_universe.filename = dagmc_abs_filepath.name

        # dagmuniverse does not have a get_all_materials
        mat_names = dag_universe.material_names

        # mat ids are not known by the dagmc file
        # assumed mat ids start at 1 and continue,
        # universe.n_cells is the equivilent approximation for cell ids
        mat_ids = range(1, len(mat_names) + 1)

        # if any of the plot_ are None then this needs calculating
        # might need to be self.bounding_box
        bb = dag_universe.bounding_box

    else:
        original_materials = self.get_all_materials()
        mat_ids = original_materials.keys()

        mat_names = []
        for key, value in original_materials.items():
            mat_names.append(value.name)

        # if any of the plot_ are None then this needs calculating
        bb = self.bounding_box

    self.export_to_xml(tmp_folder)

    plot_left, plot_right, plot_bottom, plot_top, slice_value = self.get_plot_extent(
        plot_left, plot_right, plot_bottom, plot_top, slice_value, bb, view_direction
    )

    plot_width = abs(plot_left - plot_right)
    plot_height = abs(plot_bottom - plot_top)

    aspect_ratio = plot_height / plot_width
    pixels_up = int(pixels_across * aspect_ratio)

    all_materials = []
    for i, n in zip(mat_ids, mat_names):
        new_mat = openmc.Material()
        new_mat.id = i
        new_mat.name = n
        new_mat.add_nuclide("He4", 1)
        all_materials.append(new_mat)
    nn = openmc.Materials(all_materials)
    nn.export_to_xml(tmp_folder)

    my_settings = openmc.Settings()
    my_settings.output = {"summary": False, "tallies": False}
    # add verbose setting to avoid print out
    my_settings.particles = 1
    my_settings.batches = 1
    my_settings.run_mode = "fixed source"
    my_settings.export_to_xml(tmp_folder)

    my_plot = openmc.Plot()

    plot_x = (plot_left + plot_right) / 2
    plot_y = (plot_top + plot_bottom) / 2

    width = plot_left - plot_right
    height = plot_top - plot_bottom
    my_plot.width = (width, height)

    if view_direction == "z":
        my_plot.basis = "xy"
        my_plot.origin = (plot_x, plot_y, slice_value)
    if view_direction == "x":
        my_plot.basis = "yz"
        my_plot.origin = (slice_value, plot_x, plot_y)
    if view_direction == "y":
        my_plot.basis = "xz"
        my_plot.origin = (plot_x, slice_value, plot_y)

    my_plot.pixels = (pixels_across, pixels_up)
    colors_dict = {}
    for mat_id in mat_ids:
        colors_dict[mat_id] = get_rgb_from_int(mat_id)
    my_plot.colors = colors_dict
    my_plot.background = (0, 0, 0)  # void material is 0
    my_plot.color_by = "material"
    my_plot.id = 42  # the integer used to name of the plot_1.ppm file
    my_plots = openmc.Plots([my_plot])
    my_plots.export_to_xml(tmp_folder)

    if 'cross_sections' in openmc.config.keys():
        original_cross_sections = openmc.config["cross_sections"]
    else:
        original_cross_sections = None

    package_dir = Path(__file__).parent
    openmc.config["cross_sections"] = package_dir / "cross_sections.xml"

    openmc.plot_geometry(cwd=tmp_folder, output=verbose)

    if original_cross_sections:
        openmc.config["cross_sections"] = original_cross_sections

    if verbose:
        print(f"Temporary image and xml files written to {tmp_folder}")

    # load the image
    if (Path(tmp_folder) / f"plot_{my_plot.id}.ppm").is_file():
        image = Image.open(Path(tmp_folder) / f"plot_{my_plot.id}.ppm")
    elif (Path(tmp_folder) / f"plot_{my_plot.id}.png").is_file():
        image = Image.open(Path(tmp_folder) / f"plot_{my_plot.id}.png")
    else:
        raise FileNotFoundError(f"openmc plot mode image was not found in {tmp_folder}")

    # convert the image to a numpy array
    image_values = asarray(image)

    # the image_values have three entries for RGB but we just need one.
    # this reduces the nested list to contain a single value per pixel
    # image_value = [
    #     [inner_entry[0] for inner_entry in outer_entry] for outer_entry in image_values
    # ]
    image_value = [
        [get_int_from_rgb(inner_entry) for inner_entry in outer_entry]
        for outer_entry in image_values
    ]

    # replaces rgb (255,255,255) which is 16777215 values with 0.
    # 0 is the color for void space
    # (255,255,255) gets returned by undefined regions outside the geometry
    trimmed_image_value = [
        [0 if x == 16777215 else x for x in inner_list] for inner_list in image_value
    ]

    return trimmed_image_value


def get_slice_of_cell_ids(
    self,
    view_direction: str,
    slice_value: typing.Optional[float] = None,
    plot_top: typing.Optional[float] = None,
    plot_bottom: typing.Optional[float] = None,
    plot_left: typing.Optional[float] = None,
    plot_right: typing.Optional[float] = None,
    pixels_across: int = 500,
    verbose: bool = False,
):
    """Returns a grid of cell IDs for each mesh voxel on the slice. This
    can be passed directly to plotting functions like Matplotlib imshow. 0
    values represent void space or undefined space.

    Args:
        slice_value:
        plot_top:
        plot_bottom:
        plot_left:
        plot_right:
        pixels_across:
        verbose:
    """

    tmp_folder = mkdtemp(prefix="openmc_geometry_plotter_tmp_files_")
    if verbose:
        print(f"writing files to {tmp_folder}")

    if self.is_geometry_dagmc():
        dagmc_abs_filepath = self.get_dagmc_filepath()

        os.system(f"cp {dagmc_abs_filepath} {tmp_folder}")

        dag_universe = self.get_dagmc_universe()

        if str(Path(dag_universe.filename).name) != str(Path(dag_universe.filename)):
            msg = (
                "Paths for dagmc files that contain folders are not currently "
                "supported. Try setting your DAGMCUniverse.filename "
                f"to {Path(dag_universe.filename).name} instead of "
                f"{Path(dag_universe.filename)}"
            )
            raise IsADirectoryError(msg)
        # dag_universe.filename = dagmc_abs_filepath.name

        # dagmuniverse does not have a get_all_materials
        mat_names = dag_universe.material_names

        # mat ids are not known by the dagmc file
        # assumed mat ids start at 1 and continue,
        # universe.n_cells is the equivilent approximation for cell ids
        mat_ids = range(1, len(mat_names) + 1)

        # if any of the plot_ are None then this needs calculating
        # might need to be self.bounding_box
        bb = dag_universe.bounding_box

    else:
        original_materials = self.get_all_materials()
        mat_ids = original_materials.keys()

        mat_names = []
        for key, value in original_materials.items():
            mat_names.append(value.name)

        bb = self.bounding_box

    self.export_to_xml(tmp_folder)

    plot_left, plot_right, plot_bottom, plot_top, slice_value = self.get_plot_extent(
        plot_left, plot_right, plot_bottom, plot_top, slice_value, bb, view_direction
    )

    plot_width = abs(plot_left - plot_right)
    plot_height = abs(plot_bottom - plot_top)

    aspect_ratio = plot_height / plot_width
    pixels_up = int(pixels_across * aspect_ratio)

    # might not work for dagmc model
    cell_ids = self.get_all_cells().keys()

    all_materials = []
    for i, n in zip(mat_ids, mat_names):
        new_mat = openmc.Material()
        new_mat.id = i
        new_mat.name = n
        new_mat.add_nuclide("He4", 1)
        all_materials.append(new_mat)
    nn = openmc.Materials(all_materials)
    nn.export_to_xml(tmp_folder)

    my_settings = openmc.Settings()
    my_settings.output = {"summary": False, "tallies": False}
    my_settings.particles = 1
    my_settings.batches = 1
    my_settings.run_mode = "fixed source"
    my_settings.export_to_xml(tmp_folder)

    my_plot = openmc.Plot()

    plot_x = (plot_left + plot_right) / 2
    plot_y = (plot_top + plot_bottom) / 2

    width = plot_left - plot_right
    height = plot_top - plot_bottom
    my_plot.width = (width, height)

    if view_direction == "z":
        my_plot.basis = "xy"
        my_plot.origin = (plot_x, plot_y, slice_value)
    if view_direction == "x":
        my_plot.basis = "yz"
        my_plot.origin = (slice_value, plot_x, plot_y)
    if view_direction == "y":
        my_plot.basis = "xz"
        my_plot.origin = (plot_x, slice_value, plot_y)

    my_plot.pixels = (pixels_across, pixels_up)
    # my_plot.pixels = (100,100)
    colors_dict = {}
    for cell_id in cell_ids:
        # TODO make use of rgb to int convertor
        colors_dict[cell_id] = get_rgb_from_int(cell_id)
    my_plot.colors = colors_dict
    my_plot.background = (0, 0, 0)  # void material is 0
    my_plot.color_by = "cell"
    my_plot.id = 24  # the integer used to name of the plot_1.ppm file
    my_plots = openmc.Plots([my_plot])
    my_plots.export_to_xml(tmp_folder)

    if 'cross_sections' in openmc.config.keys():
        original_cross_sections = openmc.config["cross_sections"]
    else:
        original_cross_sections = None

    # TODO unset this afterwards
    package_dir = Path(__file__).parent
    openmc.config["cross_sections"] = package_dir / "cross_sections.xml"

    openmc.plot_geometry(cwd=tmp_folder, output=verbose)

    if original_cross_sections:
        openmc.config["cross_sections"] = original_cross_sections

    if verbose:
        print(f"Temporary image and xml files written to {tmp_folder}")

    # load the image
    if (Path(tmp_folder) / f"plot_{my_plot.id}.ppm").is_file():
        image = Image.open(Path(tmp_folder) / f"plot_{my_plot.id}.ppm")
    elif (Path(tmp_folder) / f"plot_{my_plot.id}.png").is_file():
        image = Image.open(Path(tmp_folder) / f"plot_{my_plot.id}.png")
    else:
        raise FileNotFoundError(f"openmc plot mode image was not found in {tmp_folder}")

    # convert the image to a numpy array
    image_values = asarray(image)

    # the image_values have three entries for RGB but we just need one.
    # this reduces the nested list to contain a single value per pixel
    image_value = [
        [get_int_from_rgb(inner_entry) for inner_entry in outer_entry]
        for outer_entry in image_values
    ]

    # replaces rgb (255,255,255) which is 16777215 values with 0.
    # 0 is the color for void space
    # (255,255,255) gets returned by undefined regions outside the geometry
    trimmed_image_value = [
        [0 if x == 16777215 else x for x in inner_list] for inner_list in image_value
    ]

    return trimmed_image_value


def plot_plotly(
    geometry,
    origin=None,
    width=None,
    pixels=40000,
    basis='xy',
    color_by='cell',
    colors=None,
    seed=None,
    openmc_exec='openmc',
    axes=None,
    legend=False,
    axis_units='cm',
    # legend_kwargs=_default_legend_kwargs,
    outline=False,
    title='',
    **kwargs
):
        """Display a slice plot of the universe.

        Parameters
        ----------
        origin : iterable of float
            Coordinates at the origin of the plot. If left as None,
            universe.bounding_box.center will be used to attempt to ascertain
            the origin with infinite values being replaced by 0.
        width : iterable of float
            Width of the plot in each basis direction. If left as none then the
            universe.bounding_box.width() will be used to attempt to
            ascertain the plot width.  Defaults to (10, 10) if the bounding_box
            contains inf values
        pixels : Iterable of int or int
            If iterable of ints provided then this directly sets the number of
            pixels to use in each basis direction. If int provided then this
            sets the total number of pixels in the plot and the number of
            pixels in each basis direction is calculated from this total and
            the image aspect ratio.
        basis : {'xy', 'xz', 'yz'}
            The basis directions for the plot
        color_by : {'cell', 'material'}
            Indicate whether the plot should be colored by cell or by material
        colors : dict
            Assigns colors to specific materials or cells. Keys are instances of
            :class:`Cell` or :class:`Material` and values are RGB 3-tuples, RGBA
            4-tuples, or strings indicating SVG color names. Red, green, blue,
            and alpha should all be floats in the range [0.0, 1.0], for example:

            .. code-block:: python

               # Make water blue
               water = openmc.Cell(fill=h2o)
               universe.plot(..., colors={water: (0., 0., 1.))
        seed : int
            Seed for the random number generator
        openmc_exec : str
            Path to OpenMC executable.
        axes : matplotlib.Axes
            Axes to draw to

            .. versionadded:: 0.13.1
        legend : bool
            Whether a legend showing material or cell names should be drawn

            .. versionadded:: 0.14.0
        legend_kwargs : dict
            Keyword arguments passed to :func:`matplotlib.pyplot.legend`.

            .. versionadded:: 0.14.0
        outline : bool
            Whether outlines between color boundaries should be drawn

            .. versionadded:: 0.14.0
        axis_units : {'km', 'm', 'cm', 'mm'}
            Units used on the plot axis

            .. versionadded:: 0.14.0
        **kwargs
            Keyword arguments passed to :func:`matplotlib.pyplot.imshow`

        Returns
        -------
        matplotlib.axes.Axes
            Axes containing resulting image

        """


        # Determine extents of plot
        if basis == 'xy':
            x, y = 0, 1
            xlabel, ylabel = f'x [{axis_units}]', f'y [{axis_units}]'
        elif basis == 'yz':
            x, y = 1, 2
            xlabel, ylabel = f'y [{axis_units}]', f'z [{axis_units}]'
        elif basis == 'xz':
            x, y = 0, 2
            xlabel, ylabel = f'x [{axis_units}]', f'z [{axis_units}]'

        bb = geometry.bounding_box
        # checks to see if bounding box contains -inf or inf values
        if np.isinf(bb.extent[basis]).any():
            if origin is None:
                origin = (0, 0, 0)
            if width is None:
                width = (10, 10)
        else:
            if origin is None:
                # if nan values in the bb.center they get replaced with 0.0
                # this happens when the bounding_box contains inf values
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    origin = np.nan_to_num(bb.center)
            if width is None:
                bb_width = bb.width
                x_width = bb_width['xyz'.index(basis[0])]
                y_width = bb_width['xyz'.index(basis[1])]
                width = (x_width+3, y_width+3) # makes width a bit bigger so that the edges don't get cut

        if isinstance(pixels, int):
            aspect_ratio = width[0] / width[1]
            pixels_y = math.sqrt(pixels / aspect_ratio)
            pixels = (int(pixels / pixels_y), int(pixels_y))

        axis_scaling_factor = {'km': 0.00001, 'm': 0.01, 'cm': 1, 'mm': 10}

        x_min = (origin[x] - 0.5*width[0]) * axis_scaling_factor[axis_units]
        x_max = (origin[x] + 0.5*width[0]) * axis_scaling_factor[axis_units]
        y_min = (origin[y] - 0.5*width[1]) * axis_scaling_factor[axis_units]
        y_max = (origin[y] + 0.5*width[1]) * axis_scaling_factor[axis_units]

        with TemporaryDirectory() as tmpdir:
            model = openmc.Model()
            model.geometry = geometry
            if seed is not None:
                model.settings.plot_seed = seed

            # Determine whether any materials contains macroscopic data and if
            # so, set energy mode accordingly
            for mat in geometry.get_all_materials().values():
                if mat._macroscopic is not None:
                    model.settings.energy_mode = 'multi-group'
                    break

            # Create plot object matching passed arguments
            plot = openmc.Plot()
            plot.origin = origin
            plot.width = width
            plot.pixels = pixels
            plot.basis = basis
            plot.color_by = color_by

            if colors is not None:
                
                colors_based_on_ids = {}
                for key, value in colors.items():
                    colors_based_on_ids[key] = get_rgb_from_int(key.id)
                plot.colors = colors_based_on_ids

            model.plots.append(plot)

            # Run OpenMC in geometry plotting mode
            model.plot_geometry(False, cwd=tmpdir, openmc_exec=openmc_exec)

            # Read image from file
            img_path = Path(tmpdir) / f'plot_{plot.id}.png'
            if not img_path.is_file():
                img_path = img_path.with_suffix('.ppm')
            # todo see if we can just read in image once
            img = mpimg.imread(str(img_path))
            image_values = Image.open(img_path)
            image_values = np.asarray(image_values)
            image_values = [
                [get_int_from_rgb(inner_entry) for inner_entry in outer_entry]
                for outer_entry in image_values
            ]

            image_values = np.array(image_values)
            image_values[image_values == 16777215] = 0

            data = []

            if outline:
                data.append(
                    go.Contour(
                        z=image_values,
                        contours_coloring='none',
                        # colorscale=dcolorsc,
                        showscale=False,
                        x0=x_min,
                        dx=abs(x_min - x_max) / (img.shape[0] - 1),
                        y0=y_min,
                        dy=abs(y_min - y_max) / (img.shape[1] - 1),
                    )
                )
            
            dcolorsc=[
                [0, 'white'],
            ]

            rgb_cols = [f'rgb({c[0]},{c[1]},{c[2]})' for c in list(colors.values())]
            mat_ids=[mat.id for mat in colors.keys()]
            highest_mat_id = max(mat_ids)
            for rgb_col, mat_id in zip(rgb_cols, mat_ids):
                dcolorsc.append(((1/highest_mat_id)*mat_id,rgb_col))

            cbar = dict(
                        tick0= 0,
                        xref="container",
                        tickmode= 'array',
                        tickvals= mat_ids,
                        ticktext= mat_ids, # TODO add material names
                        title=f'{color_by.title()} IDs',
                    )

            hovertext = [
                [get_hover_text_from_id(inner_entry, color_by) for inner_entry in outer_entry]
                for outer_entry in image_values
            ]

            data.append(
                go.Heatmap(
                    z=image_values,
                    # showscale=True,
                    colorscale=dcolorsc,
                    x0=x_min,
                    dx=abs(x_min - x_max) / (img.shape[0] - 1),
                    y0=y_min,
                    dy=abs(y_min - y_max) / (img.shape[1] - 1),
                    colorbar= cbar,
                    showscale=legend,
                    hoverinfo='text',
                    text=hovertext
                )
            )
            plot = go.Figure(data=data)
           

            plot.update_layout(
                xaxis={"title": xlabel},
                # reversed autorange is required to avoid image needing rotation/flipping in plotly
                yaxis={"title": ylabel, "autorange": "reversed"},
                # title=title,
                autosize=False,
                height=800,
                title=title
            )
            plot.update_yaxes(
                scaleanchor="x",
                scaleratio=1,
            )
            return plot

# patching openmc

openmc.Geometry.get_dagmc_universe = get_dagmc_universe
openmc.Geometry.is_geometry_dagmc = is_geometry_dagmc
openmc.Geometry.get_dagmc_filepath = get_dagmc_filepath
openmc.Geometry.get_plot_extent = get_plot_extent
openmc.Geometry.get_side_extent = get_side_extent
openmc.Geometry.get_mpl_plot_extent = get_mpl_plot_extent
openmc.Geometry.get_mid_slice_value = get_mid_slice_value
openmc.Geometry.get_axis_labels = get_axis_labels
openmc.Geometry.get_slice_of_material_ids = get_slice_of_material_ids
openmc.Geometry.get_slice_of_cell_ids = get_slice_of_cell_ids
openmc.Geometry.find_cell_id = find_cell_id


openmc.DAGMCUniverse.get_all_universes = openmc.Universe.get_all_universes
