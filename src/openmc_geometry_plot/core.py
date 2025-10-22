import openmc
import numpy as np
import typing
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter threading issues
import matplotlib.pyplot as plt
import math
import warnings
import plotly.graph_objects as go


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


def _calculate_plot_parameters(
    geometry,
    view_direction: str,
    slice_value: typing.Optional[float],
    plot_left: typing.Optional[float],
    plot_right: typing.Optional[float],
    plot_bottom: typing.Optional[float],
    plot_top: typing.Optional[float],
):
    """Helper function to calculate origin, width, and basis for plotting.

    Args:
        geometry: OpenMC Geometry object
        view_direction: 'x', 'y', or 'z'
        slice_value: Position along the view direction
        plot_left: Left edge of plot
        plot_right: Right edge of plot
        plot_bottom: Bottom edge of plot
        plot_top: Top edge of plot

    Returns:
        tuple: (origin, width, basis) suitable for model.id_map()
    """
    bb = geometry.bounding_box

    # Get plot extent
    plot_left, plot_right, plot_bottom, plot_top, slice_value = geometry.get_plot_extent(
        plot_left, plot_right, plot_bottom, plot_top, slice_value, bb, view_direction
    )

    # Calculate center and dimensions
    plot_x = (plot_left + plot_right) / 2
    plot_y = (plot_top + plot_bottom) / 2
    width_x = abs(plot_left - plot_right)
    width_y = abs(plot_top - plot_bottom)

    # Map view_direction to basis and origin
    if view_direction == "z":
        basis = "xy"
        origin = (plot_x, plot_y, slice_value)
        width = (width_x, width_y)
    elif view_direction == "x":
        basis = "yz"
        origin = (slice_value, plot_x, plot_y)
        width = (width_x, width_y)
    elif view_direction == "y":
        basis = "xz"
        origin = (plot_x, slice_value, plot_y)
        width = (width_x, width_y)
    else:
        raise ValueError(f'view_direction must be "x", "y" or "z", not {view_direction}')

    return origin, width, basis


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
        view_direction: 'x', 'y', or 'z'
        slice_value: Position along the view direction axis
        plot_top: Top edge of the plot region
        plot_bottom: Bottom edge of the plot region
        plot_left: Left edge of the plot region
        plot_right: Right edge of the plot region
        pixels_across: Number of pixels in the horizontal direction
        verbose: Print verbose output

    Returns:
        list: 2D list of material IDs
    """
    # Calculate plot parameters
    origin, width, basis = _calculate_plot_parameters(
        self, view_direction, slice_value,
        plot_left, plot_right, plot_bottom, plot_top
    )

    # Calculate pixel dimensions maintaining aspect ratio
    aspect_ratio = width[1] / width[0]
    pixels_up = int(pixels_across * aspect_ratio)

    if verbose:
        print(f"Generating material ID map: origin={origin}, width={width}, basis={basis}, pixels=({pixels_across}, {pixels_up})")

    # Create a model from the geometry
    # Handle DAGMC geometries which need materials created from material names
    if self.is_geometry_dagmc():
        dag_universe = self.get_dagmc_universe()
        mat_names = dag_universe.material_names

        # Create dummy materials for DAGMC
        all_materials_list = []
        for i, mat_name in enumerate(mat_names):
            mat = openmc.Material(name=mat_name)
            mat.add_nuclide("He4", 1.0)
            all_materials_list.append(mat)

        model = openmc.Model(geometry=self)
        model.materials = openmc.Materials(all_materials_list)
        materials_to_restore = {}
        all_materials = {}  # Empty for DAGMC
    else:
        # For regular geometries, add dummy nuclides to empty materials
        all_materials = self.get_all_materials()
        materials_to_restore = {}
        for mat_id, mat in all_materials.items():
            if len(mat.nuclides) == 0 and mat._macroscopic is None:
                materials_to_restore[mat_id] = mat.nuclides.copy()
                mat.add_nuclide("He4", 1.0)

        model = openmc.Model(geometry=self)

    # Set up minimal cross sections if not already configured
    original_cross_sections = None
    if 'cross_sections' not in openmc.config.keys():
        package_dir = Path(__file__).parent
        openmc.config["cross_sections"] = package_dir / "cross_sections.xml"
    else:
        original_cross_sections = openmc.config["cross_sections"]

    try:
        # Use the new id_map API to get material IDs
        id_map = model.id_map(
            origin=origin,
            width=width,
            pixels=(pixels_across, pixels_up),
            basis=basis,
        )
    finally:
        # Finalize OpenMC library to clean up resources
        try:
            from openmc import lib as openmc_lib
            if openmc_lib.is_initialized:
                openmc_lib.finalize()
        except (ImportError, AttributeError):
            pass  # OpenMC lib not available or not initialized

        # Restore original materials
        for mat_id in materials_to_restore:
            all_materials[mat_id].nuclides.clear()

        # Restore original cross sections config
        if original_cross_sections is not None:
            openmc.config["cross_sections"] = original_cross_sections

    # Extract material IDs (index 2 in the last dimension)
    # id_map shape is (vertical_pixels, horizontal_pixels, 3)
    # where last dimension is [cell_id, cell_instance, material_id]
    material_ids = id_map[:, :, 2]

    # Convert negative IDs (void/undefined) to 0 for backward compatibility
    # OpenMC returns -1 for void and -2 for undefined regions
    material_ids = np.where(material_ids < 0, 0, material_ids)

    return material_ids.tolist()


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
        view_direction: 'x', 'y', or 'z'
        slice_value: Position along the view direction axis
        plot_top: Top edge of the plot region
        plot_bottom: Bottom edge of the plot region
        plot_left: Left edge of the plot region
        plot_right: Right edge of the plot region
        pixels_across: Number of pixels in the horizontal direction
        verbose: Print verbose output

    Returns:
        list: 2D list of cell IDs
    """
    # Calculate plot parameters
    origin, width, basis = _calculate_plot_parameters(
        self, view_direction, slice_value,
        plot_left, plot_right, plot_bottom, plot_top
    )

    # Calculate pixel dimensions maintaining aspect ratio
    aspect_ratio = width[1] / width[0]
    pixels_up = int(pixels_across * aspect_ratio)

    if verbose:
        print(f"Generating cell ID map: origin={origin}, width={width}, basis={basis}, pixels=({pixels_across}, {pixels_up})")

    # Create a model from the geometry
    # Handle DAGMC geometries which need materials created from material names
    if self.is_geometry_dagmc():
        dag_universe = self.get_dagmc_universe()
        mat_names = dag_universe.material_names

        # Create dummy materials for DAGMC
        all_materials_list = []
        for i, mat_name in enumerate(mat_names):
            mat = openmc.Material(name=mat_name)
            mat.add_nuclide("He4", 1.0)
            all_materials_list.append(mat)

        model = openmc.Model(geometry=self)
        model.materials = openmc.Materials(all_materials_list)
        materials_to_restore = {}
        all_materials = {}  # Empty for DAGMC
    else:
        # For regular geometries, add dummy nuclides to empty materials
        all_materials = self.get_all_materials()
        materials_to_restore = {}
        for mat_id, mat in all_materials.items():
            if len(mat.nuclides) == 0 and mat._macroscopic is None:
                materials_to_restore[mat_id] = mat.nuclides.copy()
                mat.add_nuclide("He4", 1.0)

        model = openmc.Model(geometry=self)

    # Set up minimal cross sections if not already configured
    original_cross_sections = None
    if 'cross_sections' not in openmc.config.keys():
        package_dir = Path(__file__).parent
        openmc.config["cross_sections"] = package_dir / "cross_sections.xml"
    else:
        original_cross_sections = openmc.config["cross_sections"]

    try:
        # Use the new id_map API to get cell IDs
        id_map = model.id_map(
            origin=origin,
            width=width,
            pixels=(pixels_across, pixels_up),
            basis=basis,
        )
    finally:
        # Finalize OpenMC library to clean up resources
        try:
            from openmc import lib as openmc_lib
            if openmc_lib.is_initialized:
                openmc_lib.finalize()
        except (ImportError, AttributeError):
            pass  # OpenMC lib not available or not initialized

        # Restore original materials
        for mat_id in materials_to_restore:
            all_materials[mat_id].nuclides.clear()

        # Restore original cross sections config
        if original_cross_sections is not None:
            openmc.config["cross_sections"] = original_cross_sections

    # Extract cell IDs (index 0 in the last dimension)
    # id_map shape is (vertical_pixels, horizontal_pixels, 3)
    # where last dimension is [cell_id, cell_instance, material_id]
    cell_ids = id_map[:, :, 0]

    # Convert negative IDs (void/undefined) to 0 for backward compatibility
    # OpenMC returns -1 for void and -2 for undefined regions
    cell_ids = np.where(cell_ids < 0, 0, cell_ids)

    return cell_ids.tolist()



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

        # Use model.id_map() to get ID data directly (no file I/O needed!)
        model = openmc.Model(geometry=geometry)

        # Handle materials without nuclides
        all_materials = geometry.get_all_materials()
        materials_to_restore = {}
        for mat_id, mat in all_materials.items():
            if len(mat.nuclides) == 0 and mat._macroscopic is None:
                materials_to_restore[mat_id] = mat.nuclides.copy()
                mat.add_nuclide("He4", 1.0)

        # Set up minimal cross sections if not already configured
        original_cross_sections = None
        if 'cross_sections' not in openmc.config.keys():
            package_dir = Path(__file__).parent
            openmc.config["cross_sections"] = package_dir / "cross_sections.xml"
        else:
            original_cross_sections = openmc.config["cross_sections"]

        try:
            # Get ID map directly from OpenMC
            id_map = model.id_map(
                origin=origin,
                width=width,
                pixels=pixels,
                basis=basis,
            )

            # Extract the appropriate IDs based on color_by parameter
            if color_by == 'material':
                image_values = id_map[:, :, 2]  # Material IDs
            else:  # color_by == 'cell'
                image_values = id_map[:, :, 0]  # Cell IDs

            # Convert negative IDs (void/undefined) to 0
            image_values = np.where(image_values < 0, 0, image_values)

        finally:
            # Finalize OpenMC library to clean up resources
            try:
                from openmc import lib as openmc_lib
                if openmc_lib.is_initialized:
                    openmc_lib.finalize()
            except (ImportError, AttributeError):
                pass

            # Restore original materials
            for mat_id in materials_to_restore:
                all_materials[mat_id].nuclides.clear()

            # Restore original cross sections config
            if original_cross_sections is not None:
                openmc.config["cross_sections"] = original_cross_sections

        # Build the plotly figure
        data = []

        if outline:
            data.append(
                go.Contour(
                    z=image_values,
                    contours_coloring='none',
                    showscale=False,
                    x0=x_min,
                    dx=abs(x_min - x_max) / (image_values.shape[1] - 1),
                    y0=y_min,
                    dy=abs(y_min - y_max) / (image_values.shape[0] - 1),
                )
            )

        # Build discrete color scale for cell/material IDs
        # Handle empty colors dict (e.g., void-only geometry)
        if colors and len(colors) > 0:
            # Get all unique IDs from the image data
            unique_ids = np.unique(image_values)
            unique_ids = unique_ids[unique_ids != 0]  # Remove void (0)

            # Build a mapping from ID to color
            id_to_color = {}
            for item, rgb in colors.items():
                id_to_color[item.id] = f'rgb({rgb[0]},{rgb[1]},{rgb[2]})'

            # Create discrete colorscale
            # For discrete colors, we need to create steps where each ID gets exactly one color
            all_ids = sorted(set(list(id_to_color.keys()) + [0]))  # Include 0 for void
            max_id = max(all_ids) if all_ids else 1

            # Build a stepped colorscale
            dcolorsc = []
            for i, id_val in enumerate(all_ids):
                if id_val == 0:
                    color = 'white'
                else:
                    color = id_to_color.get(id_val, 'rgb(128,128,128)')  # Gray for unmapped IDs

                # Create discrete steps by adding the same color at boundaries
                if i == 0:
                    dcolorsc.append([id_val / max_id, color])
                else:
                    # Add a tiny step before this ID to keep previous color
                    dcolorsc.append([(id_val - 0.0001) / max_id, prev_color])
                    dcolorsc.append([id_val / max_id, color])

                prev_color = color

            # Add final point
            dcolorsc.append([1.0, prev_color])

            cbar = dict(
                tick0=0,
                xref="container",
                tickmode='array',
                tickvals=sorted([id for id in id_to_color.keys()]),
                ticktext=sorted([id for id in id_to_color.keys()]),  # TODO: add material/cell names
                title=f'{color_by.title()} IDs',
            )
        else:
            # Default colorbar for void-only or empty geometries
            dcolorsc = [[0, 'white'], [1, 'white']]
            cbar = dict(
                title=f'{color_by.title()} IDs',
            )

        hovertext = [
            [get_hover_text_from_id(inner_entry, color_by) for inner_entry in outer_entry]
            for outer_entry in image_values
        ]

        data.append(
            go.Heatmap(
                z=image_values,
                colorscale=dcolorsc,
                x0=x_min,
                dx=abs(x_min - x_max) / (image_values.shape[1] - 1),
                y0=y_min,
                dy=abs(y_min - y_max) / (image_values.shape[0] - 1),
                colorbar=cbar,
                showscale=legend,
                hoverinfo='text',
                text=hovertext,
                zmin=0,
                zmax=max(all_ids) if (colors and len(colors) > 0) else 1,
            )
        )

        plot = go.Figure(data=data)

        plot.update_layout(
            xaxis={"title": xlabel},
            # reversed autorange is required to avoid image needing rotation/flipping in plotly
            yaxis={"title": ylabel, "autorange": "reversed"},
            autosize=False,
            height=800,
            title=title,
            # Enable box select mode for Streamlit's on_select to capture regions
            dragmode='select',  # Default to box select mode
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
