import openmc
import numpy as np
import typing
from pathlib import Path
import math
import warnings
import plotly.graph_objects as go
import hashlib
import json


# Cache for id_map results to avoid redundant expensive calls
_id_map_cache = {}

# Cache for hovertext to avoid regenerating on every plot update
_hovertext_cache = {}


def _get_cache_key(geometry, materials, origin, width, pixels, basis, show_overlaps):
    """Generate a cache key for id_map parameters.
    
    Uses content-based hashing instead of object IDs to avoid cache misses
    when geometry/materials objects are recreated with same content.
    """
    # Create a content-based hash for geometry
    # Use the geometry's XML representation or bounding box + cell/material counts
    try:
        geom_key_parts = [
            str(geometry.bounding_box),
            str(len(geometry.get_all_cells())),
            str(len(geometry.get_all_materials())),
            str(sorted(geometry.get_all_cells().keys())),
            str(sorted(geometry.get_all_materials().keys())),
        ]
        geom_hash = hashlib.md5(''.join(geom_key_parts).encode()).hexdigest()
    except:
        # Fallback to object ID if we can't get content
        geom_hash = str(id(geometry))
    
    # Create a content-based hash for materials
    try:
        mat_ids = sorted([m.id for m in materials])
        mat_hash = hashlib.md5(str(mat_ids).encode()).hexdigest()
    except:
        mat_hash = str(id(materials))
    
    # Serialize other parameters
    params = {
        'geom_hash': geom_hash,
        'mat_hash': mat_hash,
        'origin': tuple(origin),
        'width': tuple(width),
        'pixels': pixels if isinstance(pixels, int) else tuple(pixels),
        'basis': basis,
        'show_overlaps': show_overlaps
    }
    
    # Create hash from parameters
    cache_key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
    return cache_key


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


def get_id_map_cached(
    geometry,
    materials,
    origin,
    width,
    pixels,
    basis,
    show_overlaps
):
    """Cached wrapper for model.id_map to avoid redundant expensive calls.
    
    This function will only recompute if any of the parameters change.
    """
    import time
    start_time = time.time()
    
    # Check cache first
    cache_key = _get_cache_key(geometry, materials, origin, width, pixels, basis, show_overlaps)
    
    if cache_key in _id_map_cache:
        elapsed = time.time() - start_time
        print(f"✓ Using cached ID map (cache key: {cache_key[:8]}...) - {elapsed:.3f}s")
        return _id_map_cache[cache_key]
    
    print(f"⚙ Computing new ID map (cache key: {cache_key[:8]}...)")
    compute_start = time.time()
    
    # Use model.id_map() to get ID data directly (no file I/O needed!)
    model = openmc.Model(geometry=geometry, materials=materials)

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
        print(f"  → Calling model.id_map()...")
        idmap_start = time.time()
        id_map = model.id_map(
            origin=origin,
            width=width,
            pixels=pixels,
            basis=basis,
            color_overlaps=show_overlaps,  # enables id_map to return -3 for overlaps
        )
        idmap_elapsed = time.time() - idmap_start
        print(f"  → model.id_map() completed in {idmap_elapsed:.3f}s")
        
        # Store in cache before returning
        _id_map_cache[cache_key] = id_map
        
        # Limit cache size to prevent memory issues (keep last 10 results)
        if len(_id_map_cache) > 10:
            # Remove oldest entry
            oldest_key = next(iter(_id_map_cache))
            del _id_map_cache[oldest_key]
        
        total_elapsed = time.time() - start_time
        print(f"✓ ID map computed and cached - total {total_elapsed:.3f}s")
        return id_map

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


def plot_plotly(
    geometry,
    materials,
    origin=None,
    width=None,
    pixels=40000,
    basis='xy',
    color_by='cell',
    colors=None,
    legend=False,
    axis_units='cm',
    # legend_kwargs=_default_legend_kwargs,
    outline=False,
    title='',
    show_overlaps=False,
    overlap_color=None
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

        import time
        plot_start = time.time()
        print(f"📊 Starting plot_plotly() - legend={legend}, color_by={color_by}")

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

        # Get ID map using cached function to avoid redundant expensive calls
        idmap_fetch_start = time.time()
        id_map = get_id_map_cached(
            geometry=geometry,
            materials=materials,
            origin=origin,
            width=width,
            pixels=pixels,
            basis=basis,
            show_overlaps=show_overlaps
        )
        idmap_fetch_time = time.time() - idmap_fetch_start
        print(f"  → ID map fetch took {idmap_fetch_time:.3f}s")

        # Extract the appropriate IDs based on color_by parameter
        if color_by == 'material':
            image_values = id_map[:, :, 2]  # Material IDs
        else:  # color_by == 'cell'
            image_values = id_map[:, :, 0]  # Cell IDs

        # Handle overlaps (-3) and convert other negative IDs (void/undefined) to 0
        # We use -3 as a special marker for overlaps that will be colored separately
        if show_overlaps:
            # Keep -3 for overlaps, convert other negative values to 0
            image_values = np.where((image_values < 0) & (image_values != -3), 0, image_values)
        else:
            # Convert all negative IDs to 0
            image_values = np.where(image_values < 0, 0, image_values)

        # Flip image vertically for correct display orientation in Plotly
        # Plotly heatmap shows first row at bottom, but we want first row at top
        # This needs to be done early as it's used in multiple places
        image_values_flipped = np.flipud(image_values)

        # Build the plotly figure
        data = []

        if outline:
            data.append(
                go.Contour(
                    z=image_values_flipped,
                    contours_coloring='none',
                    showscale=False,
                    x0=x_min,
                    dx=abs(x_min - x_max) / (image_values_flipped.shape[1] - 1),
                    y0=y_min,
                    dy=abs(y_min - y_max) / (image_values_flipped.shape[0] - 1),
                )
            )

        # Build discrete color scale for cell/material IDs
        colorscale_start = time.time()
        # Handle empty colors dict (e.g., void-only geometry)
        if colors and len(colors) > 0:
            # Get all unique IDs from the image data
            unique_ids = np.unique(image_values)
            unique_ids = unique_ids[(unique_ids != 0) & (unique_ids != -3)]  # Remove void (0) and overlap (-3)

            # Build a mapping from ID to color
            id_to_color = {}
            for item, rgb in colors.items():
                id_to_color[item.id] = f'rgb({rgb[0]},{rgb[1]},{rgb[2]})'
            
            # Add overlap color if enabled
            if show_overlaps and overlap_color is not None:
                id_to_color[-3] = f'rgb({overlap_color[0]},{overlap_color[1]},{overlap_color[2]})'

            # Create discrete colorscale
            # For discrete colors, we need to create steps where each ID gets exactly one color
            # Include ALL unique IDs from the image data, not just those with assigned colors
            all_ids = sorted(set(list(id_to_color.keys()) + list(unique_ids) + [0]))  # Include 0 for void and all IDs from image
            
            # Remap -3 to a position at the bottom of the colorscale (below void)
            # We'll map it to -1 to put it at the bottom
            overlap_display_value = -1 if show_overlaps and -3 in all_ids else None
            
            # Create a remapped image for display purposes
            image_values_display = image_values.copy()
            if overlap_display_value is not None:
                image_values_display = np.where(image_values == -3, overlap_display_value, image_values)
            
            # Update the flipped image with the remapped values
            image_values_flipped = np.flipud(image_values_display)
            
            # Remove -3 from all_ids and add the remapped value if needed
            # Also create a mapping from display value to original ID for labels
            display_to_original = {}
            if -3 in all_ids:
                all_ids.remove(-3)
                if overlap_display_value is not None:
                    all_ids.insert(0, overlap_display_value)  # Insert at beginning (bottom of scale)
                    display_to_original[overlap_display_value] = -3
            
            # Map all other IDs to themselves
            for id_val in all_ids:
                if id_val not in display_to_original:
                    display_to_original[id_val] = id_val
            
            max_id = max(all_ids) if all_ids else 1
            min_id = min(all_ids) if all_ids else 0

            # Build a discrete colorscale
            # Give each ID an equal-width block in the colorscale for better visibility
            num_ids = len(all_ids)
            num_ids_inv = 1.0 / num_ids  # Pre-compute to avoid division in loop
            dcolorsc = []
            tick_positions = []
            tick_labels = []
            
            for i, id_val in enumerate(all_ids):
                original_id = display_to_original[id_val]
                
                if id_val == 0:
                    color = 'white'
                elif original_id == -3:
                    color = id_to_color.get(-3, 'rgb(255,0,0)')  # Red default for overlaps
                else:
                    color = id_to_color.get(original_id, 'rgb(128,128,128)')  # Gray for unmapped IDs

                # Give each ID an equal fraction of the colorscale [0, 1]
                block_start = i * num_ids_inv
                block_end = (i + 1) * num_ids_inv
                
                # Create discrete color blocks in normalized [0, 1] space
                if i == 0:
                    dcolorsc.append([0.0, color])
                else:
                    # Add tiny step just before this block to maintain previous color
                    dcolorsc.append([block_start - 0.00001, dcolorsc[-1][1]])
                    dcolorsc.append([block_start, color])
                
                # Extend the last block to 1.0
                if i == num_ids - 1:
                    dcolorsc.append([1.0, color])
                
                # Place tick at the center of this block in the colorbar
                # Block center in normalized [0,1] space
                block_center_norm = (block_start + block_end) / 2.0
                
                # Map normalized position back to data value range [min_id, max_id]
                tick_pos = min_id + block_center_norm * (max_id - min_id)
                tick_positions.append(tick_pos)
                
                # Determine the label using original ID
                if original_id == -3:
                    tick_labels.append("Overlap")
                elif original_id == 0:
                    tick_labels.append("void")
                else:
                    tick_labels.append(str(original_id))

            cbar = dict(
                tick0=0,
                xref="container",
                tickmode='array',
                tickvals=tick_positions,
                ticktext=tick_labels,
                title=f'{color_by.title()} IDs',
            )
            
            # Use max_id for zmax since ticks are now at actual ID values
            zmax_value = max_id
            
        else:
            # Default colorbar for void-only or empty geometries
            # image_values_flipped was already created earlier
            dcolorsc = [[0, 'white'], [1, 'white']]
            cbar = dict(
                title=f'{color_by.title()} IDs',
            )
        
        colorscale_time = time.time() - colorscale_start
        print(f"  → Colorscale building took {colorscale_time:.3f}s")

        # Build comprehensive hover text with both cell and material info
        hover_start = time.time()
        
        # Create a cache key for hovertext based on the id_map cache key
        # Hovertext depends on: id_map data, show_overlaps, and cell/material names
        idmap_cache_key = _get_cache_key(geometry, materials, origin, width, pixels, basis, show_overlaps)
        hover_cache_key = f"{idmap_cache_key}_hover"
        
        if hover_cache_key in _hovertext_cache:
            hovertext = _hovertext_cache[hover_cache_key]
            hover_time = time.time() - hover_start
            print(f"  → Using cached hovertext - {hover_time:.3f}s")
        else:
            all_cells = geometry.get_all_cells()
            all_materials = geometry.get_all_materials()

            # Extract cell and material IDs from id_map (before any flipping)
            cell_ids = id_map[:, :, 0].astype(int)
            mat_ids = id_map[:, :, 2].astype(int)
            
            # Handle negative IDs (convert to 0 for void, except -3 for overlaps)
            if show_overlaps:
                cell_ids = np.where((cell_ids < 0) & (cell_ids != -3), 0, cell_ids)
                mat_ids = np.where((mat_ids < 0) & (mat_ids != -3), 0, mat_ids)
            else:
                cell_ids = np.where(cell_ids < 0, 0, cell_ids)
                mat_ids = np.where(mat_ids < 0, 0, mat_ids)
            
            # Pre-build lookup dictionaries for cell and material names
            cell_name_lookup = {}
            for cell_id, cell in all_cells.items():
                if cell.name:
                    cell_name_lookup[cell_id] = f"Cell: {cell_id} ({cell.name})"
                else:
                    cell_name_lookup[cell_id] = f"Cell: {cell_id}"
            cell_name_lookup[0] = "Cell: void"
            cell_name_lookup[-3] = "OVERLAP DETECTED"
            
            mat_name_lookup = {}
            for mat_id, mat in all_materials.items():
                if mat.name:
                    mat_name_lookup[mat_id] = f"Material: {mat_id} ({mat.name})"
                else:
                    mat_name_lookup[mat_id] = f"Material: {mat_id}"
            mat_name_lookup[0] = "Material: void"
            
            # Vectorized hovertext generation
            # Flip the ID arrays to match the display orientation
            cell_ids_flipped = np.flipud(cell_ids)
            mat_ids_flipped = np.flipud(mat_ids)
            
            # Build hovertext array
            hovertext = []
            for row_idx in range(cell_ids_flipped.shape[0]):
                row_text = []
                for col_idx in range(cell_ids_flipped.shape[1]):
                    cell_id = cell_ids_flipped[row_idx, col_idx]
                    mat_id = mat_ids_flipped[row_idx, col_idx]
                    
                    # Check for overlap first
                    if show_overlaps and (cell_id == -3 or mat_id == -3):
                        row_text.append("OVERLAP DETECTED")
                    else:
                        # Use lookup dictionaries for fast access
                        cell_text = cell_name_lookup.get(cell_id, f"Cell: {cell_id}")
                        mat_text = mat_name_lookup.get(mat_id, f"Material: {mat_id}")
                        row_text.append(f"{cell_text}<br>{mat_text}")
                hovertext.append(row_text)
            
            # Cache the hovertext
            _hovertext_cache[hover_cache_key] = hovertext
            
            # Limit cache size
            if len(_hovertext_cache) > 10:
                oldest_key = next(iter(_hovertext_cache))
                del _hovertext_cache[oldest_key]
            
            hover_time = time.time() - hover_start
            print(f"  → Hovertext generated and cached - {hover_time:.3f}s ({cell_ids.shape[0]}x{cell_ids.shape[1]} pixels)")

        plotly_build_start = time.time()
        data.append(
            go.Heatmap(
                z=image_values_flipped,
                colorscale=dcolorsc,
                x0=x_min,
                dx=abs(x_min - x_max) / (image_values_flipped.shape[1] - 1),
                y0=y_min,
                dy=abs(y_min - y_max) / (image_values_flipped.shape[0] - 1),
                colorbar=cbar,
                showscale=legend,
                hoverinfo='text',
                text=hovertext,
                zmin=min(all_ids) if (colors and len(colors) > 0) else 0,
                zmax=zmax_value if (colors and len(colors) > 0) else 1,
            )
        )

        plot = go.Figure(data=data)
        
        plotly_build_time = time.time() - plotly_build_start
        print(f"  → Plotly figure creation took {plotly_build_time:.3f}s")

        plot.update_layout(
            xaxis={"title": xlabel, "showgrid": False, "zeroline": False},
            yaxis={"title": ylabel, "showgrid": False, "zeroline": False},
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

        total_plot_time = time.time() - plot_start
        print(f"✓ plot_plotly() completed in {total_plot_time:.3f}s")
        return plot


# patching openmc

openmc.Geometry.get_dagmc_universe = get_dagmc_universe
openmc.Geometry.is_geometry_dagmc = is_geometry_dagmc
openmc.Geometry.get_dagmc_filepath = get_dagmc_filepath
openmc.Geometry.get_plot_extent = get_plot_extent
openmc.Geometry.get_side_extent = get_side_extent
openmc.Geometry.get_mid_slice_value = get_mid_slice_value
openmc.Geometry.get_axis_labels = get_axis_labels
openmc.Geometry.find_cell_id = find_cell_id


openmc.DAGMCUniverse.get_all_universes = openmc.Universe.get_all_universes
