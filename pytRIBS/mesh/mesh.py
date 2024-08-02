import os
import shutil
import sys
from rasterio.windows import from_bounds


from rasterio.mask import mask
import numpy as np
import geopandas as gpd
from rasterio.features import shapes
import matplotlib.pyplot as plt
from whitebox import WhiteboxTools
import matplotlib as mpl
from shapely.geometry import shape, LineString, Polygon, MultiLineString, Point
import rasterio
import pywt
import pyvista as pv
from osgeo import gdal, ogr, osr

from shapely.ops import unary_union
import time


class Configuration:
    """
    Here we define the base directory and base file name, and the parameter values

    version = Base name of the files that will be produced throughout
    base_dir = The directory where all the different subfolders for the processing parts are kept
    input_dem_name = filename of the DEM
    pour_point_x = X Coordinate of the Pour Point in WGS 1984
    pour_point_y = Y Coordinate of the Pour Point in WGS 1984
    stream_threshold = Determines the number of cells that must drain out of flow accumulation raster to define a stream
    str_tolerance = Defines the interval of node points for the Stream
    wtrbdry_tolerance = Defines the interval of node points for the watershed boundary
    zres = The elevation tolerance that is defined when determining the density of internal nodes
    min_slope_value = Determines the minimum slope value
    dem_resample_res = the extent of resampling of the provided DEM
    """

    def __init__(self, version, base_dir, input_dem_name,
                 pour_point_x, pour_point_y, stream_threshold,
                 str_tolerance, wtrbdry_tolerance, zres, min_slope_value, dem_resample_res):
        self.version = version
        self.base_dir = base_dir
        self.input_dem_name = input_dem_name
        self.input_file_directory = self.set_input_file_directory(base_dir)

        # Additional parameters
        self.pour_point_x = pour_point_x
        self.pour_point_y = pour_point_y
        self.stream_threshold = stream_threshold
        self.str_tolerance = str_tolerance
        self.wtrbdry_tolerance = wtrbdry_tolerance
        self.zres = zres
        self.min_slope_value = min_slope_value
        self.dem_resample_res = dem_resample_res

    def set_input_file_directory(self, base_dir):
        return base_dir

    def get_input_file_path(self):
        return os.path.join(self.input_file_directory, self.input_dem_name)

    def display_config(self):
        print(f"Version: {self.version}")
        print(f"Base Directory: {self.base_dir}")
        print(f"Input DEM Name: {self.input_dem_name}")
        print(f"Input File Directory: {self.input_file_directory}")
        print(f"Pour Point X: {self.pour_point_x}")
        print(f"Pour Point Y: {self.pour_point_y}")
        print(f"Stream Threshold: {self.stream_threshold}")
        print(f"Stream Tolerance: {self.str_tolerance}")
        print(f"Watershed Boundary Tolerance: {self.wtrbdry_tolerance}")
        print(f"Z-Resolution: {self.zres}")
        print(f"Minimum Slope Value: {self.min_slope_value}")
        print(f"Full Input File Path: {self.get_input_file_path()}")


class _Mesh:
    def __init__(self, name, dem_path, verbose_mode, dir_proccesed):
        self.wbt = WhiteboxTools()
        self.wbt.set_verbose_mode(verbose_mode)

        if name is None:
            name = 'Basin'

        self.meta['Name'] = name

        if dir_proccesed is None:
            dir_proccesed = 'mesh_preprocessing'
            os.makedirs(dir_proccesed, exist_ok=True)

        if dem_path is not None:
            with rasterio.open(dem_path) as src:
                crs = src.crs
                self.meta['EPSG'] = crs.to_epsg()

        self.dem_preprocessing = dem_path
        self.output_dir = dir_proccesed

    def fill_depressions(self, output_path=None):
        """
        Fill Sinks within the watershed and plot the DEM afterwards
        """
        wbt = self.wbt
        name = self.meta['Name']

        if output_path is None:
            output_path = f'{self.output_dir}/{name}_filled.tif'

        wbt.fill_depressions(self.dem_preprocessing, os.path.abspath(output_path), fix_flats=True)

        return output_path

    def create_outlet(self, x, y, flow_accumulation_raster, snap_distance, output_path=None):
        """
        Uses pour point coordinates to create shapefile of the pour point and plot it accordingly
        """
        pour_point_geometry = Point(x, y)
        pour_point_gdf = gpd.GeoDataFrame(geometry=[pour_point_geometry])
        pour_point_gdf.set_crs(epsg=self.meta['EPSG'], inplace=True)
        name = self.meta['Name']

        if output_path is None:
            output_path = f'{self.output_dir}/{name}_outlet.shp'

        pour_point_gdf.to_file(output_path)

        wbt = self.wbt
        wbt.snap_pour_points(os.path.abspath(output_path), os.path.abspath(flow_accumulation_raster),
                             os.path.abspath(output_path), snap_distance)

        return output_path

    def generate_flow_direction_raster(self, filled_dem, output_path=None):
        """"
        Creates the flow direction raster based on the D8 method and plot it
        """
        wbt = self.wbt
        name = self.meta['Name']

        if output_path is None:
            output_path = f'{self.output_dir}/{name}_d8.tif'

        wbt.d8_pointer(os.path.abspath(filled_dem), os.path.abspath(output_path), esri_pntr=True)

        return output_path

    def generate_flow_accumulation_raster(self, flow_direction_raster, output_path=None):
        """
        Create the flow accumulation raster using the flow direction raster obtained from D8 method
        """
        wbt = self.wbt
        name = self.meta['Name']

        if output_path is None:
            output_path = f'{self.output_dir}/{name}_flow_acc.tif'

        wbt.d8_flow_accumulation(os.path.abspath(flow_direction_raster), os.path.abspath(output_path),
                                 pntr=True, esri_pntr=True)

        return output_path

    def generate_streams_raster(self, flow_accumulation_raster, threshold_area, output_path=None):
        """
        Obtain the stream raster from the flow accumulation raster based on the stream threshold provided before
        """
        wbt = self.wbt
        name = self.meta['Name']

        if output_path is None:
            output_path = f'{self.output_dir}/{name}_stream.tif'

        wbt.extract_streams(
            os.path.abspath(flow_accumulation_raster),
            os.path.abspath(output_path),
            threshold_area
        )

        return output_path

    def generate_watershed_mask(self, flow_direction_raster, pour_point_shp, output_path=None):
        """
        Use the pour point shapefile and the flow direction raster to delineate the watershed in the form of a raster
        """
        wbt = self.wbt
        name = self.meta['Name']

        if output_path is None:
            output_path = f'{self.output_dir}/{name}_watershed_msk.tif'

        wbt.watershed(os.path.abspath(flow_direction_raster), os.path.abspath(pour_point_shp),
                      os.path.abspath(output_path), esri_pntr=True)

        return output_path

    def generate_watershed_boundary(self, watershed_mask, output_path=None):
        """
        Develop the watershed boundary shapefile from the watershed raster
        """
        name = self.meta['Name']

        if output_path is None:
            output_path = f'{self.output_dir}/{name}_watershed_bound.tif'

        with rasterio.open(watershed_mask) as src:
            image = src.read(1)  # Read the first band
            mask = image != src.nodata  # Create a mask for valid data values

            results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v) in enumerate(
                shapes(image, mask=mask, transform=src.transform))
            )

            # Convert the results to a GeoDataFrame
            geoms = list(results)
            gdf = gpd.GeoDataFrame.from_features(geoms)
            gdf = gdf.dissolve()

        gdf.to_file(output_path)

        return gdf, output_path

    # def compute_stream_order(self, flow_direction_raster, stream_raster, output_path):
    #
    #     """
    #     Obtain the Strahler's Stream Order from the stream raster and flow direction raster
    #     """
    #     wbt = self.wbt
    #     wbt.strahler_stream_order(
    #         flow_direction_raster,
    #         stream_raster,
    #         output_path,
    #         esri_pntr=True
    #     )

    def convert_stream_raster_to_vector(self, stream_raster, flow_direction_raster, output_path=None):
        """
        Create a shapefile of the stream network from the stream raster
        """
        wbt = self.wbt
        name = self.meta['Name']

        if output_path is None:
            output_path = f'{self.output_dir}/{name}_streams.shp'

        # Convert Stream Raster to Vector
        wbt.raster_streams_to_vector(
            os.path.abspath(stream_raster),
            os.path.abspath(flow_direction_raster),
            os.path.abspath(output_path),
            esri_pntr=True
        )

        return output_path

    def clip_rasters(self, raster_list, watershed_boundary, method='boundary', output_dir=None):
        """
        Using the watershed boundary polygon to clip the filled dem, flow direction raster, stream raster and stream order raster
        """

        if output_dir is None:
            output_dir = f'{self.output_dir}/'

        ws_bound = gpd.read_file(watershed_boundary)
        geometry = [unary_union(ws_bound.geometry)]
        bounds = ws_bound.total_bounds  # [minx, miny, maxx, maxy]

        for raster_file in raster_list:

            with rasterio.open(raster_file) as src:
                if method == 'boundary' or method == 'both':

                    output_filename = os.path.basename(raster_file).replace(".", "_clipped.")
                    output_raster = os.path.join(output_dir, output_filename)

                    clipped_data, clipped_transform = mask(src, geometry, crop=True)

                    clipped_profile = src.profile
                    clipped_profile.update({
                        'height': clipped_data.shape[1],
                        'width': clipped_data.shape[2],
                        'transform': clipped_transform
                    })

                    with rasterio.open(output_raster, 'w', **clipped_profile) as dst:
                        dst.write(clipped_data)

                if method == 'extent' or method == 'both':

                    output_filename = os.path.basename(raster_file).replace(".", "_clipped_ext.")
                    output_raster = os.path.join(output_dir, output_filename)

                    window = from_bounds(*bounds, transform=src.transform)

                    clipped_data = src.read(window=window)
                    clipped_transform = src.window_transform(window)
                    clipped_meta = src.meta.copy()

                    # Update the metadata with the new dimensions and transform
                    clipped_meta.update({
                        "driver": "GTiff",
                        "height": clipped_data.shape[1],
                        "width": clipped_data.shape[2],
                        "transform": clipped_transform
                    })

                    # Save the clipped raster to the output path
                    with rasterio.open(output_raster, "w", **clipped_meta) as dst:
                        dst.write(clipped_data)



    def clip_streamline(self, stream_shapefile, watershed_boundary, output_path=None):
        """
        Streamlines shapefile to be clipped by watershed boundary polygon
        """

        name = self.meta['Name']

        if output_path is None:
            output_path = f'{self.output_dir}/{name}_streams.shp'

        lines = gpd.read_file(stream_shapefile)
        watershed_boundary_t = gpd.read_file(watershed_boundary)
        clipped_lines = gpd.clip(lines, watershed_boundary_t)
        clipped_lines.to_file(output_path)

        return output_path

    def extract_watershed_and_stream_network(self, x, y, snap_tol, threshold_area, boundary_path, output_streams_path,
                                             clean=True):

        output_dir = self.output_dir

        if clean is True:
            temp = self.output_dir + '/temp'
            os.makedirs(temp, exist_ok=False)
            self.output_dir = temp

        filled = self.fill_depressions()
        d8_raster = self.generate_flow_direction_raster(filled)
        flow_acc = self.generate_flow_accumulation_raster(d8_raster)
        streams = self.generate_streams_raster(flow_acc, threshold_area)
        outlet = self.create_outlet(x, y, flow_acc, snap_tol)
        ws_mask = self.generate_watershed_mask(d8_raster, outlet)
        _, ws_bound = self.generate_watershed_boundary(ws_mask, output_path=f'{output_dir}/{boundary_path}')
        stream_shp = self.convert_stream_raster_to_vector(streams, d8_raster)
        self.clip_rasters([filled], ws_bound, output_dir=output_dir, method='both')
        self.clip_streamline(os.path.abspath(stream_shp), ws_bound, output_path=f'{output_dir}/{output_streams_path}')

        if clean is True:
            shutil.rmtree(temp)


    class WaveletTree:
        def __init__(self, path_to_raster, maxlevel):
            self.raster = path_to_raster
            self.extract_raster_and_wavelet_info(maxlevel)
            self.populate_levels()

        def extract_raster_and_wavelet_info(self, maxlevel):
            with rasterio.open(self.raster) as src:
                dem = src.read(1)  # Read the first band
                dem_affine = src.transform  # Affine transformation
                bounds = src.bounds

                left, bottom, right, top = bounds
                width = right - left
                height = top - bottom

                self.top = top
                self.bottom = bottom
                self.left = left
                self.right = right
                self.width = width
                self.height = height
                self.affine_transform = dem_affine
                self.upper_left = (dem_affine[2], dem_affine[5])
                self.dem_data = dem

                self.wavelet_packet = pywt.WaveletPacket2D(data=dem, wavelet='db1', mode='symmetric', maxlevel=maxlevel)
                self.maxlevel = self.wavelet_packet.maxlevel
                self.normalizing_coeff = self.find_max_average_coeffs()

        def transform_wavelet_grid(self, level):
            r, c = np.shape(self.wavelet_packet['a' * level].data)
            dx = self.width / c
            dy = self.height / r
            return dx, dy

        def populate_levels(self):
            levels = {}

            for level in range(1, self.maxlevel + 1):
                dx, dy = self.transform_wavelet_grid(level)

                r, c = np.shape(self.wavelet_packet['a' * level].data)

                x = [self.upper_left[0] + dx * i for i in range(c)]
                y = [self.upper_left[1] - dy * i for i in range(r)]

                qcells = np.array([[WaveCell(xi, yi, dx, dy) for xi in x] for yi in y])

                normalized_coeffs = self.compute_normalized_coefficients(level)

                levels.update({level: {'qcells': qcells, 'norm_detail_coeffs': normalized_coeffs}})

            self.levels = levels

        # WAVELET RELATED METHODS

        def find_max_average_coeffs(self):
            max_avg_coeffs_per_level = []

            for level in range(1, self.maxlevel + 1):
                # extract detail coeffs for given level
                v = self.wavelet_packet['v' * level].data.copy()
                h = self.wavelet_packet['h' * level].data.copy()
                d = self.wavelet_packet['d' * level].data.copy()

                r, c = np.shape(v)

                avg_coefs = [np.mean(np.abs([v[x, y], h[x, y], d[x, y]])) for y in range(0, c) for x in range(0, r)]
                max_avg_coeffs_per_level.append(max(avg_coefs))

            return max(max_avg_coeffs_per_level)

        def compute_normalized_coefficients(self, level):

            def normalize_detail_coefficients(h_ij, v_ij, d_ij):
                max_detail = np.max(np.abs([h_ij, v_ij, d_ij]))
                normalized_detail = max_detail / self.normalizing_coeff
                return normalized_detail

            v = self.wavelet_packet['v' * level].data.copy()
            h = self.wavelet_packet['h' * level].data.copy()
            d = self.wavelet_packet['d' * level].data.copy()

            r, c = np.shape(h)
            norm_coefs = [[normalize_detail_coefficients(h[x, y], v[x, y], d[x, y]) for y in range(0, c)]
                          for x in range(0, r)]

            return np.array(norm_coefs)

        def select_significant_detail_coefficients(self, error_threshold):

            masks = []
            thresholds = []
            lmax = self.maxlevel

            for level in range(1, lmax + 1):
                normalized_coeffs = self.levels[level]['norm_detail_coeffs']
                r, c = np.shape(normalized_coeffs)
                threshold = 2 ** ((lmax - (
                            level - 1)) - lmax) * error_threshold  # levels are inverted here, i.e. level 1 is max refinement level
                sig_mask = [[normalized_coeffs[x, y] > threshold for y in range(0, c)]
                            for x in range(0, r)]

                masks.append(np.array(sig_mask))
                thresholds.append(threshold)

            self.sig_coeffs_mask = masks
            self.thresholds = thresholds

        def refine_cell(self, qcell, level):

            if level == 1:
                return

            dx, dy = self.transform_wavelet_grid(level - 1)  # cell size for next finest level

            r = int(qcell.dx / dx)
            c = int(qcell.dy / dy)

            x = [qcell.ul_x + dx * i for i in range(r)]
            y = [qcell.ul_y - dy * i for i in range(c)]

            qcells = np.array([[WaveCell(xi, yi, dx, dy) for xi in x] for yi in y])

            return qcells

        def get_centroids(self, mask=True):
            if mask:
                centroids = [cell.polygon.centroid
                             for level in range(1, self.maxlevel + 1)
                             for cells, masks in zip(self.levels[level]['qcells'], self.sig_coeffs_mask[level - 1])
                             for cell, mask in zip(cells.flatten(), masks.flatten())
                             if mask]
            else:
                centroids = [cell.polygon.centroid
                             for level in range(1, self.maxlevel + 1)
                             for cells in self.levels[level]['qcells']
                             for cell in cells.flatten()]
            return centroids

        def get_elevation_values(self, centroids):
            elevations = []
            coords = [(centroid.x, centroid.y) for centroid in centroids]

            with rasterio.open(self.raster) as src:
                elevations = [val[0] for val in src.sample(coords)]

            return elevations

        def plot_significant_cells(self, level):

            masks = self.sig_coeffs_mask[level - 1].flatten()
            cells = self.levels[level]['qcells'].flatten()
            scalar = self.levels[level]['norm_detail_coeffs'].flatten()

            fig, ax = plt.subplots()

            img = ax.imshow(self.levels[level]['norm_detail_coeffs'],
                            extent=(self.left, self.right, self.bottom, self.top))
            ax.set_aspect('equal', adjustable='box')
            plt.colorbar(img, ax=ax)

            for cell, scale in zip(cells[masks], scalar[masks]):
                cell.plot_cell(ax=ax)

            plt.show()

        @staticmethod
        def generate_mesh(centroids, elevations, exaggeration):
            points = np.array([(p.x, p.y, z * exaggeration) for p, z in zip(centroids, elevations)])
            valid_points = points[~np.isnan(points[:, 2])]
            point_cloud = pv.PolyData(valid_points)
            mesh = point_cloud.delaunay_2d()
            return mesh

        @staticmethod
        def plot_mesh(mesh):
            plotter = pv.Plotter()
            plotter.add_mesh(mesh)
            plotter.show()

class WaveCell:
    def __init__(self, ul_x, ul_y, dx, dy):
        self.ul_x = ul_x
        self.ul_y = ul_y
        self.dx = dx
        self.dy = dy
        self.vertices = [
            (self.ul_x, self.ul_y),
            (self.ul_x + self.dx, self.ul_y),
            (self.ul_x + self.dx, self.ul_y - self.dy),
            (self.ul_x, self.ul_y - self.dy)
        ]
        self.polygon = Polygon(self.vertices)

    def __repr__(self):
        return f'WaveCell(ul_x={self.ul_x}, ul_y={self.ul_y}, dx={self.dx}, dy={self.dy}, polygon={self.polygon})'

    def plot_cell(self, ax=None, centroid=False, edge_color=None, face_color=None, scalar=None, scalar_vector=None,
                  cmap='viridis'):
        if centroid:
            centroid = self.polygon.centroid
            x, y = centroid.x, centroid.y
        else:
            x, y = self.polygon.exterior.xy

        return_flag = False

        if ax is None:
            fig, ax = plt.subplots()
            return_flag = True

        if edge_color is None:
            edge_color = 'blue'

        # Determine face color based on scalar value if provided
        if scalar is not None:
            # Normalize scalar value to [0, 1] range based on colormap
            norm_scalar = (scalar - np.min(scalar_vector)) / (np.max(scalar_vector) - np.min(scalar_vector))
            cmap = mpl.colormaps[cmap]
            face_color = cmap(norm_scalar)

        if centroid:
            ax.scatter(x, y, color=edge_color)
        else:
            ax.plot(x, y, color=edge_color)
            ax.fill(x, y, alpha=0.5, fc=face_color, ec=edge_color)

        if return_flag:
            return fig, ax

class tin_analysis:

    def __init__(self, config):
        self.config = config

        # Create new directory for temporary files in the process leading to developing TIN
        tintempfiles = os.path.join(self.config.base_dir, "4_Tin_Temporary_fILES")
        os.makedirs(tintempfiles, exist_ok=True)

        self.uc_slope_raster = os.path.join(tintempfiles, f"{self.config.version}slp_uc.tif")
        self.uc_slope_raster_corrected = os.path.join(tintempfiles, f"{self.config.version}slpc_uc.tif")
        self.slope_raster = os.path.join(tintempfiles, f"{self.config.version}slp.tif")
        self.slope_raster_corrected = os.path.join(tintempfiles, f"{self.config.version}slpc.tif")
        self.outer_ring_boundary = os.path.join(self.config.base_dir, "4_Tin_Temporary_fILES",
                                                f"{self.config.version}oring.shp")
        self.generalized_watershed_boundary_polygon = os.path.join(self.config.base_dir, "4_Tin_Temporary_fILES",
                                                                   f"{self.config.version}bdngen.shp")
        self.inner_ring_buffer_output = os.path.join(self.config.base_dir, "4_Tin_Temporary_fILES",
                                                     f"{self.config.version}iring.shp")
        self.inner_ring_bdry = os.path.join(self.config.base_dir, "4_Tin_Temporary_fILES",
                                            f"{self.config.version}irgen.shp")
        self.dem_path = os.path.join(self.config.base_dir, "3_DEM_Analysis_Clipped", f"{self.config.version}fill_c.tif")
        self.resampled_dem = os.path.join(self.config.base_dir, "3_DEM_Analysis_Clipped",
                                          f"{self.config.version}fill_c_r.tif")
        self.pour_point_shapefile = os.path.join(self.config.base_dir, "2_DEM_ANALYSIS",
                                                 f"{self.config.version}pourpoint.shp")
        self.generalized_stream_line = os.path.join(self.config.base_dir, "4_Tin_Temporary_fILES",
                                                    f"{self.config.version}strgen.shp")

        # Define the CRS for WGS 1984 UTM Zone 12N
        self.crs = 'EPSG:32612'

        # Placeholder for significant points
        self.significant_points = None

        # Initialize GeoDataFrame attributes
        self.generalized_watershed_boundary_gdf = None
        self.simplified_inner_buffer_gdf = None
        self.simplified_stream_lines_gdf = None
        self.outer_buffer_gdf = None
        self.pour_point_gdf = None

        # Initialize DEMPreprocessor class
        self.dem_prep = DEMPreprocessor(self.config)

    def create_slope_raster(self):

        """"
        These are for the purpose of creating the various slope rasters which really have not been used thus far
        """

        # Check if output already exists and delete if it does
        if os.path.exists(self.uc_slope_raster):
            os.remove(self.uc_slope_raster)

        if os.path.exists(self.uc_slope_raster_corrected):
            os.remove(self.uc_slope_raster_corrected)

        if os.path.exists(self.slope_raster):
            os.remove(self.slope_raster)

        if os.path.exists(self.slope_raster_corrected):
            os.remove(self.slope_raster_corrected)

        # Compute slope using the finite difference method
        gdal.DEMProcessing(self.uc_slope_raster, self.dem_prep.filled_dem, 'slope', computeEdges=True)

        # Adjust slope for zero values using minimum
        with rasterio.open(self.uc_slope_raster) as src:
            slope_data = src.read(1)
            slope_data[slope_data <= 0] = self.config.min_slope_value

            # Create a new raster file with the corrected slope values
            profile = src.profile
            profile.update(dtype=rasterio.float64)
            with rasterio.open(self.uc_slope_raster_corrected, 'w', **profile) as dst:
                dst.write(slope_data, 1)

        # Clip slope_raster
        gdal.Warp(self.slope_raster, self.uc_slope_raster, format='GTiff',
                  cutlineDSName=self.dem_prep.watershed_boundary, cropToCutline=True)

        # Clip slope_raster_corrected
        gdal.Warp(self.slope_raster_corrected, self.uc_slope_raster_corrected, format='GTiff',
                  cutlineDSName=self.dem_prep.watershed_boundary, cropToCutline=True)

        print("Slope calculation complete.")

    def generate_generalized_shapes(self):

        """
        Produce shapefile of the generalized shapefiles of the watershed boundary and the streamline
        """

        # Setting output paths for the generalized watershed boundary and stream network shapefiles
        generalized_stream_line = os.path.join(self.config.base_dir, "4_Tin_Temporary_fILES",
                                               f"{self.config.version}strgen.shp")
        generalized_watershed_boundary_polygon = os.path.join(self.config.base_dir, "4_Tin_Temporary_fILES",
                                                              f"{self.config.version}bdngen.shp")

        # Remove existing output if they exist
        if os.path.exists(generalized_stream_line):
            os.remove(generalized_stream_line)

        if os.path.exists(generalized_watershed_boundary_polygon):
            os.remove(generalized_watershed_boundary_polygon)

        # Load the clipped stream line shapefile
        stream_line_clipped_gdf = gpd.read_file(self.dem_prep.stream_line_clipped)
        stream_lines = unary_union(stream_line_clipped_gdf.geometry)

        # Simplify the stream lines
        simplified_stream_lines = stream_lines.simplify(self.config.str_tolerance, preserve_topology=False)

        # Create a new GeoDataFrame with the simplified stream lines
        simplified_stream_lines_gdf = gpd.GeoDataFrame(geometry=[simplified_stream_lines], crs=self.crs)

        # Save the simplified stream lines to a new shapefile
        simplified_stream_lines_gdf.to_file(generalized_stream_line, driver='ESRI Shapefile')

        # Load the watershed boundary polygon shapefile
        watershed_boundary_polygon_gdf = gpd.read_file(self.dem_prep.watershed_boundary)
        boundary = unary_union(watershed_boundary_polygon_gdf.geometry)

        # Simplify the watershed boundary polygon
        simplified_boundary = boundary.simplify(self.config.wtrbdry_tolerance, preserve_topology=False)

        # Create a new GeoDataFrame with the simplified watershed boundary
        simplified_watershed_boundary_gdf = gpd.GeoDataFrame(geometry=[simplified_boundary], crs=self.crs)

        # Save the simplified watershed boundary to a new shapefile
        simplified_watershed_boundary_gdf.to_file(generalized_watershed_boundary_polygon, driver='ESRI Shapefile')

        print("Generalized shapes generation complete.")

    def generate_inner_ring_buffer(self):

        """
        Create an inner ring using a 30 meter buffer
        """

        # Remove existing output if they exist
        if os.path.exists(self.inner_ring_buffer_output):
            os.remove(self.inner_ring_buffer_output)

        if os.path.exists(self.inner_ring_bdry):
            os.remove(self.inner_ring_bdry)

        # Define the inner ring buffer distance
        iring_buffer_value = -30  # Negative value to create an inward buffer

        # Define the CRS for WGS 1984 UTM Zone 12N
        crs = 'EPSG:32612'

        # Load the generalized watershed boundary polygon
        generalized_watershed_boundary_gdf = gpd.read_file(
            os.path.join(self.config.base_dir, "4_Tin_Temporary_fILES", f"{self.config.version}bdngen.shp"))
        generalized_watershed_boundary_gdf = generalized_watershed_boundary_gdf.to_crs(self.crs)

        # Create the inner ring buffer
        inner_ring_buffer = generalized_watershed_boundary_gdf.geometry.buffer(iring_buffer_value, resolution=16)

        # Combine the inner buffer geometries into a single geometry
        inner_buffer_combined = unary_union(inner_ring_buffer)

        # Create a GeoDataFrame with the inner buffer geometry
        inner_buffer_gdf = gpd.GeoDataFrame(geometry=[inner_buffer_combined], crs=self.crs)

        # Save the inner buffer as a shapefile
        inner_buffer_gdf.to_file(self.inner_ring_buffer_output, driver='ESRI Shapefile')

        # Simplify the inner buffer geometry
        simplified_inner_buffer = inner_buffer_combined.simplify(self.config.wtrbdry_tolerance - 0.1,
                                                                 preserve_topology=False)

        # Create a GeoDataFrame with the simplified inner buffer geometry
        simplified_inner_buffer_gdf = gpd.GeoDataFrame(geometry=[simplified_inner_buffer], crs=crs)

        # Save the simplified inner buffer as a shapefile
        simplified_inner_buffer_gdf.to_file(self.inner_ring_bdry, driver='ESRI Shapefile')

        print("Inner ring buffer generation complete.")

        return simplified_inner_buffer_gdf

    def generate_outer_ring_buffer(self):

        """
        Create an outer ring boundary having a 1-m internal buffer
        """

        # Remove existing output if they exist
        if os.path.exists(self.outer_ring_boundary):
            os.remove(self.outer_ring_boundary)

        # Load the generalized watershed boundary polygon
        generalized_watershed_boundary_gdf = gpd.read_file(self.generalized_watershed_boundary_polygon)
        generalized_watershed_boundary_gdf = generalized_watershed_boundary_gdf.to_crs(self.crs)

        # Create the outer ring buffer
        outer_buffer = generalized_watershed_boundary_gdf.geometry.buffer(-1)  # Negative value for inward buffer

        # Combine the outer buffer geometries into a single geometry
        outer_buffer_combined = unary_union(outer_buffer)

        # Create a GeoDataFrame with the outer buffer geometry
        outer_buffer_gdf = gpd.GeoDataFrame(geometry=[outer_buffer_combined], crs=self.crs)

        # Save the outer buffer as a shapefile
        outer_buffer_gdf.to_file(self.outer_ring_boundary, driver='ESRI Shapefile')

    def resample_dem(self):

        """
        Resample the Clipped Filled DEM into a coarser resolution DEM
        """

        ds = gdal.Open(self.dem_path)
        dsRes = gdal.Warp(self.resampled_dem, ds, xRes=self.config.dem_resample_res, yRes=self.config.dem_resample_res,
                          resampleAlg="bilinear")

    def get_internal_nodes(self):

        """
        Use the drop heuristic method to get internal nodes for the watershed using the resampled DEM.

        This section is incomplete
        """

        # Read the DEM raster file using Rasterio
        with rasterio.open(self.resampled_dem) as src:
            dem_data = src.read(1)
            dem_transform = src.transform

        # Get the shape of the DEM raster
        rows, cols = dem_data.shape

        # Initialize lists to store coordinates and elevations
        x_coords = []
        y_coords = []
        z_values = []

        # Iterate over each cell in the DEM raster
        for i in range(rows):
            for j in range(cols):
                # Check if the elevation value is valid (not NoData)
                if dem_data[i, j] != src.nodata:
                    # Compute X and Y coordinates of the cell centerpoint
                    x, y = rasterio.transform.xy(dem_transform, i + 0.5, j + 0.5)

                    # Append X and Y coordinates to lists
                    x_coords.append(x)
                    y_coords.append(y)

                    # Get elevation value of the cell
                    z = dem_data[i, j]

                    # Append elevation value to list
                    z_values.append(z)

        # Convert lists to NumPy arrays
        x_coords = np.array(x_coords).reshape(-1, 1)
        y_coords = np.array(y_coords).reshape(-1, 1)
        z_values = np.array(z_values).reshape(-1, 1)

        # Verify the shape of the arrays
        print("Shape of X array:", x_coords.shape)
        print("Shape of Y array:", y_coords.shape)
        print("Shape of Z array:", z_values.shape)

        points = np.c_[x_coords, y_coords, z_values]

        pv.set_jupyter_backend('static')

        # Initialize the start time
        start_time = time.time()

        # to initialize the milestone counter
        total_iterations = len(points)
        print(len(points))
        current_iteration = 0

        # Initialize lists to store point information
        points_info = []

        # Initialize lists to store points with significant elevation differences
        significant_points = []

        # Define the progress milestones
        milestones = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        # Create the original triangulation with all points intact
        original_triangulation = pv.PolyData(points).delaunay_2d()

        # Loop through each point and perform simplification with exclusion
        for point_index in range(len(points)):
            # Create a temporary copy of the points array
            filtered_points = points.copy()

            # Remove the current point from the temporary array
            removed_point = filtered_points[point_index]  # Store the removed point
            filtered_points = np.delete(filtered_points, point_index, axis=0)

            # Extract X and Y coordinates from filtered points
            filtered_x = filtered_points[:, 0]
            filtered_y = filtered_points[:, 1]

            # Find the neighboring cells of the removed point

            z_x = original_triangulation.find_containing_cell(points[
                                                                  point_index])  # finds the cell ID of the cell containing the specific point using only level 1 neighbor
            neighbor_cell = original_triangulation.cell_neighbors(z_x, "points")

            # Get the points along the neighboring cells
            neighbor_points = []
            for cell_index in neighbor_cell:
                cell = original_triangulation.get_cell(cell_index)
                neighbor_points.extend(cell.points)

            # Remove duplicate points and the main point
            neighbor_points = np.array(list(set(map(tuple, neighbor_points))))
            neighbor_points = neighbor_points[~np.all(neighbor_points == removed_point, axis=1)]

            # Triangulate without the main point
            neighbor_polydata = pv.PolyData(neighbor_points)
            triangulation_without_main_point = neighbor_polydata.delaunay_2d()

            # Perform ray tracing to determine elevation
            bottom, top = triangulation_without_main_point.bounds[-2:]
            buffer = 1
            start = [removed_point[0], removed_point[1], bottom - buffer]
            stop = [removed_point[0], removed_point[1], top + buffer]
            points_X, _ = triangulation_without_main_point.ray_trace(start, stop)
            # Consider the first intersection point as the elevation
            elevation = points_X[0][2] if points_X.size > 0 else np.nan
            elevation_diff = elevation - removed_point[2]

            # Check if the elevation difference is significant
            if abs(elevation_diff) > self.config.zres:
                significant_points.append(points[point_index])

            # Create a dictionary to store point information
            point_info = {'x': removed_point[0], 'y': removed_point[1], 'z': removed_point[2], 'elevation_diff': np.nan}
            # Update point information
            point_info['elevation_diff'] = elevation_diff

            # Append point information to the list
            points_info.append(point_info)

            # Increase milestone counter
            current_iteration += 1

            # Calculate the progress percentage
            progress = (current_iteration / total_iterations) * 100

            # Check if we've reached a milestone
            for milestone in milestones:
                if progress >= milestone:
                    print(f"{milestone}% complete")
                    # Remove the milestone from the list to avoid duplicate messages
                    milestones.remove(milestone)

                    # print("Elevation Difference of Point", point_index, ":", elevation_diff)

        # Print completion message
        print("Completed processing all points!")

        # Convert points_info to a numpy array
        points_info_array = np.array([(p['x'], p['y'], p['z'], p['elevation_diff']) for p in points_info])
        # print(points_info_array)

        # At the end of the loop, triangulate the significant points
        self.significant_points = np.array(significant_points)
        significant_polydata = pv.PolyData(significant_points)
        triangulation_significant_points = significant_polydata.delaunay_2d()

        # Separate points with NaN elevation differences
        nan_points = points_info_array[np.isnan(points_info_array[:, 3])]
        numeric_points = points_info_array[~np.isnan(points_info_array[:, 3])]

        # Plot numeric points as green markers
        plt.scatter(numeric_points[:, 0], numeric_points[:, 1], color='green', label='Numeric', marker='o')

        # Plot NaN points as black markers
        plt.scatter(nan_points[:, 0], nan_points[:, 1], color='black', label='NaN', marker='o')

        # Plot the original triangulation
        original_triangulation.plot(color='cyan', cpos="xy", show_edges=True)

        # Plot the significant points
        triangulation_significant_points.plot(color='cyan', cpos="xy", show_edges=True)

        # Determine elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")

    def load_all_gdfs(self):
        self.pour_point_gdf = gpd.read_file(self.pour_point_shapefile)
        self.generalized_watershed_boundary_gdf = gpd.read_file(self.generalized_watershed_boundary_polygon)
        self.simplified_inner_buffer_gdf = gpd.read_file(self.inner_ring_bdry)
        self.simplified_stream_lines_gdf = gpd.read_file(self.generalized_stream_line)
        self.outer_buffer_gdf = gpd.read_file(self.outer_ring_boundary)

    def plot_nodes_and_triangulation(self):

        """
        Obtains the shapefile of the internal nodes, outer boundary notes, internal boundary nodes, pour point and the
        streamline and use their vertices to obtain nodes and plot them accordingly. Then create the triangulation using all the nodes
        """

        # Extract nodes from the outer ring geometry
        nodes_outer_ring = []
        for geom in self.outer_buffer_gdf.geometry:
            exterior_coords = geom.exterior.coords
            nodes_outer_ring.extend(exterior_coords)

        # Convert nodes to a numpy array
        nodes_array_outer_ring = np.array(nodes_outer_ring)

        # Extract nodes from the inner ring geometry
        nodes_inner_ring = []
        for geom in self.simplified_inner_buffer_gdf.geometry:
            exterior_coords = geom.exterior.coords
            nodes_inner_ring.extend(exterior_coords)

        # Convert nodes to a numpy array
        nodes_array_inner_ring = np.array(nodes_inner_ring)

        # Extract nodes from the line geometry
        nodes_line = []
        for geom in self.simplified_stream_lines_gdf.geometry:
            if isinstance(geom, MultiLineString):
                for line in geom.geoms:  # Use .geoms to access individual LineStrings
                    coords = list(line.coords)
                    nodes_line.extend(coords)
            elif isinstance(geom, LineString):
                coords = list(geom.coords)
                nodes_line.extend(coords)

        # Convert nodes to a numpy array
        nodes_array_line = np.array(nodes_line)

        #### Pour Point #######
        point_coords = self.pour_point_gdf.geometry.tolist()
        point_coords = np.asarray([point.x for point in self.pour_point_gdf.geometry])
        point_coords = np.column_stack((point_coords, [point.y for point in self.pour_point_gdf.geometry]))

        # Separate x and y coordinates (assuming point_coords from tolist())
        point_x = np.array([point[0] for point in point_coords])
        point_y = np.array([point[1] for point in point_coords])

        # Assuming point_coords is a list of lists (from tolist())
        point_array = np.array(point_coords)

        ###### TIN INTERNAL NODES ######

        # Assuming significant_points is a numpy array
        significant_points_x = self.significant_points[:, 0]  # Assuming x is in the first column
        significant_points_y = self.significant_points[:, 1]  # Assuming y is in the second column

        # Plot the nodes
        plt.figure(figsize=(10, 10))
        plt.plot(nodes_array_outer_ring[:, 0], nodes_array_outer_ring[:, 1], 'ro', label='Outer Boundary (bc = 1)',
                 markersize=1)
        plt.plot(nodes_array_inner_ring[:, 0], nodes_array_inner_ring[:, 1], 'bo', label='Inner Boundary (bc = 0)',
                 markersize=1)
        plt.plot(nodes_array_line[:, 0], nodes_array_line[:, 1], 'go', label='Streamlines', markersize=1)
        plt.plot(point_x, point_y, 'o', markersize=5, markerfacecolor='pink', markeredgewidth=1,
                 markeredgecolor='black', label='Pour Points')
        plt.plot(significant_points_x, significant_points_y, 'o', markersize=2, markerfacecolor='red',
                 markeredgewidth=1, markeredgecolor='black', label='Significant Points')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Nodes from Polygon and Line Shapefiles')
        plt.legend()
        plt.grid(True)  # Turn on the grid
        plt.show()

        significant_point_2d = np.c_[self.significant_points[:, 0], self.significant_points[:, 1]]
        print(significant_point_2d)

        # Concatenate all point arrays vertically
        all_points = np.vstack(
            (nodes_array_outer_ring, nodes_array_inner_ring, nodes_array_line, point_array, significant_point_2d))

        # Now 'all_points' will be a single array with all points stacked vertically
        print(all_points)  # This will print the shape of the combined array

        # Check if all_points has two columns and add a column of ones if necessary
        if all_points.shape[1] == 2:
            all_points = np.c_[all_points, np.ones((all_points.shape[0], 1))]

        print(all_points)  # Optional: Print the entire array

        # Create the original triangulation with all points intact
        merged_triangulation = pv.PolyData(all_points).delaunay_2d()

        # Plot the triangulation without the removed point
        merged_triangulation.plot(color='cyan', cpos="xy", show_edges=True)
