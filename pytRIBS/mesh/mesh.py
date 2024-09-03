import os
import shutil

import pandas as pd
from rasterio.windows import from_bounds
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from scipy.spatial import distance

from shapely.vectorized import contains
from shapely.ops import nearest_points

from math import ceil
from rasterio.mask import mask
import numpy as np
import geopandas as gpd
from rasterio.features import shapes
from whitebox import WhiteboxTools
from shapely.geometry import Point
import rasterio
import pywt
import pyvista as pv
from shapely.ops import unary_union

from pytRIBS.shared.inout import InOut

from pytRIBS.shared.shared_mixin import Meta, SharedMixin
from pytRIBS.mesh.run_docker import MeshBuilderDocker


class Preprocess:
    """
     A class for preprocessing digital elevation models (DEMs) and extracting watershed and stream network data.

     This class provides methods for preparing DEMs, setting up metadata, and processing elevation data to
     extract watershed and stream network features. It uses the WhiteboxTools library for various geospatial
     operations and supports setting verbose mode for logging.

     :param outlet: A tuple containing the outlet coordinates used in the processing.
     :type outlet: tuple (float, float)

     :param snap_distance: The distance used for snapping outlet points to the nearest flow path.
     :type snap_distance: float

     :param threshold_area: The area threshold used to identify streams in the flow accumulation raster.
     :type threshold_area: float

     :param dem_path: The file path to the digital elevation model (DEM) to be processed.
     :type dem_path: str

     :param verbose_mode: Flag to enable verbose mode for WhiteboxTools, which controls the amount of log output.
     :type verbose_mode: bool

     :param meta: A dictionary containing metadata for the project, including 'Name', 'EPSG', and 'Scenario'.
     :type meta: dict

     :param dir_proccesed: Optional directory path for saving processed files. If not provided, defaults to 'preprocessing'.
     :type dir_proccesed: str, optional

     :raises OSError: If there is an issue creating directories or accessing files.
     :raises ValueError: If the EPSG code of the DEM does not match the project metadata and cannot be reconciled.

     Attributes:
     -----------
     outlet : tuple (float, float)
         The outlet coordinates used in the processing.
     snap_distance : float
         The distance used for snapping outlet points to the nearest flow path.
     threshold_area : float
         The area threshold used to identify streams.
     dem_preprocessing : str
         The absolute path to the digital elevation model (DEM) file.
     output_dir : str
         The directory where processed files are saved.
     meta : dict
         Metadata for the project, including 'Name', 'EPSG', and 'Scenario'.
     wbt : WhiteboxTools
         Instance of WhiteboxTools for performing geospatial operations.
     """
    def __init__(self, outlet, snap_distance, threshold_area, dem_path, verbose_mode, meta, dir_proccesed=None):

        self.outlet = outlet
        self.snap_distance = snap_distance
        self.threshold_area = threshold_area

        Meta.__init__(self)

        self.meta['Name'] = meta['Name']
        self.meta['EPSG'] = meta['EPSG']
        self.meta['Scenario'] = meta['Scenario']

        self.wbt = WhiteboxTools()
        self.wbt.set_verbose_mode(verbose_mode)

        name = self.meta['Name']

        if name is None:
            name = 'Basin'
            self.meta['Name'] = name

        if dir_proccesed is None:
            dir_proccesed = 'preprocessing'
            os.makedirs(dir_proccesed, exist_ok=True)

        if dem_path is not None:
            with rasterio.open(dem_path) as src:
                crs = src.crs
                epsg = self.meta['EPSG']
                if epsg != crs.to_epsg():
                    print(f'EPSG code for DEM { crs.to_epsg()}  does not match with project meta data {epsg}.\n'
                          'Converting project EPSG to DEM EPSG')
                    self.meta['EPSG'] = crs.to_epsg()

        self.dem_preprocessing = os.path.abspath(dem_path)
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
        gdf.set_crs(epsg=self.meta["EPSG"],inplace=True)
        gdf.to_file(output_path)

        return gdf, output_path

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

        name = self.meta['Name']
        ws_bound = gpd.read_file(watershed_boundary)
        geometry = [unary_union(ws_bound.geometry)]
        bounds = ws_bound.total_bounds  # [minx, miny, maxx, maxy]

        for raster_file in raster_list:

            with rasterio.open(raster_file) as src:
                if method == 'boundary' or method == 'both':
                    output_raster = f'{output_dir}/{name}_clipped.tif'

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
                    output_raster = f'{output_dir}/{name}_clipped_ext.tif'

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
        lines.set_crs(epsg=self.meta["EPSG"],inplace=True)
        watershed_boundary_t = gpd.read_file(watershed_boundary)
        clipped_lines = gpd.clip(lines, watershed_boundary_t)
        clipped_lines.set_crs(epsg=self.meta["EPSG"],inplace=True)
        clipped_lines.to_file(output_path)

        return output_path

    def extract_watershed_and_stream_network(self, outlet_path, boundary_path, output_streams_path, clean=True):
        """
        Extracts the watershed and stream network from elevation data and outputs the results to specified paths.

        This function performs a series of operations to process elevation data and extract both the watershed
        and stream network. It includes filling depressions, generating flow direction and accumulation rasters,
        identifying streams, and creating and clipping the watershed boundary and stream network. The results
        are saved to the paths provided. Optionally, temporary files and directories are cleaned up after the
        operation if specified.

        :param outlet_path: The path where the outlet raster will be saved.
        :type outlet_path: str

        :param boundary_path: The path where the watershed boundary shapefile will be saved.
        :type boundary_path: str

        :param output_streams_path: The path where the clipped stream network shapefile will be saved.
        :type output_streams_path: str

        :param clean: If True, temporary files and directories will be cleaned up after processing. Defaults to True.
        :type clean: bool, optional

        :returns: None

        :raises OSError: If there is an issue creating directories or accessing files.
        :raises ValueError: If the input parameters are not correctly specified.

        """
        output_dir = self.output_dir

        if clean is True:
            temp = self.output_dir + '/temp'
            os.makedirs(temp, exist_ok=True)
            self.output_dir = temp

        filled = self.fill_depressions()
        d8_raster = self.generate_flow_direction_raster(filled)
        flow_acc = self.generate_flow_accumulation_raster(d8_raster)
        streams = self.generate_streams_raster(flow_acc, self.threshold_area)
        outlet = self.create_outlet(self.outlet[0], self.outlet[1], flow_acc, self.snap_distance,
                                    output_path=f'{outlet_path}')
        ws_mask = self.generate_watershed_mask(d8_raster, outlet)
        _, ws_bound = self.generate_watershed_boundary(ws_mask, output_path=f'{boundary_path}')
        stream_shp = self.convert_stream_raster_to_vector(streams, d8_raster)
        self.clip_rasters([filled], ws_bound, output_dir=output_dir, method='both')
        self.clip_streamline(os.path.abspath(stream_shp), ws_bound, output_path=f'{output_streams_path}')

        if clean is True:
            shutil.rmtree(temp)


class GenerateMesh:
    """
    A class for generating a locally refined triangular irregular network (TIN) mesh from raster data, watershed boundaries, and stream networks.

    This class performs the following operations:
    - Extracts raster and wavelet information from the provided raster file.
    - Loads watershed boundaries, stream networks, and outlet points from the specified files.
    - Utilizes wavelet decomposition to refine the mesh and incorporate significant details.
    - Generates a TIN mesh incorporating elevations and boundary conditions from the input data.

    :param path_to_raster: Path to the raster file used for extracting elevation data and wavelet analysis.
    :type path_to_raster: str

    :param path_to_watershed: Path to the shapefile containing watershed boundary data.
    :type path_to_watershed: str

    :param path_to_stream_network: Path to the shapefile containing stream network data.
    :type path_to_stream_network: str

    :param path_to_outlet: Path to the shapefile containing outlet point data.
    :type path_to_outlet: str

    :param maxlevel: Maximum level for wavelet decomposition. If None, the maximum level is determined from the data.
    :type maxlevel: int, optional

    Attributes:
    - `normalizing_coeff`: Coefficient used for normalizing wavelet coefficients (initially None).
    - `raster`: Path to the raster file.
    - `maxlevel`: Maximum level for wavelet decomposition.
    - `watershed`: GeoDataFrame containing watershed boundaries.
    - `stream_network`: GeoDataFrame containing stream network data.
    - `outlet`: GeoDataFrame containing outlet points.

    Methods:
    - `extract_raster_and_wavelet_info`: Extracts and processes raster data and wavelet information.
    - `find_max_average_coeffs`: Computes the maximum average coefficients from wavelet data.
    - `extract_points_from_significant_details`: Extracts significant detail points from wavelet data.
    - `convert_coords_to_mesh`: Converts coordinate data into a TIN mesh.
    - `interpolate_elevations`: Interpolates elevation values at specified points.
    - `partition_mesh`: Partitions the mesh into regions suitable for parallel processing (external method, not directly part of this class).

    """
    def __init__(self, path_to_raster, path_to_watershed, path_to_stream_network, path_to_outlet, maxlevel=None):
        self.normalizing_coeff = None
        self.raster = path_to_raster
        self.maxlevel = maxlevel
        self.extract_raster_and_wavelet_info()
        self.watershed = gpd.read_file(path_to_watershed)
        self.stream_network = gpd.read_file(path_to_stream_network)
        self.outlet = gpd.read_file(path_to_outlet)

    def extract_raster_and_wavelet_info(self):
        """
        Extracts information from a raster file and performs wavelet decomposition on the raster data.
        """
        with rasterio.open(self.raster) as src:
            self.data = src.read(1)  # Read the first band
            self.transform = src.transform
            self.bounds = src.bounds
            self.width = src.width
            self.height = src.height

        self.wavelet_packet = pywt.WaveletPacket2D(data=self.data, wavelet='db1',
                                                   maxlevel=self.maxlevel)
        # update maxlevel incase it's none
        self.maxlevel = self.wavelet_packet.maxlevel
        self.normalizing_coeff = self.find_max_maximum_coeffs()

    def get_extent(self):
        """
        Returns extent (xmin, xmax, ymin, ymax) of dem data used in mesh generation.
        """

        x_min = self.bounds.left
        x_max = self.bounds.right
        y_min = self.bounds.bottom
        y_max = self.bounds.top

        return x_min, x_max, y_min, y_max

    def find_max_average_coeffs(self):
        """
        Calculates the maximum average coefficient from the wavelet packet decomposition.

        This method computes the average of the absolute values of the vertical, horizontal, and diagonal
        coefficients at each level of the wavelet decomposition. It then finds and returns the maximum of these
        average coefficients across all levels.

        The method uses the `self.wavelet_packet` attribute, which should be an instance of `pywt.WaveletPacket2D`
        with decomposed data, and `self.maxlevel`, which defines the maximum level of decomposition.

        :returns: The maximum average coefficient across all levels of the wavelet decomposition.
        :rtype: float

        :raises AttributeError: If `self.wavelet_packet` or `self.maxlevel` is not properly set.
        :raises ValueError: If there are issues with the wavelet coefficients or their dimensions.
        """

        max_avg_coeffs = []
        for level in range(1, self.maxlevel + 1):
            v = self.wavelet_packet['v' * level].data
            h = self.wavelet_packet['h' * level].data
            d = self.wavelet_packet['d' * level].data
            avg_coeffs = np.mean(np.abs([v, h, d]), axis=0)
            max_avg_coeffs.append(np.max(avg_coeffs))
        return max(max_avg_coeffs)

    def find_max_maximum_coeffs(self):
        """
        Calculates the maximum of coefficient from the wavelet packet decomposition.

        This method computes the maximum of the absolute values of the vertical, horizontal, and diagonal
        coefficients at each level of the wavelet decomposition. It then finds and returns the maximum of these
        average coefficients across all levels.

        The method uses the `self.wavelet_packet` attribute, which should be an instance of `pywt.WaveletPacket2D`
        with decomposed data, and `self.maxlevel`, which defines the maximum level of decomposition.

        :returns: The maximum average coefficient across all levels of the wavelet decomposition.
        :rtype: float

        :raises AttributeError: If `self.wavelet_packet` or `self.maxlevel` is not properly set.
        :raises ValueError: If there are issues with the wavelet coefficients or their dimensions.
        """
        max_coeffs = []
        for level in range(1, self.maxlevel + 1):
            v = self.wavelet_packet['v' * level].data
            h = self.wavelet_packet['h' * level].data
            d = self.wavelet_packet['d' * level].data
            avg_coeffs = np.max(np.abs([v, h, d]), axis=0)
            max_coeffs.append(np.max(avg_coeffs))
        return max(max_coeffs)

    def extract_points_from_significant_details(self, threshold):
        """
        Extracts significant points from wavelet decomposition details and processes them.

        This method identifies significant details from wavelet decomposition at various levels, filters and
        removes closely spaced points, and generates additional points along stream paths. It then combines
        these points with boundary codes and interpolated elevations, and returns the final set of points along
        with a buffered watershed geometry.

        :param threshold: The minimum threshold for identifying significant details in the wavelet decomposition.
        :type threshold: float

        :returns: A tuple containing:
            - An array of points with their elevations and boundary codes. The array has columns for the x and y
              coordinates, elevations, and boundary codes.
            - A buffered watershed geometry, which may be used for further spatial analysis.
        :rtype: tuple (numpy.ndarray, geometry)

        :raises AttributeError: If `self.wavelet_packet`, `self.maxlevel`, or other required attributes are not properly set.
        :raises ValueError: If there are issues with the processing or filtering of points.
        """

        centers = set()
        for level in range(1, self.maxlevel + 1):
            centers.update(self._process_level(level, threshold))

        centers = np.array(list(centers))

        # remove any points closer to each other than the resolution of the DEM
        centers = self.remove_close_points(centers,self.transform[0]) # resolution of raster
        centers = np.array(list(centers))


        stream_points = self._generate_points_along_stream(centers)
        stream_code = np.ones(len(stream_points)) * 3

        med_distance, max_distance = self._distance_to_nearest_n(centers, n=6)

        centers, boundary_codes, buffered_watershed = self._filter_coords_within_geometry(centers, med_distance)

        centers = np.vstack((centers, stream_points))  # stream_points
        boundary_codes = np.hstack((boundary_codes, stream_code))  # stream_code

        unique_centers, unique_indices = np.unique(centers, axis=0, return_index=True)
        centers = unique_centers
        boundary_codes = boundary_codes[unique_indices]

        elevations = self.interpolate_elevations(centers)

        return np.column_stack((centers, elevations, boundary_codes)), buffered_watershed

    def _filter_coords_within_geometry(self, coords, buffer_distance):
        """
        Filters and categorizes coordinates based on their position relative to a buffered watershed geometry.

        Parameters
        ----------
        coords : numpy.ndarray
            An array of shape (N, 2) representing coordinates to be filtered.
        buffer_distance : float
            The distance to buffer the watershed geometry.

        Returns
        -------
        all_coords : numpy.ndarray
            An array of filtered coordinates that include original, boundary, and inner boundary points.
        all_boundary_codes : numpy.ndarray
            An array of boundary codes corresponding to the filtered coordinates. Codes indicate whether a coordinate
            is part of the original watershed (0), the outer boundary (1), or an updated outlet point (2).
        buffered_watershed : shapely.geometry.Polygon
            The buffered version of the original watershed geometry used for filtering.

        Notes
        -----
        The function processes the original watershed geometry by applying a specified buffer distance and generates boundary points.
        It finds coordinates within the original watershed and boundary points outside of it, adjusts coordinates based on proximity
        to the updated outlet, and categorizes coordinates with boundary codes.

        Additional Information
        ----------------------
        - Coordinates within the original watershed are assigned a boundary code of 0.
        - Coordinates on the outer boundary are assigned a boundary code of 1.
        - The outlet is adjusted to the nearest boundary point and given a boundary code of 2.
        """

        def find_nearest_boundary_point(point, polygon):
            # If the point is inside the polygon
            if polygon.contains(point):
                # Compute the nearest point on the boundary
                boundary_point = polygon.exterior.interpolate(polygon.exterior.project(point))
                return boundary_point
            else:
                # If the point is outside, find the nearest point on the polygon
                nearest = nearest_points(point, polygon)
                return nearest[1]

        original_watershed = self.watershed.geometry.iloc[0]

        inner_buffered_watershed = original_watershed.buffer(1e-6)

        buffered_watershed = original_watershed.buffer(buffer_distance)

        num_inner_boundary_points = int(ceil(inner_buffered_watershed.length / buffer_distance))

        num_boundary_points = int(ceil(buffered_watershed.length / buffer_distance))

        inner_boundary_coords = np.array(list(
            inner_buffered_watershed.exterior.interpolate(i / num_inner_boundary_points, normalized=True).coords[0] for
            i in
            range(num_inner_boundary_points)))

        inner_boundary_codes = np.zeros(inner_boundary_coords.shape[0], dtype=int)

        boundary_coords = np.array(list(
            buffered_watershed.exterior.interpolate(i / num_boundary_points, normalized=True).coords[0] for i in
            range(num_boundary_points)))

        boundary_codes = np.ones(boundary_coords.shape[0], dtype=int)

        within_original = contains(original_watershed, coords[:, 0], coords[:, 1])

        # because the buffered watershed is altered you can have closed nodes within the watershed
        bcoords_within_orignal = contains(original_watershed, boundary_coords[:, 0], boundary_coords[:, 1])

        boundary_coords = boundary_coords[~bcoords_within_orignal, :]
        boundary_codes = boundary_codes[~bcoords_within_orignal]

        original_outlet = self.outlet.geometry.iloc[0]

        # Find the nearest point on the polygon boundary
        updated_outlet = find_nearest_boundary_point(original_outlet, buffered_watershed)

        x = updated_outlet.x
        y = updated_outlet.y

        out_points = [[x, y]]
        out_code = 2

        buffer_outlet = updated_outlet.buffer(buffer_distance)
        bcoords_within_outlet = contains(buffer_outlet, boundary_coords[:, 0], boundary_coords[:, 1])
        inner_bcoords_within_outlet = contains(buffer_outlet, inner_boundary_coords[:, 0], inner_boundary_coords[:, 1])

        all_coords = np.vstack([coords[within_original, :], boundary_coords[~bcoords_within_outlet],
                                inner_boundary_coords[~inner_bcoords_within_outlet], out_points])
        all_boundary_codes = np.hstack(
            [np.zeros(np.sum(within_original), dtype=int), boundary_codes[~bcoords_within_outlet],
             inner_boundary_codes[~inner_bcoords_within_outlet], out_code])

        return all_coords, all_boundary_codes, buffered_watershed

    def _process_level(self, level, threshold):
        """
        Processes a specific level of wavelet decomposition to extract significant detail points.

        This method processes the vertical, horizontal, and diagonal coefficients from the wavelet packet decomposition
        at a given level. It calculates normalization coefficients, creates a significance mask based on the provided
        threshold, and converts the mask to geographic coordinates.

        :param level: The level of wavelet decomposition to process.
        :type level: int
        :param threshold: The threshold value used to filter significant coefficients.
        :type threshold: float

        :returns: An iterator of tuples containing the x and y coordinates of significant points.
        :rtype: iterator of tuples (float, float)

        :raises KeyError: If the specified level is not present in the wavelet packet.
        :raises AttributeError: If `self.wavelet_packet`, `self.bounds`, or `self.normalizing_coeff` are not properly set.
        :raises ValueError: If there are issues with coefficient shapes or calculations.
        """

        v = self.wavelet_packet['v' * level].data
        h = self.wavelet_packet['h' * level].data
        d = self.wavelet_packet['d' * level].data

        r, c = v.shape
        dx = (self.bounds.right - self.bounds.left) / c
        dy = (self.bounds.top - self.bounds.bottom) / r

        norm_coeffs = np.maximum.reduce([np.abs(v), np.abs(h), np.abs(d)]) / self.normalizing_coeff
        sig_mask = norm_coeffs > (2 ** (-level + 1) * threshold)

        rows, cols = np.where(sig_mask)
        x_coords = self.bounds.left + cols * dx
        y_coords = self.bounds.top - rows * dy

        return zip(x_coords, y_coords)
    @staticmethod
    def remove_close_points(points, threshold):
        """
        Removes points that are closer than a specified threshold.
        """
        # Convert list of points to a numpy array
        points = np.array(points)

        # Compute pairwise distances
        dist_matrix = distance.cdist(points, points)

        # Create a boolean mask for distances below the threshold (excluding diagonal)
        mask = (dist_matrix < threshold) & (dist_matrix > 0)

        # Get indices of points to remove
        to_remove = set()
        for i in range(len(points)):
            if i not in to_remove:
                close_points = np.where(mask[i])[0]
                to_remove.update(close_points)

        # Remove points
        result = [tuple(points[i]) for i in range(len(points)) if i not in to_remove]

        return result

    @staticmethod
    def _distance_to_nearest_n(points, n=6):
        """
        Calculates the average distance to the nearest `n` points for each point in a list.

        This function uses a KD-tree to efficiently compute the distances between points and determine the
        distances to the `n` nearest neighbors for each point. It returns the median and maximum distances to
        these neighbors.

        :param points: A list of (x, y) coordinates representing the points to analyze.
        :type points: list of tuples
        :param n: The number of nearest neighbors to consider for each point. Defaults to 6.
        :type n: int

        :returns: A tuple containing:
            - The median distance to the nearest `n` points for each point.
            - The maximum distance to the nearest `n` points for each point.
        :rtype: tuple (float, float)

        :raises ValueError: If the input `points` is empty or if `n` is less than 1.
        :raises TypeError: If `points` is not a list of tuples or if `n` is not an integer.
        """
        points_array = np.array(points)
        tree = cKDTree(points_array)
        distances, _ = tree.query(points_array, k=n + 1)

        median_distance = np.max(np.median(distances[:, :], axis=0))
        max_distance = np.max(np.max(distances[:, :], axis=0))

        return median_distance, max_distance

    def convert_coords_to_mesh(self, coords):
        """
        Converts a set of coordinates into a 2D mesh using Delaunay triangulation.

        This method takes an array of coordinates and converts them into a 2D mesh. The coordinates include
        x, y, and z values, with an optional boundary code. The boundary codes are used to label points in the
        mesh, but currently, they do not affect the mesh creation.

        :param coords: An array where each row represents a point with x, y, z coordinates and a boundary code.
        :type coords: numpy.ndarray

        :returns: A 2D mesh generated from the input coordinates using Delaunay triangulation.
        :rtype: pyvista.PolyData

        :raises IndexError: If the input `coords` array does not have at least four columns.
        :raises ValueError: If the input `coords` array is empty or has incorrect dimensions.
        """

        points = coords[:, :3]  # First three columns are x, y, z
        boundary_codes = coords[:, 3]  # Fourth column is boundary code

        point_cloud = pv.PolyData(points)

        elevation = points[:, 2]
        # elevation[boundary_codes == 1] = np.nan  # Set elevation to NaN where boundary code is 1
        point_cloud['Elevation'] = elevation
        point_cloud['BoundaryCode'] = boundary_codes

        mesh = point_cloud.delaunay_2d()

        return mesh

    def interpolate_elevations(self, points):
        """
        Interpolates elevation values for a set of geographic coordinates.

        This method uses a regular grid interpolator to estimate elevation values at specified geographic
        coordinates based on a given elevation data grid. It performs linear interpolation and handles
        coordinates that fall outside the bounds of the grid by returning NaN.

        :param points: An array of points where each row contains x and y coordinates for which elevation
                       values are to be interpolated.
        :type points: numpy.ndarray

        :returns: An array of interpolated elevation values corresponding to the input coordinates.
        :rtype: numpy.ndarray

        :raises ValueError: If `points` does not have exactly two columns.
        :raises AttributeError: If `self.data` or `self.transform` is not properly set.
        """

        height, width = self.data.shape

        x = np.arange(width) * self.transform[0] + self.transform[2]
        y = np.arange(height) * self.transform[4] + self.transform[5]

        interpolator = RegularGridInterpolator((y, x), self.data, method='linear', bounds_error=False,
                                               fill_value=None)

        elevations = interpolator((points[:, 1], points[:, 0]))

        return elevations

    def _generate_points_along_stream(self, coords):
        """
        Generates points along a stream network ensuring they are sufficiently spaced from each other.

        This method computes points along the stream network by interpolating positions at regular intervals
        based on the resolution of the DEM. It ensures that these points are spaced sufficiently from each
        other by checking their distances from both the stream network and any existing interior points.

        :param coords: An array of coordinates representing the existing points in the interior.
        :type coords: numpy.ndarray

        :returns: A list of points along the stream network, ensuring they are not too close to existing
                  interior points.
        :rtype: list of lists
        """

        stream = self.stream_network
        dem_res = self.transform[0]

        # Initialize the interior KDTree
        interior_tree = cKDTree(coords)  # Assumes coords is a NumPy array

        final_stream_pts = []

        for idx, row in stream.iterrows():
            # Extract line and compute points
            line = row['geometry']
            length = line.length
            num_points = int(ceil(length / dem_res))
            xy = [line.interpolate(i * dem_res) for i in range(num_points + 1)]
            stream_xy = np.array([[p.x, p.y] for p in xy])

            # Setup KDTree for stream points
            stream_tree = cKDTree(stream_xy)

            # Initialize stream points
            begin, end = stream_xy[0, :], stream_xy[-1, :]
            working_stream_pts = [begin, end]
            temp_stream_pts = [begin]

            # Check and remove close interior points
            dis_check, idx = stream_tree.query(coords, k=1)
            close_interior_pts = np.where(dis_check <= dem_res)[0]

            if len(close_interior_pts) > 0:
                coords_list = coords.tolist()
                for i in sorted(close_interior_pts, reverse=True):
                    temp_stream_pts.append(coords_list.pop(i))
                coords = np.array(coords_list)
                interior_tree = cKDTree(coords)

            # Initial conditions for while loop
            distance = np.linalg.norm(working_stream_pts[0] - working_stream_pts[1])
            dis_interior, _ = interior_tree.query(working_stream_pts[0], k=1)
            dis_stream, idx = stream_tree.query(working_stream_pts[0], k=len(stream_xy))
            idx_last = 0

            # Iterate to find additional stream points
            while distance > dis_interior:
                ids = idx[dis_stream < dis_interior]
                if len(ids) == 0:
                    break
                id_next = ids[-1]

                if id_next < idx_last:
                    id_next = ids[ids > idx_last][-1]

                temp_stream_pts.append(stream_xy[id_next])
                working_stream_pts = [stream_xy[id_next], end]
                idx_last = id_next

                distance = np.linalg.norm(working_stream_pts[0] - working_stream_pts[1])
                dis_interior, _ = interior_tree.query(working_stream_pts[0], k=1)
                dis_stream, idx = stream_tree.query(working_stream_pts[0], k=len(stream_xy))

            temp_stream_pts.append(end)
            final_stream_pts.extend(temp_stream_pts)

        return final_stream_pts

    @staticmethod
    def convert_points_to_gdf(points):
        """Helper function for converting points generated from extract_points_from_significant_details to a geopandas
        data frame."""

        df = pd.DataFrame(points, columns=['x', 'y', 'elevation', 'bc'])
        df['geometry'] = df.apply(lambda row: Point(row['x'], row['y']), axis=1)
        gdf = gpd.GeoDataFrame(df, geometry='geometry')
        gdf = gdf.drop(columns=['x', 'y'])
        return gdf

    @staticmethod
    def write_point_file(gdf, output):
        InOut.write_point_file(gdf, output)

    @staticmethod
    def plot_mesh(mesh, scalar=None, **kwargs):
        plotter = SharedMixin.plot_mesh(mesh, scalar, **kwargs)
        return plotter

    @staticmethod
    def generate_meshbuild_input_file(filename, base_name, point_filename):
        " Helper function that generates the input file to run with the MeshBuilder app."
        with open(filename, 'w') as file:
            file.write("VELOCITYRATIO:\n")
            file.write(f"{str(1.2)}\n")
            file.write("BASEFLOW:\n")
            file.write(f"{str(0.2)}\n")
            file.write("VELOCITYCOEF:\n")
            file.write(f"{str(60)}\n")
            file.write("FLOWEXP:\n")
            file.write(f"{str(0.3)}\n")
            file.write("OUTFILENAME:\n")
            file.write(f"{base_name}\n")
            file.write("POINTFILENAME:\n")
            file.write(f"{point_filename}\n")

    @staticmethod
    def partition_mesh(volume, partition_args):
        """
        Partitions a mesh and produces a .reach file for parallel execution with tRIBS.

        This function handles the partitioning of a mesh by interacting with Docker to build and process
        the mesh based on the provided input files and partitioning parameters. It performs the following steps:
        - Changes the working directory to the specified volume containing the mesh files.
        - Initializes and runs a Docker container to perform the mesh partitioning.
        - Executes the mesh partitioning workflow with the provided arguments.
        - Cleans up the Docker container and restores the working directory.

        :param volume: Path to the directory containing the .in and .points files for the mesh.
        :type volume: str

        :param partition_args: A list of arguments for partitioning:
            - The name of the input file (str)
            - The number of nodes for partitioning (int)
            - The partition method (1-3) (int)
            - The basename for output files (str)
        :type partition_args: list

        :returns: None
        """
        current = os.getcwd()
        os.chdir(volume)

        meshbuild = MeshBuilderDocker(os.getcwd())
        meshbuild.start_docker_desktop()
        meshbuild.pull_image()
        meshbuild.run_container()
        meshbuild.execute_meshbuild_workflow(*partition_args)
        meshbuild.cleanup_container()
        meshbuild.clean_directory()

        os.chdir(current)
