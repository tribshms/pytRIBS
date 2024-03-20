import os
import re

import numpy as np
import rasterio
from scipy.optimize import curve_fit
from scipy.ndimage import map_coordinates
from shapely.geometry import LineString, Point
from owslib.wcs import WebCoverageService
import geopandas as gpd
from rosetta import rosetta, SoilData

from pytRIBS.model.inout import InOut


### PREPROCESSING
# 1 delineate watersheds (pyshed)
# 1.create points file for hydrological conditioned tin Mesh
# 2 ingest soil and land use (pyshed)
# 3 apply pedotransfer functions
# others?

class Preprocess(InOut):
    """

    """

    # MESH TOOLS
    @staticmethod
    def remove_duplicates_points(gdf, radius):
        # Create a new column with a buffer around each point
        gdf['buffer'] = gdf['geometry'].buffer(radius)

        # Iterate through each row and check if there are other points within the buffer
        to_remove = set()
        for idx, row in gdf.iterrows():
            if idx not in to_remove:
                # Extract the buffer of the current row
                current_buffer = row['buffer']

                # Check for overlapping points within the buffer
                overlapping_points = gdf[gdf['geometry'].within(current_buffer) & (gdf.index != idx)]

                # Mark overlapping points for removal, excluding those with 'bc' equal to 2
                to_remove.update(overlapping_points[overlapping_points['bc'] != 2].index)

        # Drop unnecessary columns
        result = gdf.drop(to_remove, axis=0).drop('buffer', axis=1)

        return result

    @staticmethod
    def generate_points_along_stream(gdf, points_per_meter):
        """
        Generate evenly distributed points along each stream segment in a GeoDataFrame.

        Parameters:
        - gdf: GeoDataFrame
            GeoDataFrame containing the stream network with LineString geometries.
        - points_per_meter: int
            Number of points to generate per meter along each stream segment.

        Returns:
        GeoDataFrame
            GeoDataFrame containing the generated points.
        """
        # Create an empty GeoDataFrame to store the generated points
        points_list = []

        # Iterate over each stream segment in the GeoDataFrame
        for idx, row in gdf.iterrows():
            # Extract the LineString geometry for the current stream segment
            line = row['geometry']

            # Calculate the total length of the stream segment
            total_length = line.length

            # Calculate the total number of points based on the total length and points_per_meter
            total_points = int(total_length * points_per_meter)

            # Generate evenly distributed points along the stream segment
            points = [line.interpolate(i / total_points, normalized=True) for i in range(total_points + 1)]

            # Add the points to the list
            points_list.extend({'geometry': Point(p)} for p in points)

        # Create the GeoDataFrame from the list of dictionaries
        points_gdf = gpd.GeoDataFrame(points_list)

        return points_gdf

    @staticmethod
    def generate_buffer_points_along_stream(network, resolution=20, buffer_distance=25):
        # Create buffer geometries
        buffer_geoms = [line.buffer(buffer_distance) for line in network['geometry']]
        buffer_gdf = gpd.GeoDataFrame(geometry=buffer_geoms)
        combined_geometry = buffer_gdf.unary_union

        # Get bounds of the combined geometry
        latmin, lonmin, latmax, lonmax = combined_geometry.bounds

        # Generate points within the specified resolution
        points = []

        for lat in np.arange(latmin, latmax, resolution):
            for lon in np.arange(lonmin, lonmax, resolution):
                point = Point(round(lat, 4), round(lon, 4))

                # Calculate the distance to the nearest point on the stream network
                distance_to_stream = min(point.distance(line) for line in network['geometry'])

                # Adjust the point density based on distance (you can customize this function)
                weight = 1 / (1 + distance_to_stream)

                # Randomly discard points with a probability based on the weight
                if np.random.rand() < weight and combined_geometry.contains(point):
                    points.append(point)

        # Create a GeoDataFrame with empty columns 'bc' and 'elevation'
        columns = ['bc', 'elevation']
        valid_points_gdf = gpd.GeoDataFrame(geometry=points, columns=columns)

        return valid_points_gdf

    # def generate_buffer_points_along_stream(network, resolution=20, buffer_distance=25):
    #     # Create buffer geometries
    #     buffer_geoms = [line.buffer(buffer_distance) for line in network['geometry']]
    #     buffer_gdf = gpd.GeoDataFrame(geometry=buffer_geoms)
    #     combined_geometry = buffer_gdf.unary_union
    #
    #     # Get bounds of the combined geometry
    #     latmin, lonmin, latmax, lonmax = combined_geometry.bounds
    #
    #     # Generate points within the specified resolution
    #     points = [Point((round(lat, 4), round(lon, 4)))
    #               for lat in np.arange(latmin, latmax, resolution)
    #               for lon in np.arange(lonmin, lonmax, resolution)]
    #
    #     # Filter valid points inside the shape
    #     valid_points = [point for point in points if combined_geometry.contains(point)]
    #
    #     # Create a GeoDataFrame with empty columns 'bc' and 'elevation'
    #     columns = ['bc', 'elevation']
    #     valid_points_gdf = gpd.GeoDataFrame(geometry=valid_points, columns=columns)
    #
    #     return valid_points_gdf
    @staticmethod
    def remove_points_near_boundary(points_gdf, boundary_gdf, distance_threshold):
        """
        Removes points from the points file (points_gdf) that are closer than distance_threshold to the boundary of
        the watershed (boundary_gdf).

        Parameters:
        - points_gdf: GeoDataFrame containing points with a 'geometry' column.
        - boundary_gdf: GeoDataFrame containing the boundary with a 'geometry' column.
        - distance_threshold: Threshold distance for removing points near the boundary.

        Returns:
        - GeoDataFrame with points removed near the boundary.
        """

        # Ensure the GeoDataFrames have 'geometry' columns with appropriate geometries
        if 'geometry' not in points_gdf.columns or not isinstance(points_gdf['geometry'].iloc[0], Point):
            raise ValueError("points_gdf must have a 'geometry' column with Point geometries.")

        if 'geometry' not in boundary_gdf.columns:
            raise ValueError("boundary_gdf must have a 'geometry' column.")

        # Make a copy of the DataFrame
        points_gdf_copy = points_gdf.copy()

        # Calculate distances from points to the boundary
        dist = np.array([boundary_gdf.boundary.distance(points_gdf_copy.iloc[x].geometry).min() for x in
                         range(len(points_gdf_copy))])
        points_gdf_copy['dist'] = dist
        idx = (points_gdf_copy['dist'] < distance_threshold) & (
                (points_gdf_copy['bc'] == 0) | (points_gdf_copy['bc'] == 3))

        # Remove points within the distance threshold
        points_gdf_copy = points_gdf_copy[~idx]

        return points_gdf_copy

    @staticmethod
    def update_elevation_from_dem(gdf, dem_path, elevation_column='elevation'):
        """
        Update the elevation column in a GeoDataFrame using values from a Digital Elevation Model (DEM).

        Parameters:
        - gdf: GeoDataFrame
            GeoDataFrame containing points with coordinates.
        - dem_path: str
            Path to the Digital Elevation Model (DEM) raster file.
        - elevation_column: str, optional
            Name of the column to be updated with elevation values. Default is 'elevation'.

        Returns:
        GeoDataFrame
            GeoDataFrame with updated elevation values.
        """
        # Open the DEM raster using rasterio
        with rasterio.open(dem_path) as dem:
            # Loop through each point in the GeoDataFrame
            for index, point in gdf.iterrows():
                # Get the coordinates of the point
                lon, lat = point['geometry'].x, point['geometry'].y

                # Sample elevation from the DEM raster at the point coordinates
                for val in dem.sample([(lon, lat)]):
                    # Update the elevation column in the GeoDataFrame
                    gdf.at[index, elevation_column] = val[0]

        return gdf

    ### SOIL PRE-PROCESSING
    def getSoil250(self, bbox, depths, soil_vars, stats, fillnans=True):
        """
        Retrieves soil data from ISRIC WCS service and saves it as GeoTIFFs.

        Args:
            bbox (list): The bounding box coordinates in the format [x1, y1, x2, y2].
            depths (list): List of soil depths to retrieve data for.
            soil_vars (list): List of soil variables to retrieve.
            stats (list): List of statistics to compute for each variable and depth.

        Example:
            bbox = [387198, 3882394, 412385, 3901885]  # x1,y1,x2,y2
            depths = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm']
            soil_vars = ['bdod', 'clay', 'sand', 'silt', 'wv1500', 'wv0033', 'wv0010'], see https://maps.isric.org/
            stats = ['mean'], see prediciton quantiles at https://www.isric.org/explore/soilgrids/faq-soilgrids
        """
        epsg = self.geo['EPSG']

        if epsg is None:
            print("No EPSG code found. Please update model attribute .geo['EPSG'] with EPSG code.")
            return

        complete = False

        match = re.search(r'EPSG:(\d+)', epsg)
        if match:
            epsg = match.group(1)

        data_dir = 'SoilGrid_250_data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Directory '{data_dir}' created.")

        os.chdir(data_dir)

        crs = f'urn:ogc:def:crs:EPSG::{epsg}'

        files = []

        for var in soil_vars:
            wcs = WebCoverageService(f'http://maps.isric.org/mapserv?map=/map/{var}.map', version='1.0.0')
            for depth in depths:
                # make directory to store tiffs grouped by depth
                if not os.path.exists(depth):
                    os.makedirs(depth)
                    print(f"Directory '{depth}' created.")

                # for a give variable, depth, and stat write out a geotif
                for stat in stats:
                    soil_key = f'{var}_{depth}_{stat}'
                    response = wcs.getCoverage(identifier=soil_key, crs=crs, bbox=bbox, resx=250, resy=250,
                                               format='GEOTIFF_INT16',timeout=120)

                    file = f'{depth}/{soil_key}.tif'
                    files.append(file)

                    with open(file, 'wb') as file:
                        file.write(response.read())
                    complete = True
        if complete:
            print('Download of SoilGrids250 data complete')
        else:
            print('Error in downloading SoilGrids250 data, check inputs.')

        if fillnans:
            print("Filling NaNs!")
            for f in files:
                with rasterio.open(f) as src:
                    data = src.read(1)
                    filled_data = rasterio.fill.fillnodata(data, src.profile['nodata'], max_search_distance=3)
                    src.write(filled_data, 1)

        os.chdir('..')


    # Compute Soil Texture Class
    @staticmethod
    def soiltexturalclass(sand, clay):
        """
        Returns USDA soil textural class given percent sand and clay.
        :param sand:
        :param clay: Location to write input file to.
        """
        silt = 100 - sand - clay

        if sand + clay > 100 or sand < 0 or clay < 0:
            raise ValueError('Inputs add up to more than 100% or are negative')
        elif silt + 1.5 * clay < 15:
            textural_class = 1  # sand
        elif silt + 1.5 * clay >= 15 and silt + 2 * clay < 30:
            textural_class = 2  # loamy sand
        elif (clay >= 7 and clay < 20 and sand > 52 and silt + 2 * clay >= 30) or (
                clay < 7 and silt < 50 and silt + 2 * clay >= 30):
            textural_class = 3  # sandy loam
        elif clay >= 7 and clay < 27 and silt >= 28 and silt < 50 and sand <= 52:
            textural_class = 4  # loam
        elif (silt >= 50 and clay >= 12 and clay < 27) or (silt >= 50 and silt < 80 and clay < 12):
            textural_class = 5  # silt loam
        elif silt >= 80 and clay < 12:
            textural_class = 6  # silt
        elif clay >= 20 and clay < 35 and silt < 28 and sand > 45:
            textural_class = 7  # 'sandy clay loam'
        elif clay >= 27 and clay < 40 and sand > 20 and sand <= 45:
            textural_class = 8  # 'clay loam'
        elif clay >= 27 and clay < 40 and sand <= 20:
            textural_class = 9  # 'silty clay loam'
        elif clay >= 35 and sand > 45:
            textural_class = 10  # 'sandy clay'
        elif clay >= 40 and silt >= 40:
            textural_class = 11  # 'silty clay'
        elif clay >= 40 > silt and sand <= 45:
            textural_class = 12
        else:
            textural_class = 'na'

        return textural_class

    def process_raw_soil(self, grid_input, output=None, ks_only=False):
        """
        Writes ascii grids Ks, theta_s, theta_r, psib, and m from gridded soil data for % sand, silt, clay, bulk density, and volumetric water content at 33 and 1500 kPa.

        Parameters:
        - grid_input (list of dict or str): If a dictionary list, keys are "grid_type" and "path" for each soil property.
                                    Format of dictionary list follows:
                                    [{'type':'sand_fraction', 'path':'path/to/grid'},
                                    {'type':'silt_fraction', 'path':'path/to/grid'},
                                    {'type':'clay_fraction', 'path':'path/to/grid'},
                                    {'type':'bulk_density', 'path':'path/to/grid'},
                                    {'type':'vwc_33', 'path':'path/to/grid'},
                                    {'type':'vwc_1500', 'path':'path/to/grid'}]

                                    If a string is provided, it is treated as the path to a file containing the
                                    configuration file must be written in json format.

        - output (list, optional): List of output file names for different soil properties.
        - ks_only True will only write rasters for ks, this is useful if using compute_decay_ks

        Note:
        - The 'grid_types' key should contain a list of dictionaries, each specifying a grid type and its corresponding file path.
        - The 'output_files' key should contain a list of exactly 5 output file names for different soil properties.
        - The file paths in 'grid_types' should be valid, and the 'output_files' list should have the correct size.
        """

        # Check if grid_input is a string (path to a config file)
        if isinstance(grid_input, str):
            # Read configuration from the file
            config = self.read_json(grid_input)
            grids = config['grid_types']
            output_files = config['output_files']
        elif isinstance(grid_input, list):
            # Use provided dictionary
            grids = grid_input
            output_files = output or ['Ks.asc', 'theta_r.asc', 'theta_s.asc', 'psib.asc', 'm.asc']
        else:
            print('Invalid input format. Provide either a list of dictionaries specifying type and path, or a path to a configuration file.')
            return

        # Check if each file specified in the dictionary or config exists
        for g in grids:
            grid_type, path = g['type'], g['path']
            if not os.path.isfile(path):
                raise FileNotFoundError(f'Cannot find: {path} for grid type: {grid_type}')

        # Ensure output_files is a list of the correct size (5 elements)
        if output_files is not None and (not isinstance(output_files, list) or len(output_files) != 5):
            print('Output must be a list with 5 elements.')
            return

        sg250_data = None
        size = None
        geo_tiff = None

        # Loop through specified file paths
        for cnt, g in enumerate(grids):
            grid_type, path = g['type'], g['path']
            print(f"Ingesting {grid_type} from: {path}")
            geo_tiff = self.read_ascii(path)
            array = geo_tiff['data']
            size = array.shape

            if cnt == 0:
                sg250_data = np.zeros((6, size[0], size[1]))

            # each z layer follows:[sa (%), si (%), cl (%), bd (g/cm3), th33, th1500]
            if grid_type == 'sand_fraction':
                array = array / 1000 * 100  # convert SSC from g/kg to % SSC
                sg250_data[0, :, :] = array
            elif grid_type == 'silt_fraction':
                array = array / 1000 * 100  # convert SSC from g/kg to % SSC
                sg250_data[1, :, :] = array
            elif grid_type == 'clay_fraction':
                array = array / 1000 * 100  # convert SSC from g/kg to % SSC
                sg250_data[2, :, :] = array
            elif grid_type == 'bulk_density':
                array = array / 100  # convert bulk density from cg/cm3 to g/cm3
                sg250_data[3, :, :] = array
            elif grid_type == 'vwc_33':
                array = array / 1000  # convert bulk density from cg/cm3 to g/cm3
                sg250_data[4, :, :] = array
            elif grid_type == 'vwc_1500':
                array = array / 1000  # convert bulk density from cg/cm3 to g/cm3
                sg250_data[5, :, :] = array

        profile = geo_tiff['profile']

        # Initialize parameter grids, 3 grids - 1 mean values, 2 std deviations, 3 code/flag
        theta_r, theta_s, ks, psib, m = np.zeros((3, *size)), np.zeros((3, *size)), np.zeros((3, *size)), np.zeros(
            (1, *size)), np.zeros((1, *size))

        # Loop through raster's and compute soil properties using rosetta-soil package
        # Codes/Flags
        # 2	sand, silt, clay (SSC)
        # 3	SSC + bulk density (BD)
        # 4	SSC + BD + field capacity water content (TH33)
        # 5	SSC + BD + TH33 + wilting point water content (TH1500)
        # -1 no result returned, inadequate or erroneous data
        # each z layer follows:[sa (%), si (%), cl (%), bd (g/cm3), th33, th1500]
        # i.e SoilData([sa (%), si (%), cl (%), bd (g/cm3), th33, th1500])

        for i in range(0, size[0]):
            for j in range(0, size[1]):
                # Organize array for input into packag
                data = [sg250_data[x, i, j] for x in np.arange(0, 6)]
                soil_data = SoilData.from_array([data])
                mean, stdev, codes = rosetta(3, soil_data)  # apply Rosetta version 3
                theta_r[:, i, j] = [mean[0, 0], stdev[0, 0], codes[0]]
                theta_s[:, i, j] = [mean[0, 1], stdev[0, 1], codes[0]]
                # Convert ks from log10(cm/day) into mm/hr
                ks[:, i, j] = [(10 ** mean[0, 4]) * 10 / 24, (10 ** stdev[0, 4]) * 10 / 24, codes[0]]

                # Alpha parameter from rosetta corresponds approximately to the inverse of the air-entry value, cm−1
                # https://doi.org/10.1029/2019MS001784
                # Convert from log10(cm) into -1/mm
                psib[0, i, j] = -1 / ((10 ** mean[0, 2]) * 10)

                # Pore-size Distribution can be calculated from n using m = 1-1/n
                # http://dx.doi.org/10.4236/ojss.2012.23025
                # Convert from log10(n) into n
                m[0, i, j] = 1 - (1 / (10 ** mean[0, 3]))

        # for now only write out mean values
        soil_prop = [ks[0, :, :], theta_r[0, :, :], theta_s[0, :, :], psib[0, :, :], m[0, :, :]]

        if ks_only:
            soi_raster = {'data': soil_prop[0], 'profile': profile}
            self.write_ascii(soi_raster, output_files[0])
        else:
            for soil_property, name in zip(soil_prop, output_files):
                soi_raster = {'data': soil_property, 'profile': profile}
                self.write_ascii(soi_raster, name)


    def compute_ks_decay(self, grid_input, output=None):
        """
        Produces raster for the conductivity decay parameter f, following Ivanov et al., 2004.
        :param dict grid_input: If a dictionary list, keys are "depth" and "path" for each soil property.
                                Depth should be provided in units of mm.
                                    Format of dictionary list follows (from shallowest to deepest):
                                    [{'depth':25 , 'path':'path/to/25_mm_ks'},
                                    ...
                                    {'depth':800 , 'path':'path/to/800_mm_ks'},]

                                    If a string is provided, it is treated as the path to a configuration file. The
                                    configuration file must be written in json format.
        :param str output: Location to save raster with conductivity decay parameter f.
        """
        # Check if grid_input is a string (path to a config file)
        if isinstance(grid_input, str):
            # Read configuration from the file
            config = self.read_json(grid_input)
            grids = config['grid_depth']
            output_file = config['output_file']
        elif isinstance(grid_input, list):
            # Use provided dictionary
            grids = grid_input
            output_file = output or ['f.asc']
        else:
            print('Invalid input format. Provide either a list or a path to a configuration file.')
            return

        # Check if each file specified in the dictionary or config exists
        for g in grids:
            grid_type, path = g['depth'], g['path']
            if not os.path.isfile(path):
                raise FileNotFoundError(f'Cannot find: {path} for grid type: {grid_type}')

        ks_data = None
        size = None
        raster = None
        depth_vec = np.zeros(len(grids))

        # Loop through specified file paths
        for cnt, g in enumerate(grids):
            depth, path = g['depth'], g['path']
            print(f"Ingesting Ks grid at {depth} from: {path}")
            raster = self.read_ascii(path)
            array = raster['data']
            depth_vec[cnt] = depth

            if cnt == 0:
                size = array.shape
                ks_data = np.zeros((len(grids), size[0], size[1]))
                ks_data[cnt, :, :] = array
            else:
                ks_data[cnt, :, :] = array

        # Ensure that ks grids are sorted from surface to the deepest depth
        depth_sorted = np.argsort(depth_vec)
        ks_data = ks_data[depth_sorted]

        depth_vec = depth_vec[depth_sorted]
        depth_vec = depth_vec.astype(float)  # ensure float for fitting

        profile = raster['profile']  # for writing later

        # Initialize parameter grids
        f_grid = np.zeros(np.shape(array))  # parameter grid
        fcov = np.zeros(np.shape(array))  # coef of variance grid

        # Loop through raster's and compute soil properties using rosetta-soil package
        for i in range(0, size[0]):
            for j in range(0, size[1]):
                y = np.array([ks_data[n, i, j] for n in np.arange(0, len(grids))])

                if np.any(y == profile['nodata']):
                    f_grid[i, j] = profile['nodata']
                    fcov[i, j] = profile['nodata']
                else:

                    try:
                        # ensure float
                        y = y.astype(float)
                    except ValueError:
                        raise ValueError("Input data must be convertible to float")

                    k0 = y[0]

                    # define exponential decay function, Ivanov et al. (2004) eqn 17
                    decay = lambda x, f, k=k0: k * (f * x / (np.exp(f * x) - 1.0))

                    # perform curve fitting, set limits on f and surface Ksat for stability
                    minf, maxf = 1E-7, k0 - 1E-5
                    minks = 1

                    param, param_cov = curve_fit(decay, depth_vec, y, bounds=([minf, maxf], [minks, k0]))

                    # Write Curve fitting results to grid
                    f_grid[i, j] = param[0]
                    fcov[i, j] = param_cov[0, 0]

        f_raster = {'data': f_grid, 'profile': profile}
        self.write_ascii(f_raster, output_file[0])
