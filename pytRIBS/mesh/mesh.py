import os
import sys
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
import numpy as np
import pyvista as pv
import geopandas as gpd
import matplotlib.pyplot as plt
import whitebox
from whitebox import WhiteboxTools
from shapely.geometry import shape, LineString, Polygon, MultiLineString, Point
from matplotlib.colors import ListedColormap
from rasterio.plot import show as rasterio_show
from osgeo import gdal, ogr, osr
import fiona
from fiona.crs import from_epsg
import shapely
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

class DEMPreprocessor:

    def __init__(self, config):
        self.config = config
        self.output_file_name = "{filename}dem.tif".format(filename=self.config.version)
        self.input_path = os.path.join(self.config.base_dir, self.config.input_file_directory, self.config.input_dem_name)
        self.preprocessed_dem = os.path.join(self.config.base_dir, "1_PREPROCESSED_DEM", self.output_file_name)
        self.filled_dem = os.path.join(self.config.base_dir, "2_DEM_ANALYSIS", f"{self.config.version}fill.tif")
        self.pour_point_shp = os.path.join(self.config.base_dir, "2_DEM_ANALYSIS", f"{self.config.version}pourpoint.shp")
        self.flow_direction_raster = os.path.join(self.config.base_dir, "2_DEM_ANALYSIS", f"{self.config.version}flowdir.tif")
        self.flow_accumulation_raster = os.path.join(self.config.base_dir, "2_DEM_ANALYSIS",
                                                     f"{self.config.version}flowacc.tif")
        self.watershed_raster = os.path.join(self.config.base_dir, "2_DEM_ANALYSIS", f"{self.config.version}wsh.tif")
        self.watershed_boundary = os.path.join(self.config.base_dir, "2_DEM_ANALYSIS", f"{self.config.version}bound1.shp")
        self.stream_order_raster = os.path.join(self.config.base_dir, "2_DEM_ANALYSIS", f"{self.config.version}strord.tif")
        self.stream_line_clipped = os.path.join(self.config.base_dir, "3_DEM_Analysis_Clipped", f"{self.config.version}line_c.shp")

        self.wbt = whitebox.WhiteboxTools()

        self.filled_dem_data = None
        self.filled_dem_transform = None

        # Ensure output directories exist
        preprocessed_dir = os.path.dirname(self.preprocessed_dem)
        if not os.path.exists(preprocessed_dir):
            os.makedirs(preprocessed_dir)

        analysis_dir = os.path.dirname(self.filled_dem)
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)

    def preprocess_dem(self):

        """

        Saves the DEM based on the version name and also reproject it to WGS 1984 UTM 12N if not already

        """

        # Open the input raster file
        with rasterio.open(self.input_path) as src:
            # Check if the CRS is already WGS 1984 (EPSG:4326)
            if 'EPSG' in src.crs and src.crs['init'] == 'EPSG:32612':
                print("Input DEM is already in WGS 1984 (EPSG:32612). No reprojection needed.")
                profile = src.profile
                preprocessed_dem = self.preprocessed_dem
            else:
                print("Reprojecting input DEM to WGS 1984 (EPSG:32612)...")
                transform, width, height = calculate_default_transform(
                    src.crs, 'EPSG:32612', src.width, src.height, *src.bounds)
                profile = src.profile.copy()
                profile.update({
                    'crs': 'EPSG:32612',
                    'transform': transform,
                    'width': width,
                    'height': height
                })
                data = src.read(
                    out_shape=(src.count, height, width),
                    resampling=Resampling.nearest
                )

                # Define output path for reprojected DEM in the same directory
                preprocessed_dem = os.path.join(os.path.dirname(self.preprocessed_dem), self.output_file_name)

            # Write raster data to GeoTIFF file
            with rasterio.open(preprocessed_dem, 'w', **profile) as dst:
                dst.write(data)

            print(f"Preprocessed DEM saved to: {preprocessed_dem}")

            print("Preprocessing complete.")

    def fill_depressions_and_plot(self):

        """
        Fill Sinks within the watershed and plot the DEM afterwards

        """

        # Check if filled DEM already exists and delete if it does
        if os.path.exists(self.filled_dem):
            os.remove(self.filled_dem)

        # Use WhiteboxTools for filling depressions
        wbt = WhiteboxTools()

        # Fill depressions in the preprocessed DEM
        wbt.fill_depressions(self.preprocessed_dem, self.filled_dem, fix_flats=True)

        print("Filled DEM saved to:", self.filled_dem)

        # Load filled DEM data
        with rasterio.open(self.filled_dem) as src:
            filled_dem_data = src.read(1, masked=True)  # Read the first band and mask NoData values
            filled_dem_data = np.ma.masked_where(filled_dem_data == src.nodata, filled_dem_data)  # Mask NoData values

            # Plot filled DEM raster
            plt.figure(figsize=(16, 12))
            plt.imshow(filled_dem_data, cmap='gray',
                       extent=[src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top])
            plt.colorbar(label='Elevation (m)')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title('Filled Digital Elevation Model (DEM)')
            plt.grid(False)
            plt.show()

    def create_pour_point_shapefile_and_plot(self):

        """Uses pour point coordinates to create shapefile of the pour point and plot it accordingly"""

        # Check if pour point shapefile already exists and delete if it does
        if os.path.exists(self.pour_point_shp):
            os.remove(self.pour_point_shp)

        # Create a Point geometry object
        pour_point_geometry = Point(self.config.pour_point_x, self.config.pour_point_y)

        # Create a GeoDataFrame with the Point geometry
        pour_point_gdf = gpd.GeoDataFrame(geometry=[pour_point_geometry])

        # Set the coordinate reference system (CRS) of the GeoDataFrame
        pour_point_gdf.crs = "EPSG:32612"  # WGS 1984 UTM Zone 12N

        # Save the GeoDataFrame to a shapefile
        pour_point_gdf.to_file(self.pour_point_shp)

        # Load filled DEM raster
        with rasterio.open(self.filled_dem) as src:
            filled_dem_data = src.read(1, masked=True)  # Read the first band and mask NoData values
            filled_dem_transform = src.transform  # Get the transform for plotting

        # Plot filled DEM raster with pour points
        plt.figure(figsize=(10, 8))
        rasterio_show(filled_dem_data, cmap='Greys', transform=filled_dem_transform, origin='upper', ax=plt.gca())
        pour_point_gdf.plot(ax=plt.gca(), color='red', markersize=50, label='Pour Point')

        # Add legend
        plt.legend()

        # Add title and labels
        plt.title('Filled DEM with Pour Point')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        # Show plot
        plt.show()


    def generate_flow_direction_raster_and_plot(self):

        """"
        Creates the flow direction raster based on the D8 method and plot it
        """

        # Check if flow direction raster already exists and delete if it does
        if os.path.exists(self.flow_direction_raster):
            os.remove(self.flow_direction_raster)

        # Use WhiteboxTools for generating D8 flow direction raster
        wbt = WhiteboxTools()
        wbt.d8_pointer(self.filled_dem, self.flow_direction_raster, esri_pntr=True)

        print("Flow Direction raster saved to:", self.flow_direction_raster)

        # Load flow direction raster
        with rasterio.open(self.flow_direction_raster) as src:
            d8_data = src.read(1, masked=True)  # Read the first band and mask NoData values
            d8_data = np.ma.masked_where(d8_data == src.nodata, d8_data)  # Mask NoData values
            transform = src.transform

        # Define custom colormap with unique colors for each value
        colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange', 'purple']
        unique_values = [1, 2, 4, 8, 16, 32, 64, 128]
        cmap = ListedColormap(colors)

        # Plot D8 flow direction raster with custom legend
        plt.figure(figsize=(20, 16))
        plt.imshow(d8_data, cmap=cmap, extent=[transform[2], transform[2] + transform[0] * src.width,
                                               transform[5] + transform[4] * src.height, transform[5]])
        plt.title('D8 Flow Direction')

        # Create custom legend with unique colors and values
        for i, value in enumerate(unique_values):
            plt.scatter([], [], color=colors[i], label=str(value))  # Create empty scatter plot for each value with custom color
        plt.legend(title='Flow Direction Values', loc='center', bbox_to_anchor=(1.1, 0.5))  # Create legend with custom colors
        plt.gca().set_aspect('equal')  # Set aspect ratio to equal
        plt.show()

    def generate_flow_accumulation_raster_and_plot(self):

        """
        Create the flow accumulation raster using the flow direction raster obtained from D8 method
        """

        # Check if flow accumulation raster already exists and delete if it does
        if os.path.exists(self.flow_accumulation_raster):
            os.remove(self.flow_accumulation_raster)

        # Use WhiteboxTools for generating D8 flow accumulation raster
        wbt = WhiteboxTools()
        wbt.d8_flow_accumulation(self.flow_direction_raster, self.flow_accumulation_raster, pntr=True, esri_pntr=True)

        print("Flow Accumulation raster saved to:", self.flow_accumulation_raster)

        # Load flow accumulation raster
        with rasterio.open(self.flow_accumulation_raster) as src:
            flow_accumulation_raster_data = src.read(1, masked=True)  # Read the first band and mask NoData values
            flow_accumulation_raster_data = np.ma.masked_where(flow_accumulation_raster_data == src.nodata,
                                                               flow_accumulation_raster_data)  # Mask NoData values

            # Plot flow accumulation raster
            plt.figure(figsize=(16, 12))
            plt.imshow(flow_accumulation_raster_data, cmap='gray',
                       extent=[src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top])
            plt.colorbar(label='Flow Accumulation Cells')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title('Flow Accumulation Raster')
            plt.grid(False)
            plt.show()

    def generate_watershed_raster(self):

        """
        Use the pour point shapefile and the flow direction raster to delineate the watershed in the form of a raster

        """

        # Check if watershed raster already exists and delete if it does
        if os.path.exists(self.watershed_raster):
            os.remove(self.watershed_raster)

        # Use WhiteboxTools for generating watershed raster
        wbt = WhiteboxTools()
        wbt.watershed(self.flow_direction_raster, self.pour_point_shp, self.watershed_raster, esri_pntr=True)

        print("Watershed raster saved to:", self.watershed_raster)

    def generate_watershed_boundary(self):

        """
        Develop the watershed boundary shapefile from the watershed raster
        """

        if os.path.exists(self.watershed_boundary):
            os.remove(self.watershed_boundary)

            # Generate the watershed raster
        wbt = WhiteboxTools()
        wbt.watershed(self.flow_direction_raster, self.pour_point_shp, self.watershed_raster, esri_pntr=True)

        # Generate the watershed boundary shapefile
        raster = gdal.Open(self.watershed_raster)
        band = raster.GetRasterBand(1)
        raster_array = band.ReadAsArray()
        raster_array = np.where(raster_array != band.GetNoDataValue(), 1, 0)
        proj = raster.GetProjection()
        shp_proj = osr.SpatialReference()
        shp_proj.ImportFromWkt(proj)
        driver = ogr.GetDriverByName('ESRI Shapefile')
        output_shapefile = driver.CreateDataSource(self.watershed_boundary)
        layer = output_shapefile.CreateLayer('layername', srs=shp_proj)
        field_defn = ogr.FieldDefn("ID", ogr.OFTInteger)
        layer.CreateField(field_defn)
        gdal.Polygonize(band, None, layer, 0, [], callback=None)
        output_shapefile = None
        raster = None

        # Load the shapefile into a GeoDataFrame
        gdf = gpd.read_file(self.watershed_boundary)

        # Remove polygons with ID == -32768 {means no data}
        filtered_gdf = gdf[gdf['ID'] != -32768]

        # Save the filtered GeoDataFrame back to a shapefile
        filtered_gdf.to_file(self.watershed_boundary)

    def extract_streams(self):

        """
        Obtain the stream raster from the flow accumulation raster based on the stream threshold provided before
        """

        # Initialize the WhiteboxTools object
        self.wbt = whitebox.WhiteboxTools()

        # Set path for stream raster
        self.stream_raster = os.path.join(self.config.base_dir, "2_DEM_ANALYSIS", f"{self.config.version}str.tif")

        # Check if output already exists and delete if it does
        if os.path.exists(self.stream_raster):
            os.remove(self.stream_raster)

        # Extract streams using flow accumulation raster and threshold
        self.wbt.extract_streams(
            self.flow_accumulation_raster,
            self.stream_raster,
            self.config.stream_threshold
        )

        print("Stream raster saved to:", self.stream_raster)

    def compute_stream_order(self):

        """
        Obtain the Strahler's Stream Order from the stream raster and flow direction raster
        """

        # Initialize the WhiteboxTools object
        self.wbt = whitebox.WhiteboxTools()

        # Set path for stream order raster
        self.stream_order_raster = os.path.join(self.config.base_dir, "2_DEM_ANALYSIS",
                                                f"{self.config.version}strord.tif")

        # Check if output already exists
        if os.path.exists(self.stream_order_raster):
            os.remove(self.stream_order_raster)

        # Compute Stream order
        self.wbt.strahler_stream_order(
            self.flow_direction_raster,
            self.stream_raster,
            self.stream_order_raster,
            esri_pntr=True
        )

    def convert_stream_raster_to_vector(self):

        """
        Create a shapefile of the stream network from the stream raster
        """

        # Set path for stream vector
        self.streamline_shp = os.path.join(self.config.base_dir, "2_DEM_ANALYSIS", f"{self.config.version}line.shp")

        # Check if output already exists and delete if it does
        if os.path.exists(self.streamline_shp):
            os.remove(self.streamline_shp)

        # Convert Stream Raster to Vector
        self.wbt.raster_streams_to_vector(
            self.stream_raster,
            self.flow_direction_raster,
            self.streamline_shp,
            esri_pntr=True
        )

        print("Streamline shapefile saved to:", self.streamline_shp)



    def clip_rasters(self):

        """
        Using the watershed boundary polygon to clip the filled dem, flow direction raster, stream raster and stream order raster
        """

        # Create Output Directory for clipped Rasters
        self.clipped_directory = os.path.join(self.config.base_dir, "3_DEM_Analysis_Clipped")
        os.makedirs(self.clipped_directory, exist_ok=True)

        # Define input rasters
        input_rasters = [
            self.filled_dem,
            self.flow_direction_raster,
            self.flow_accumulation_raster,
            self.stream_raster,
            self.stream_order_raster
        ]

        # Loop through each input raster
        for raster_file in input_rasters:
            # Define the output filename for the clipped raster
            output_filename = os.path.basename(raster_file).replace(".", "_c.")
            output_raster = os.path.join(self.clipped_directory, output_filename)

            # Open the raster file
            with rasterio.open(raster_file) as src:
                # Open the polygon shapefile
                with fiona.open(self.watershed_boundary, "r") as shapefile:
                    geometries = [feature["geometry"] for feature in shapefile]

                # Clip the raster with the polygon
                clipped_data, clipped_transform = mask(src, geometries, crop=True)

                # Update the metadata for the clipped raster
                clipped_profile = src.profile
                clipped_profile.update({
                    'height': clipped_data.shape[1],
                    'width': clipped_data.shape[2],
                    'transform': clipped_transform
                })

                # Write the clipped raster to the output directory
                with rasterio.open(output_raster, 'w', **clipped_profile) as dst:
                    dst.write(clipped_data)
            print(f"Clipped raster saved to: {output_raster}")

    def clip_streamline(self):

        """
        Streamlines shapefile to be clipped by watershed boundary polygon
        """

        # check if output file already exists and delete if it does
        if os.path.exists(self.stream_line_clipped):
            os.remove(self.stream_line_clipped)

        # Read the line shapefile
        lines = gpd.read_file(self.streamline_shp)

        # Set the CRS to EPSG:32612 [WGS 1984]
        lines.crs = from_epsg(32612)

        # Read the watershed boundary shapefile
        watershed_boundary_t = gpd.read_file(self.watershed_boundary)

        # Clip the lines with the watershed boundary polygon
        clipped_lines = gpd.clip(lines, watershed_boundary_t)

        # Save the clipped lines to a new shapefile
        clipped_lines.to_file(self.stream_line_clipped)


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
        self.inner_ring_bdry = os.path.join(self.config.base_dir, "4_Tin_Temporary_fILES", f"{self.config.version}irgen.shp")
        self.dem_path = os.path.join(self.config.base_dir, "3_DEM_Analysis_Clipped", f"{self.config.version}fill_c.tif")
        self.resampled_dem = os.path.join(self.config.base_dir, "3_DEM_Analysis_Clipped", f"{self.config.version}fill_c_r.tif")
        self.pour_point_shapefile = os.path.join(self.config.base_dir, "2_DEM_ANALYSIS", f"{self.config.version}pourpoint.shp")
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
        dsRes = gdal.Warp(self.resampled_dem, ds, xRes=self.config.dem_resample_res, yRes=self.config.dem_resample_res, resampleAlg="bilinear")


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
