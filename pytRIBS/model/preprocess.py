import os

import numpy as np
import rasterio
import rasterio.fill as fill
from shapely.geometry import Point
import geopandas as gpd
from datetime import datetime

from pytRIBS.shared.inout import InOut


### PREPROCESSING
# 1 delineate watersheds (pyshed)
# 1.create points file for hydrological conditioned tin Mesh
# 2 ingest soil and land use (pyshed)
# 3 apply pedotransfer functions
# others?

class Preprocess(InOut):
    """

    """

    def create_results_dir(self,path=None):
    # content for make results directories
        if path is None:
            basin = self.name #'WS04'
            scenario = self.scenario #
            base_name = f"{basin}_"
            todays_date = datetime.now()
            todays_date = todays_date.strftime("%Y-%m-%d")
            results_base_path = f"results/{basin}/{todays_date}/{scenario}/"
    # results_base_hyd_path = f"{results_base_path}/hyd/"
    # ## USE this to add subdirectories if they do not exist
    # # Initialize the base directory
    # directories = results_base_hyd_path.split('/')
    # results_base_hyd_path = ''
    #
    # # Create the directories one by one
    # for directory in directories:
    #     results_base_hyd_path = os.path.join(results_base_hyd_path, directory)
    #     try:
    #         os.mkdir(results_base_hyd_path)
    #     except FileExistsError:
    #         pass


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

    # Compute Soil Texture Class
