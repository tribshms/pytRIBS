import numpy as np
from shapely.geometry import Point
from datetime import datetime
from pytRIBS.shared.inout import InOut
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

