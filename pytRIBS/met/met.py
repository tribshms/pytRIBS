import numpy as np
import pandas as pd
from shapely.geometry import box
import pynldas2 as nldas
import pyproj
import os
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import xarray as xr
import requests
from io import BytesIO
from pytRIBS.shared.inout import InOut



class _Met():
    def get_nldas_point(self, centroids, begin, end, epsg=None, write_path=None, **hyriver_env_vars):
        """
        Fetches NLDAS data for a given set of coordinates and time period, with optional caching and environment variable configuration.

        This function is a wrapper around the pynldas2 library, which is cited as follows:
        Chegini T, Li H-Y, Leung LR. 2021. HyRiver: Hydroclimate Data Retriever. Journal of Open Source Software 6: 3175. DOI: 10.21105/joss.03175

        :param str geom: The geometry for which the data is being requested.
        :param str begin: The start date for the data request in 'YYYY-MM-DD' format.
        :param str end: The end date for the data request in 'YYYY-MM-DD' format.
        :param int epsg: The EPSG code for the coordinate reference system of the geometry.
        :param str write_path: The path where the resulting xarray dataset should be saved as a NetCDF file, optional.
        :param **hyriver_env_vars: Additional keyword arguments representing environment variables to control request/response caching and verbosity.

        The following environment variables can be set via **hyriver_env_vars:
        - HYRIVER_CACHE_NAME: Path to the caching SQLite database for asynchronous HTTP requests. Defaults to ./cache/aiohttp_cache.sqlite.
        - HYRIVER_CACHE_NAME_HTTP: Path to the caching SQLite database for HTTP requests. Defaults to ./cache/http_cache.sqlite.
        - HYRIVER_CACHE_EXPIRE: Expiration time for cached requests in seconds. Defaults to one week.
        - HYRIVER_CACHE_DISABLE: Disable reading/writing from/to the cache. Defaults to false.
        - HYRIVER_SSL_CERT: Path to an SSL certificate file.

        :returns: The dataset containing the NLDAS data for the specified geometry and time period.
        :rtype: pandas dataframe
        """

        # Set environment variables from hyriver_env_vars
        for key, item in hyriver_env_vars.items():
            os.environ[key] = item

        if epsg is None:
            epsg = self.meta['EPSG']
        # Fetch data using the nldas library
        df = nldas.get_bycoords(centroids, begin, end, crs=epsg, source='netcdf')

        return df
    @staticmethod
    def get_nldas_geom(geom, begin, end, epsg, write_path=None, **hyriver_env_vars):
        """
        Fetches NLDAS data for a given geometry and time period, with optional caching and environment variable configuration.

        This function is a wrapper around the pynldas2 library, which is cited as follows:
        Chegini T, Li H-Y, Leung LR. 2021. HyRiver: Hydroclimate Data Retriever. Journal of Open Source Software 6: 3175. DOI: 10.21105/joss.03175

        :param str geom: The geometry for which the data is being requested.
        :param str begin: The start date for the data request in 'YYYY-MM-DD' format.
        :param str end: The end date for the data request in 'YYYY-MM-DD' format.
        :param int epsg: The EPSG code for the coordinate reference system of the geometry.
        :param str write_path: The path where the resulting xarray dataset should be saved as a NetCDF file, optional.
        :param **hyriver_env_vars: Additional keyword arguments representing environment variables to control request/response caching and verbosity.

        The following environment variables can be set via **hyriver_env_vars:
        - HYRIVER_CACHE_NAME: Path to the caching SQLite database for asynchronous HTTP requests. Defaults to ./cache/aiohttp_cache.sqlite.
        - HYRIVER_CACHE_NAME_HTTP: Path to the caching SQLite database for HTTP requests. Defaults to ./cache/http_cache.sqlite.
        - HYRIVER_CACHE_EXPIRE: Expiration time for cached requests in seconds. Defaults to one week.
        - HYRIVER_CACHE_DISABLE: Disable reading/writing from/to the cache. Defaults to false.
        - HYRIVER_SSL_CERT: Path to an SSL certificate file.

        :returns: The dataset containing the NLDAS data for the specified geometry and time period.
        :rtype: xarray.Dataset
        """

        # Assuming gdf is your GeoDataFrame with a 'geometry' column

        # Check if geometry column contains only Polygons
        if geom.geom_type == 'Polygon':
            # Convert Polygon to MultiPolygon
            geom = MultiPolygon([geom])

        # Set environment variables from hyriver_env_vars
        for key, item in hyriver_env_vars.items():
            os.environ[key] = item

        # Fetch data using the nldas library
        ds_xarray = nldas.get_bygeom(geom, begin, end, epsg, source='netcdf')

        # Write to NetCDF file if write_path is provided
        if write_path is not None:
            ds_xarray.to_netcdf(write_path)

        return ds_xarray

    from pyproj import CRS
    @staticmethod
    def get_nldas_elevation(watershed, epsg):
        """
        Downloads a NetCDF file of the NLDAS elevation grid and returns it as an xarray DataSet.

        :param geopandas.GeoDataFrame watershed: The watershed to clip the elevation data to.
        :param int epsg: The EPSG code for the desired projection of the output data.

        :returns: The processed elevation data clipped to the watershed extent and reprojected.
        :rtype: xarray.Dataset
        """

        # TODO: Need to make it so that this can be cached or passed in as a variable rather than downloaded.

        url = "https://ldas.gsfc.nasa.gov/sites/default/files/ldas/nldas/NLDAS_elevation.nc4"

        try:
            # Send a GET request to the URL
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for unsuccessful status codes

            # Open the downloaded content as an xarray DataSet
            with xr.open_dataset(BytesIO(response.content)) as ds:
                dataset = ds.load()  # Load the dataset into memory

            # Drop unnecessary variables
            if 'time_bnds' in dataset.variables:
                dataset = dataset.drop_vars('time_bnds')

            # Write the CRS to the dataset
            dataset = dataset.rio.write_crs(32662)  # EPSG code 32662 for Equidistant Cylindrical projection, default

            return dataset

        except requests.exceptions.RequestException as e:
            print(f"Error downloading NLDAS elevation file: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    @staticmethod
    def create_nldas_grid_mask(ds, epsg=None):
        """
        Create polygons representing each pixel in a grid based on GeoTransform parameters.

        :param xarray.Dataset ds: The input dataset with spatial reference information.

        :returns: GeoDataFrame containing polygons representing each pixel.
        :rtype: geopandas.GeoDataFrame
        """

        # Extract geotransform from the dataset's spatial reference
        geotransform_str = ds.spatial_ref.GeoTransform
        geotransform = tuple(map(float, geotransform_str.split()))

        # Get number of rows and columns from the dataset
        cols, rows = len(ds.x.values), len(ds.y.values)

        # Initialize a list to hold polygon geometries
        polygons = []

        # Calculate polygons for each pixel
        for row in range(rows):
            for col in range(cols):
                # Calculate pixel coordinates
                left = geotransform[0] + col * geotransform[1]
                right = geotransform[0] + (col + 1) * geotransform[1]
                top = geotransform[3] + row * geotransform[5]
                bottom = geotransform[3] + (row + 1) * geotransform[5]

                # Correct the order of top and bottom if necessary
                if geotransform[5] > 0:
                    top, bottom = bottom, top

                # Create polygon geometry
                polygon = box(left, bottom, right, top)
                polygons.append(polygon)

        # Create a GeoDataFrame from polygons
        gdf = gpd.GeoDataFrame(geometry=polygons)

        if epsg is not None:
            gdf.set_crs(epsg, inplace=True)

        return gdf
    @staticmethod
    def clip_nldas_grid_mask_to_watershed(mask, watershed, epsg):
        """
        Clip a target GeoDataFrame by each polygon in the pixel GeoDataFrame.

        :param geopandas.GeoDataFrame mask: GeoDataFrame containing pixel polygons.
        :param geopandas.GeoDataFrame watershed: GeoDataFrame to be clipped.

        :returns: List of clipped GeoDataFrames, one for each pixel polygon.
        :rtype: list
        """

        # need to convert to utm
        utm_min = [int((x + 180) // 6) + 1 for x in watershed.bounds.minx.values]
        utm_max = [int((x + 180) // 6) + 1 for x in watershed.bounds.maxx.values]

        # for hemisphere check
        min_y = watershed.bounds.miny.min()
        max_y = watershed.bounds.maxy.max()

        # checks if watershed is in one utm zone and assigns EPSG code accordingly. If it spans utm zones or hemispheres
        # than the web meractor projeciton is used. Note defaults on NAD83 since NLDAS-2 is for north america, similarly the
        # above conditionals are likely not needed, but here for future modificaitons/options.

        if np.unique(utm_min) == np.unique(utm_max):
            utm_zone = np.unique(utm_min)[0]
            if min_y >= 0:
                utm_crs = f"EPSG:269{utm_zone}"
            elif max_y <= 0:
                utm_crs = f"EPSG:269{utm_zone}"
            else:
                utm_crs = f"EPSG:3857"  # web mercator
        else:
            utm_crs = f"EPSG:3857"  # web mercator

        clipped_gdfs = []
        for i, pixel in enumerate(mask.geometry):
            clipped = gpd.clip(watershed, pixel)
            clipped_gdfs.append(clipped)

        clipped_watershed = pd.concat(clipped_gdfs, ignore_index=True)
        clipped_watershed.set_crs(watershed.crs, inplace=True)
        # set lat and long columns for centroid
        clipped_watershed.to_crs(utm_crs, inplace=True)
        clipped_watershed['x'] = clipped_watershed.centroid.x.values
        clipped_watershed['y'] = clipped_watershed.centroid.y.values

        # Define the coordinate transformation
        proj = pyproj.CRS(epsg)  # geographic coordinate system
        utm = pyproj.CRS(utm_crs)  # UTM coordinate system
        transformer = pyproj.Transformer.from_crs(utm, watershed.crs, always_xy=True)
        clipped_watershed['long'], clipped_watershed['lat'] = transformer.transform(clipped_watershed['x'],
                                                                                    clipped_watershed['y'])

        # set area for thresholding in extract_nldas_timeseries
        clipped_watershed['area'] = clipped_watershed.geometry.area

        return clipped_watershed, utm_crs
    @staticmethod
    def extract_nldas_timeseries(gridded_watershed, nldas_met_xarray, nldas_elev_xarray, threshold_area=0):
        """
        Convert an xarray dataset to a pandas dataframe for a given location.

        :param xarray.Dataset ds: xarray dataset.
        :param tuple location: coordinates (e.g., (lat, lon)).
        :param list coords: coordinates to include in the dataframe, optional.
        :param list variables: variables to include in the dataframe, optional.

        :returns: pandas dataframe.
        :rtype: pd.DataFrame
        """

        nldas_time_series = []
        station_coordinates = []  # x,y,z

        for count in range(0, len(gridded_watershed)):

            sub_watershed = gridded_watershed.iloc[count]
            area = sub_watershed.area

            if area > threshold_area:
                # get coords
                long = sub_watershed.long
                lat = sub_watershed.lat
                x = sub_watershed.x
                y = sub_watershed.y

                # extract time series and convert to data frame
                met_station = nldas_met_xarray.sel(x=long, y=lat, method='nearest')
                elev_station = nldas_elev_xarray.sel(lon=long, lat=lat, method='nearest')
                met_df = met_station.to_dataframe()
                elev_df = elev_station.to_dataframe()

                # append results to list
                z = elev_df.NLDAS_elev.iloc[0]
                nldas_time_series.append(met_df)
                station_coordinates.append([long, x, lat, y, z])

        return nldas_time_series, station_coordinates

    def convert_and_write_nldas_timeseries(self, list_dfs, station_coords,gmt, prefix=None, met_path=None, precip_path=None):
        """
        Convert NLDAS timeseries data to UTM coordinates and prepare for tRIBS input.

        :param list list_dfs: List of DataFrames, each containing NLDAS timeseries data with specific columns such as 'date', 'psurf', 'wind_u', 'wind_v', 'temp', 'humidity', 'rsds', and 'prcp'.
        :param list station_coords: List of tuples, each containing the (longitude, latitude, elevation) for each station.
        :param str prefix: Prefix for the output filenames.
        :param str met_path: Directory path where meteorological files will be saved.
        :param str precip_path: Directory path where precipitation files will be saved.
        :param int gmt: GMT offset for the data.
        :param str utm_epsg: EPSG code for the UTM coordinate system.

        :returns: The function writes the transformed timeseries data and station details to specified files.
        :rtype: None
        """

        if prefix is None and self.hydrometbasename['value'] is not None:
            prefix = self.hydrometbasename['value']
        else:
            prefix = 'MetResults'

        if met_path is None and self.hydrometstations['value'] is not None:
            met_path = self.hydrometstations['value']
        else:
            prefix = ''

        if precip_path is None and self.gaugestations ['value'] is not None:
            precip_path = self.gaugestations ['value']
        else:
            prefix = ''

        if os.path.isfile(met_path):
            met_dir = os.path.dirname(met_path)
        else:
            met_dir = ''

        if os.path.isfile(precip_path):
            precip_dir = os.path.dirname(precip_path)
        else:
            precip_dir = ''

        met_sdf_list = []
        precip_sdf_list = []

        # Hard coded params for writing
        count = 1
        num_params_precip = 5
        num_params_met = 13

        # Physical constants
        L = 2.453 * 10 ** 6  # Latent heat of vaporization (J/kg)
        Rv = 461  # Gas constant for moist air (J/kg-K)

        for df in list_dfs:
            # Initialize dictionaries for station details
            met_sdf = {'station_id': None, 'file_path': None, 'lat_dd': None, 'y': None, 'long_dd': None, 'x': None,
                       'GMT': None, 'record_length': None, 'num_parameters': None, 'other': None}
            precip_sdf = {'station_id': None, 'file_path': None, 'y': None, 'x': None, 'record_length': None,
                          'num_parameters': None, 'elevation': None}

            # Update to tRIBS variables
            df['XC'] = 9999.99
            df['TS'] = 9999.99
            df['NR'] = 9999.99
            df['psurf'] *= 0.01  # Convert pressure from Pa to hPa

            df['US'] = (df['wind_u'] ** 2 + df['wind_v'] ** 2) ** 0.5  # Wind speed
            df['TA'] = df['temp'] - 273.15  # Temperature in Celsius
            df['e_sat'] = 6.11 * np.exp(
                (L / Rv) * ((1 / 273.15) - (1 / df['temp'])))  # Saturation vapor pressure in hPa

            # Calculate saturation vapor pressure (e_sat) in hPa (mb)
            df['VP'] = (df['humidity'] * df['psurf']) / 0.622
            df['RH'] = 100 * (df['VP'] / df['e_sat'])

            df.rename(columns={'rsds': 'IS', 'prcp': 'R', 'psurf': 'PA'}, inplace=True)
            df['date'] = df.index.values

            # Write out files with pytrib utility class InOut
            precip_file = f'precip_{prefix}_{count}.mdf'
            met_file = f'met_{prefix}_{count}.mdf'

            precip_file_path = os.path.join(precip_dir, precip_file)
            met_file_path = os.path.join(met_dir, met_file)

            InOut.write_precip_station(df[['R', 'date']].copy(), precip_file_path)
            InOut.write_met_station(df[['PA', 'RH', 'XC', 'TS', 'NR', 'TA', 'US', 'VP', 'IS', 'date']].copy(),
                                    met_file_path)

            # Update sdf dictionaries
            met_sdf['station_id'] = count
            precip_sdf['station_id'] = count
            met_sdf['file_path'] = met_file_path
            precip_sdf['file_path'] = precip_file_path

            # Geographic coordinates
            lat = station_coords[count - 1][2]
            y = station_coords[count - 1][3]
            long = station_coords[count - 1][0]
            x = station_coords[count - 1][1]

            met_sdf['lat_dd'] = lat
            met_sdf['long_dd'] = long

            met_sdf['x'] = x
            met_sdf['y'] = y
            precip_sdf['x'] = x
            precip_sdf['y'] = y

            met_sdf['GMT'] = gmt
            precip_sdf['elevation'] = station_coords[count - 1][4]

            met_sdf['num_parameters'] = num_params_met
            precip_sdf['num_parameters'] = num_params_precip

            length = len(df['date'])

            met_sdf['record_length'] = length
            precip_sdf['record_length'] = length

            met_sdf_list.append(met_sdf)
            precip_sdf_list.append(precip_sdf)

            count += 1

        InOut.write_met_sdf(met_path, met_sdf_list)
        InOut.write_precip_sdf(precip_sdf_list, precip_path)

    def run_met_workflow(self,watershed, begin, end):
        epsg = self.meta['EPSG']
        elev = self.get_nldas_elevation(watershed, epsg=epsg)
        nldas_ds = self.get_nldas(watershed.to_crs(epsg).geometry[0], begin, end, epsg)
        mask = self.create_nldas_grid_mask(nldas_ds, epsg=epsg)
        grid_watershed, _ = self.clip_nldas_grid_mask_to_watershed(mask, watershed.to_crs(epsg), epsg)
        dfs, coords = self.extract_nldas_timeseries(grid_watershed.to_crs(epsg), nldas_ds, elev)


