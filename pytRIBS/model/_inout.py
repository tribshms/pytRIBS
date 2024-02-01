import os
from datetime import datetime
import getpass
import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import Point

def read_point_files(instance):
    """
    Returns Pandas dataframe of nodes or point used in tRIBS mesh.
    """

    file_path = instance.options['pointfilename']['value']

    node_points = []
    node_z = []
    node_bc = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        num_points = lines.pop(0)

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 0:
                x, y, z, bc = map(float, parts)
                node_points.append(Point(x, y))
                node_z.append(z)
                node_bc.append(bc)

        node_features = {'bc': node_bc, 'geometry': node_points, 'elevation': node_z}
        if instance.geo["EPSG"] is not None:
            nodes = gpd.GeoDataFrame(node_features, crs=instance.geo["EPSG"])
        else:
            nodes = gpd.GeoDataFrame(node_features)
            print("Coordinate Reference System (CRS) was not added to the GeoDataFrame")
        return nodes

def write_input_file(instance, output_file_path):
    """
    Writes .in file for tRIBS model simulation.
    :param instance:
    :param output_file_path: Location to write input file to.
    """
    with open(output_file_path, 'w') as output_file:

        current_datetime = datetime.now()
        current_user = getpass.getuser()

        formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        output_file.write(f"Created by: {current_user}\n")
        output_file.write(f"On: {formatted_datetime}\n\n")

        for key, subdict in instance.options.items():
            if "key_word" in subdict and "value" in subdict:
                keyword = subdict["key_word"]
                value = subdict["value"]
                output_file.write(f"{keyword}\n")
                output_file.write(f"{value}\n\n")


def read_precip_sdf(instance, file_path=None):
    """
    Returns list of precip stations, where information from each station is stored in a dictionary.
    :param file_path: Reads from options["hydrometstations"]["value"], but can be separately specified.
    :return: List of dictionaries.
    """

    if file_path is None:
        file_path = instance.options["gaugestations"]["value"]

        if file_path is None:
            print(instance.options["gaugestations"]["key_word"] + "is not specified.")
            return None

    station_list = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    metadata = lines.pop(0)
    num_stations, num_parameters = map(int, metadata.strip().split())

    for l in lines:
        station_info = l.strip().split()
        if len(station_info) == 7:
            station_id, file_path, lat, long, record_length, num_params, elevation = station_info
            station = {
                "station_id": station_id,
                "file_path": file_path,
                "y": float(lat),
                "x": float(long),
                "record_length": int(record_length),
                "num_parameters": int(num_params),
                "elevation": float(elevation)
            }
            station_list.append(station)

    if len(station_list) != num_stations:
        print("Error: Number of stations does not match the specified count.")

    return station_list
@staticmethod
def read_precip_station(file_path):
    """
    Returns pandas dataframe of precipitation from a station specified by file_path.
    :param file_path: Flat file with columns Y M D H R
    :return: Pandas dataframe
    """
    # TODO add var for specifying Station ID
    df = pd.read_csv(file_path, header=0, sep=r"\s+")
    df.rename(columns={'Y': 'year', 'M': 'month', 'D': 'day', 'H': 'hour'}, inplace=True)
    df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df.drop(['year', 'month', 'day', 'hour'], axis=1, inplace=True)

    return df
@staticmethod
def write_precip_sdf(station_list, output_file_path):
    """
    Writes a list of precip stations to a flat file.
    :param station_list: List of dictionaries containing station information.
    :param output_file_path: Output flat file path.
    """
    with open(output_file_path, 'w') as file:
        # Write metadata line
        metadata = f"{len(station_list)} {len(station_list[0])}\n"
        file.write(metadata)

        # Write station information
        for station in station_list:
            line = f"{station['station_id']} {station['file_path']} {station['y']} {station['x']} " \
                   f"{station['record_length']} {station['num_parameters']} {station['elevation']}\n"
            file.write(line)
def write_precip_station(df, output_file_path):
    """
    Converts a DataFrame with 'date' and 'R' columns to flat file format with columns Y M D H R.
    :param df: Pandas DataFrame with 'date' and 'R' columns.
    :param output_file_path: Output flat file path.
    """
    # Extract Y, M, D, and H from the 'date' column
    df['Y'] = df['date'].dt.year
    df['M'] = df['date'].dt.month
    df['D'] = df['date'].dt.day
    df['H'] = df['date'].dt.hour

    # Reorder columns
    df = df[['Y', 'M', 'D', 'H', 'R']]

    # Write DataFrame to flat file
    df.to_csv(output_file_path, sep=' ', index=False)
def read_met_sdf(instance, file_path=None):
    """
    Returns list of met stations, where information from each station is stored in a dictionary.
    :param file_path: Reads from options["hydrometstations"]["value"], but can be separately specified.
    :return: List of dictionaries.
    """
    if file_path is None:
        file_path = instance.options["hydrometstations"]["value"]

        if file_path is None:
            print(instance.options["hydrometstations"]["key_word"] + "is not specified.")
            return None

    station_list = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    metadata = lines.pop(0)
    num_stations, num_parameters = map(int, metadata.strip().split())

    for l in lines:
        station_info = l.strip().split()

        if len(station_info) == 10:
            station_id, file_path, lat, y, long, x, gmt, record_length, num_params, other = station_info
            station = {
                "station_id": station_id,
                "file_path": file_path,
                "lat_dd": float(lat),
                "x": float(x),
                "long_dd": float(long),
                "y": float(y),
                "GMT": int(gmt),
                "record_length": int(record_length),
                "num_parameters": int(num_params),
                "other": other
            }
            station_list.append(station)

    if len(station_list) != num_stations:
        print("Error: Number of stations does not match the specified count.")

    return station_list

def read_met_station(file_path):
    # TODO add var for specifying Station ID and doc
    df = pd.read_csv(file_path,header=0,sep='\s+')
    # convert year, month, day to datetime and drop columns
    df.rename(columns={'Y': 'year', 'M': 'month', 'D': 'day', 'H': 'hour'}, inplace=True)
    df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.drop(['year', 'month', 'day', 'hour'], axis=1)
    return df

def write_met_station(instance):
    # TODO
    pass

def read_soil_table(instance):
    pass

def write_soil_table(instance):
    pass

def read_landuse_table(instance):
    pass

def write_landuse_table(instance):
    pass

def read_grid_data_file(instance, grid_type):
    """
    Returns dictionary with content of a specified Grid Data File (.gdf)
    :param grid_type: string set to "weather", "soil", of "land", with each corresponding to HYDROMETGRID, SCGRID, LUGRID
    :return: dictionary containg keys and content: "Number of Parameters","Latitude", "Longitude","GMT Time Zone", "Parameters" (a  list of dicts)
    """

    if grid_type == "weather":
        option = instance.options["hydrometgrid"]["value"]
    elif grid_type == "soil":
        option = instance.options["scgrid"]["value"]
    elif grid_type == "land":
        option = instance.options["lugrid"]["value"]

    parameters = []

    with open(option, 'r') as file:
        num_parameters = int(file.readline().strip())
        location_info = file.readline().strip().split()
        latitude, longitude, gmt_timezone = location_info

        variable_count = 0

        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                variable_name, raster_path, raster_extension = parts
                variable_count += 1

                path_components = raster_path.split(os.path.sep)

                # Exclude the last directory as its actually base name
                raster_path = os.path.sep.join(path_components[:-1])

                if raster_path != "NO_DATA":
                    if not os.path.exists(raster_path):
                        print(
                            f"Warning: Raster file not found for Variable '{variable_name}': {raster_path}")
                        raster_path = None
                    elif os.path.getsize(raster_path) == 0:
                        print(
                            f"Warning: Raster file is empty for Variable '{variable_name}': {raster_path}")
                        raster_path = None
                elif raster_path == "NO_DATA":
                    print(
                        f"Warning: No rasters set for variable '{variable_name}'")
                    raster_path = None

                parameters.append({
                    'Variable Name': variable_name,
                    'Raster Path': raster_path,
                    'Raster Extension': raster_extension
                })
            else:
                print(f"Skipping invalid line: {line}")

        if variable_count > num_parameters:
            print(
                "Warning: The number of variables exceeds the number of parameters. This variable has been reset in dictionary.")

    return {
        'Number of Parameters': variable_count,
        'Latitude': latitude,
        'Longitude': longitude,
        'GMT Time Zone': gmt_timezone,
        'Parameters': parameters
        }

def read_ascii(file_path):
    """
    Returns dictionary containing 'data', 'profile', and additional metadata.
    :param file_path: Path to ASCII raster.
    :return: Dict
    """
    raster = {}

    # Open the raster file using rasterio
    with rasterio.open(file_path) as src:
        # Read the raster data as a NumPy array
        raster['data'] = src.read(1)  # Assuming a single band raster, adjust accordingly

        # Access the metadata
        raster['profile'] = src.profile

    return raster

def write_ascii(raster_dict, output_file_path):
    """
    Writes raster data and metadata from a dictionary to an ASCII raster file.
    :param raster_dict: Dictionary containing 'data', 'profile', and additional metadata.
    :param output_file_path: Output ASCII raster file path.
    """
    # Extract data and metadata from the dictionary
    data = raster_dict['data']
    profile = raster_dict['profile']

    # Write the data and metadata to the ASCII raster file
    with rasterio.open(output_file_path, 'w', **profile) as dst:
        dst.write(data, 1)
