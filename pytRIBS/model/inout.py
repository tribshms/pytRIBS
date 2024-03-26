import os
from datetime import datetime
import getpass
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import math
from shapely.geometry import Point
import json


class InOut:
    def read_point_files(self):
        """
        Returns Pandas dataframe of nodes or point used in tRIBS mesh.
        """

        file_path = self.options['pointfilename']['value']

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
            if self.geo["EPSG"] is not None:
                nodes = gpd.GeoDataFrame(node_features, crs=self.geo["EPSG"])
            else:
                nodes = gpd.GeoDataFrame(node_features)
                print("Coordinate Reference System (CRS) was not added to the GeoDataFrame")
            return nodes
    @staticmethod
    def write_point_file(nodes_gdf, output_file):
        """
        Write a points file from a GeoDataFrame of nodes.

        Parameters:
        - nodes_gdf: GeoDataFrame
            GeoDataFrame containing nodes with 'geometry', 'elevation', and 'bc' columns.
        - output_file: str
            Path to the output points file.

        Returns:
        None
        """
        # Ensure 'geometry' column is of Point type
        if not isinstance(nodes_gdf['geometry'].iloc[0], Point):
            nodes_gdf['geometry'] = nodes_gdf['geometry'].apply(Point)

        # Open the output file for writing
        with open(output_file, 'w') as file:
            # Write the number of points as the first line
            file.write(f"{len(nodes_gdf)}\n")

            # Write each point's x, y, z, and bc values on separate lines
            for _, row in nodes_gdf.iterrows():
                x, y = row['geometry'].x, row['geometry'].y
                z, bc = row['elevation'], int(row['bc'])
                file.write(f"{x} {y} {z} {bc}\n")

    def write_input_file(self, output_file_path):
        """
        Writes .in file for tRIBS model simulation.
        :param self:
        :param output_file_path: Location to write input file to.
        """
        with open(output_file_path, 'w') as output_file:

            current_datetime = datetime.now()
            current_user = getpass.getuser()

            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

            output_file.write(f"Created by: {current_user}\n")
            output_file.write(f"On: {formatted_datetime}\n\n")

            for key, subdict in self.options.items():
                if "key_word" in subdict and "value" in subdict:
                    keyword = subdict["key_word"]
                    value = subdict["value"]
                    if value is None:
                        value = ""
                    output_file.write(f"{keyword}\n")
                    output_file.write(f"{value}\n\n")

    def read_precip_sdf(self, file_path=None):
        """
        Returns list of precip stations, where information from each station is stored in a dictionary.
        :param file_path: Reads from options["hydrometstations"]["value"], but can be separately specified.
        :return: List of dictionaries.
        """

        if file_path is None:
            file_path = self.options["gaugestations"]["value"]

            if file_path is None:
                print(self.options["gaugestations"]["key_word"] + "is not specified.")
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

    @staticmethod
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

    def read_met_sdf(self, file_path=None):
        """
        Returns list of met stations, where information from each station is stored in a dictionary.
        :param file_path: Reads from options["hydrometstations"]["value"], but can be separately specified.
        :return: List of dictionaries.
        """
        if file_path is None:
            file_path = self.options["hydrometstations"]["value"]

            if file_path is None:
                print(self.options["hydrometstations"]["key_word"] + "is not specified.")
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

    @staticmethod
    def read_met_station(file_path):
        # TODO add var for specifying Station ID and doc
        df = pd.read_csv(file_path, header=0, sep=r'\s+')
        # convert year, month, day to datetime and drop columns
        df.rename(columns={'Y': 'year', 'M': 'month', 'D': 'day', 'H': 'hour'}, inplace=True)
        df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df = df.drop(['year', 'month', 'day', 'hour'], axis=1)
        return df
    @staticmethod
    def write_met_station(df, output_file_path):
        """
        Converts a DataFrame with 'date' and 'PA','TD' or 'RH' or 'VP','XC','US','TA','TS','NR' columns to flat file format.
        See tRIBS documentation for more details on weather station data structure (i.e. *mdf files).
        :param df: Pandas DataFrame with 'date' and 'R' columns.
        :param output_file_path: Output flat file path.
        """
        # Extract Y, M, D, and H from the 'date' column
        df['Y'] = df['date'].dt.year
        df['M'] = df['date'].dt.month
        df['D'] = df['date'].dt.day
        df['H'] = df['date'].dt.hour

        # Format 'D' and 'H' columns with zero-padding
        df['D'] = df['D'].apply(lambda x: str(x).zfill(2))
        df['H'] = df['H'].apply(lambda x: str(x).zfill(2))

        # Check which column ('TD', 'RH', or 'VP') is present in the DataFrame
        present_column = next((col for col in ['TD', 'RH', 'VP'] if col in df.columns), None)

        if present_column is not None:
            # Reorder columns
            df = df[['Y', 'M', 'D', 'H', 'PA', present_column, 'XC', 'US', 'TA', 'IS', 'TS', 'NR']]

            # Write DataFrame to flat file with tab as separator
            df.to_csv(output_file_path, sep='\t', index=False)
        else:
            print("Error: One of 'TD', 'RH', or 'VP' column must be present in the DataFrame.")


    @staticmethod
    def write_met_sdf(output_file_path, station_list):
        """
        Writes a list of meteorological stations to a flat file (i.e. *.sdf file).
        :param station_list: List of dictionaries containing station information.
        :param output_file_path: Output flat file path.
        """
        with open(output_file_path, 'w') as file:
            # Write metadata line
            metadata = f"{len(station_list)} {len(station_list[0])}\n"
            file.write(metadata)

            # Write station information
            for station in station_list:
                line = f"{station['station_id']} {station['file_path']} {station['lat_dd']} {station['y']} {station['long_dd']} {station['x']} " \
                       f"{station['GMT']} {station['record_length']} {station['num_parameters']} {station['other']}\n"
                file.write(line)


    def read_soil_table(self, file_path=None):
        """
        Soil Reclassification Table Structure (*.sdt)
        #Types #Params
        ID Ks thetaS thetaR m PsiB f As Au n ks Cs
        """
        if file_path is None:
            file_path = self.options["soiltablename"]["value"]

            if file_path is None:
                print(self.options["soiltablename"]["key_word"] + "is not specified.")
                return None

        soil_list = []

        with open(file_path, 'r') as file:
            lines = file.readlines()

        metadata = lines.pop(0)
        num_types, num_params = map(int, metadata.strip().split())
        param_standard = 12

        if num_params != param_standard:
            print(f"The number parameters in {file_path} do not conform with stand .sdt format.")
            return

        for l in lines:
            soil_info = l.strip().split()

            if len(soil_info) == param_standard:
                _id, ks, theta_s, theta_r, m, psi_b, f, a_s, a_u, n, _ks, c_s = soil_info
                station = {
                    "ID": _id,
                    "Ks": float(ks),
                    "thetaS": float(theta_s),
                    "thetaR": float(theta_r),
                    "m": float(m),
                    "PsiB": float(psi_b),
                    "f": float(f),
                    "As": float(a_s),
                    "Au": float(a_u),
                    "n": float(n),
                    "ks": float(_ks),
                    "Cs": float(c_s)
                }
                soil_list.append(station)

        if len(soil_list) != num_types:
            print("Error: Number of soil types does not match the specified count.")

        return soil_list
    @staticmethod
    def write_soil_table(soil_list,file_path):
        """
        Writes out Soil Reclassification Table(*.sdt) file with the following format:
        #Types #Params
        ID Ks thetaS thetaR m PsiB f As Au n ks Cs

        :param soil_list: List of dictionaries containing soil information specified by .sdt structure above.
        :param file_path: Path to save *.sdt file.
        """
        param_standard = 12

        with open(file_path, 'w') as file:
            # Write metadata line
            metadata = f"{len(soil_list)} {param_standard}\n"
            file.write(metadata)

            # Write station information
            for type in soil_list:
                line = f"{type['ID']}   {type['Ks']}    {type['thetaS']}    {type['thetaR']}    {type['m']}    {type['PsiB']}    " \
                       f"{type['f']}    {type['As']}    {type['Au']}    {type['n']}    {type['ks']}    {type['Cs']}\n"
                file.write(line)

    def read_landuse_table(self, file_path=None):
        """
        Returns list of dictionaries for each type of landuse specified in the .ldt file.

        Land Use Reclassification Table Structure (*.ldt, see tRIBS documentation for more details)
        #Types	#Params
        ID	a	b1	 P	S	K	b2	Al	 h	Kt	Rs	V LAI theta*_s theta*_t

        """
        if file_path is None:
            file_path = self.options["landtablename"]["value"]

            if file_path is None:
                print(self.options["landtablename"]["key_word"] + "is not specified.")
                return

        landuse_list = []

        with open(file_path, 'r') as file:
            lines = file.readlines()

        metadata = lines.pop(0)
        num_types, num_params = map(int, metadata.strip().split())
        param_standard = 15

        if num_params != param_standard:
            print(f"The number parameters in {file_path} do not conform with stand .sdt format.")
            return

        for l in lines:
            land_info = l.strip().split()

            if len(land_info) == param_standard:
                _id, a, b_1, p, s, k, b_2, al, h, kt, rs, v, lai, tstar_s, tstar_t = land_info
                station = {
                    "ID": _id,
                    "a": float(a),
                    "b1": float(b_1),
                    "P": float(p),
                    "S": float(s),
                    "K": float(k),
                    "b2": float(b_2),
                    "Al": float(al),
                    "h": float(h),
                    "Kt": float(kt),
                    "Rs": float(rs),
                    "V": float(v),
                    "LAI": float(lai),
                    "theta*_s": float(tstar_s),
                    "theta*_t": float(tstar_t)
                }
                landuse_list.append(station)

        if len(landuse_list) != num_types:
            print("Error: Number of soil types does not match the specified count.")

        return landuse_list
    @staticmethod
    def write_landuse_table(landuse_list,file_path):
        """
        Writes out Land Use Reclassification Table(*.ldt) file with the following format:
        #Types	#Params
        ID	a	b1	 P	S	K	b2	Al	 h	Kt	Rs	V LAI theta*_s theta*_t

        :param landuse_list: List of dictionaries containing soil information specified by .ldt structure above.
        :param file_path: Path to save *.sdt file.
        """
        param_standard = 15

        with open(file_path, 'w') as file:
            # Write metadata line
            metadata = f"{len(landuse_list)} {param_standard}\n"
            file.write(metadata)

            # Write station information
            for type in landuse_list:
                line = f"{type['ID']} {type['a']} {type['b1']} {type['P']} {type['S']} {type['K']} " \
                       f" {type['b2']} {type['Al']} {type['h']} {type['Kt']} {type['Rs']} {type['V']}" \
                       f" {type['LAI']} {type['theta*_s']} {type['theta*_t']}"
                file.write(line)

    def read_grid_data_file(self, grid_type):
        """
        Returns dictionary with content of a specified Grid Data File (.gdf)
        :param grid_type: string set to "weather", "soil", of "land", with each corresponding to HYDROMETGRID, SCGRID, LUGRID
        :return: dictionary containg keys and content: "Number of Parameters","Latitude", "Longitude","GMT Time Zone", "Parameters" (a  list of dicts)
        """

        if grid_type == "weather":
            option = self.options["hydrometgrid"]["value"]
        elif grid_type == "soil":
            option = self.options["scgrid"]["value"]
        elif grid_type == "land":
            option = self.options["lugrid"]["value"]

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
                    "Warning: The number of variables exceeds the number of parameters. This variable has been reset "
                    "in dictionary.")

        return {
            'Number of Parameters': variable_count,
            'Latitude': latitude,
            'Longitude': longitude,
            'GMT Time Zone': gmt_timezone,
            'Parameters': parameters
        }

    # def write_grid_data_file(self, grid_type):
    #     """
    #     Write out the content of a specified Grid Data File (.gdf)
    #     :param grid_type: string set to "weather", "soil", of "land", with each corresponding to HYDROMETGRID, SCGRID, LUGRID
    #     :return: dictionary containg keys and content: "Number of Parameters","Latitude", "Longitude","GMT Time Zone", "Parameters" (a  list of dicts)
    #     """
    #
    #     if grid_type == "weather":
    #         option = self.options["hydrometgrid"]["value"]
    #     elif grid_type == "soil":
    #         option = self.options["scgrid"]["value"]
    #     elif grid_type == "land":
    #         option = self.options["lugrid"]["value"]
    #
    #     parameters = []
    #
    #     with open(option, 'r') as file:
    #         num_parameters = int(file.readline().strip())
    #         location_info = file.readline().strip().split()
    #         latitude, longitude, gmt_timezone = location_info
    #
    #         variable_count = 0
    #
    #         for line in file:
    #             parts = line.strip().split()
    #             if len(parts) == 3:
    #                 variable_name, raster_path, raster_extension = parts
    #                 variable_count += 1
    #
    #                 path_components = raster_path.split(os.path.sep)
    #
    #                 # Exclude the last directory as its actually base name
    #                 raster_path = os.path.sep.join(path_components[:-1])
    #
    #                 if raster_path != "NO_DATA":
    #                     if not os.path.exists(raster_path):
    #                         print(
    #                             f"Warning: Raster file not found for Variable '{variable_name}': {raster_path}")
    #                         raster_path = None
    #                     elif os.path.getsize(raster_path) == 0:
    #                         print(
    #                             f"Warning: Raster file is empty for Variable '{variable_name}': {raster_path}")
    #                         raster_path = None
    #                 elif raster_path == "NO_DATA":
    #                     print(
    #                         f"Warning: No rasters set for variable '{variable_name}'")
    #                     raster_path = None
    #
    #                 parameters.append({
    #                     'Variable Name': variable_name,
    #                     'Raster Path': raster_path,
    #                     'Raster Extension': raster_extension
    #                 })
    #             else:
    #                 print(f"Skipping invalid line: {line}")
    #
    #         if variable_count > num_parameters:
    #             print(
    #                 "Warning: The number of variables exceeds the number of parameters. This variable has been reset "
    #                 "in dictionary.")
    #
    #     return {
    #         'Number of Parameters': variable_count,
    #         'Latitude': latitude,
    #         'Longitude': longitude,
    #         'GMT Time Zone': gmt_timezone,
    #         'Parameters': parameters
    #     }
    #

    @staticmethod
    def read_ascii(file_path):
        """
        Returns dictionary containing 'data', 'profile', and additional metadata.
        :param file_path: Path to ASCII (or other formats) raster.
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

    @staticmethod
    def read_json(file_path):
        with open(file_path, 'r') as f:
            input = json.load(f)
        return input

    @staticmethod
    def write_ascii(raster_dict, output_file_path,dtype='float32'):
        """
        Writes raster data and metadata from a dictionary to an ASCII raster file.
        :param raster_dict: Dictionary containing 'data', 'profile', and additional metadata.
        :param output_file_path: Output ASCII raster file path.
        """
        # Extract data and metadata from the dictionary
        data = raster_dict['data']
        profile = raster_dict['profile']

        # Remove unsupported creation options
        unsupported_options = ['blockxsize', 'blockysize', 'tiled', 'interleave']
        for option in unsupported_options:
            profile.pop(option, None)

        profile.update(dtype=dtype)



        # Check if the driver is set to 'AAIGrid' (ASCII format)
        if 'driver' not in profile or profile['driver'] != 'AAIGrid':
            # Update the profile for ASCII raster format
            profile.update(
                count=1,
                compress=None,
                driver='AAIGrid',  # ASCII Grid format
                nodata=-9999.0,
            )


        # Replace nan values with nodata value
        data = np.where(np.isnan(data), profile['nodata'], data)

        # Write the data and metadata to the ASCII raster file
        with rasterio.open(output_file_path, 'w', **profile) as dst:
            dst.write(data, 1)

        # ensure that header has the following format:
        # ncols
        # nrows
        # xllcorner
        # yllcorner
        # cellsize
        # NODATA_value

        with open(output_file_path, 'r') as file:
            lines = file.readlines()

        updated_lines = []
        replaced = False
        for line in lines:
            if line.startswith("dx") or line.startswith("dy"):
                if replaced == False:
                    updated_lines.append("cellsize"+ " " + str(math.ceil(float(line.split()[1]))) + "\n")
                    replaced = True
            else:
                updated_lines.append(line)
        # Write the updated content back to the file

        with open(output_file_path, 'w') as file:
            file.writelines(updated_lines)