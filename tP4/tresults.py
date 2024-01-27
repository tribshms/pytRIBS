import glob
import os
import re

import pandas as pd

import tP4.results._post as _post
import tP4.results._waterbalance as _waterbalance
from tP4.mixins.infile_mixin import InfileMixin
from tP4.mixins.shared_mixin import SharedMixin


class Results(InfileMixin, SharedMixin):
    """
    A tRIBS Results Class.

    This class provides a framework for analyzing and visualizing individual tRIBS simulations. It takes an instance of
    Class Simulation and provides time-series and water balance analysis of the model results.


    Attributes:

    Methods:

    Example:
        Provide an example of how to create and use an instance of this class

    """

    def __init__(self, input_file, EPSG = None, UTM_Zone = None):
        # setup model paths and options for Result Class
        self.options = self.create_input_file()  # input options for tRIBS model run
        self.read_input_file(input_file)

        # attributes for analysis, plotting, and archiving model results
        self.element = {}
        self.mrf = {'mrf': None, 'waterbalance': None}
        self.geo = {"UTM_Zone": UTM_Zone, "EPSG": EPSG, "Projection": None}  # Geographic properties of tRIBS model domain.

        parallel_flag = int(self.options["parallelmode"]['value'])

        # read in integrated spatial vars for waterbalance calcs and spatial maps
        if parallel_flag == 1:
            self.int_spatial_vars = self.merge_parallel_spatial_files(suffix="_00i",
                                                                      dtime=int(self.options['runtime']['value']))
        elif parallel_flag == 0:
            runtime = int(self.options["runtime"]["value"])
            outfilename = self.options["outfilename"]["value"]
            intfile = f"{outfilename}.{runtime}_00i"

            self.int_spatial_vars = pd.read_csv(intfile)

        else:
            print('Unable To Read Integrated Spatial File (*_00i).')
            self.voronoi = None

        # read in voronoi files only once
        if parallel_flag == 1:
            self.voronoi = self.merge_parallel_voi()

        elif parallel_flag == 0:
            self.voronoi, _ = self.read_voi_file()

        else:
            print('Unable To Load Voi File(s).')
            self.voronoi = None

        self.watershed_stats = {"area": None, "avg_porosity": None, "avg_bedrock_depth": None}
        self.get_watershed_stats()

        #
        # # SIMULATION METHODS
        # def __getattr__(self, name):
        #     if name in self.options:
        #         return self.options[name]
        #     raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        #
        # def __dir__(self):
        #     # Include the keys from the options dictionary and the methods of the class
        #     return list(
        #         set(super().__dir__() + list(self.options.keys()))) if self.options is not None else super().__dir__()

    def get_watershed_stats(self):
        # drainage area
        self.watershed_stats['area'] = self.voronoi['geometry'].area.sum()

        # average porosity
        self.watershed_stats['avg_porosity'] = self.int_spatial_vars['Porosity'].mean()

        # average bedrock depth
        self.watershed_stats['avg_bedrock_depth'] = self.int_spatial_vars['Bedrock_Depth_mm'].mean()

    def get_mrf_results(self, mrf_file=None):
        """

        """
        if mrf_file is None:
            mrf_file = self.options["outhydrofilename"]["value"] + str(self.options["runtime"]["value"]) + "_00.mrf"

        # Read the first two rows to get column names and units
        with open(mrf_file, 'r') as file:
            column_names = file.readline().strip().split('\t')  # Assuming tab-separated data
            units = file.readline().strip().split('\t')  # Assuming tab-separated data

        # Read the data into a DataFrame, skipping the first two rows
        results_data_frame = pd.read_csv(mrf_file, skiprows=1, sep='\t')

        # Assign column names to the DataFrame
        results_data_frame.columns = column_names

        # Add units as metadata
        results_data_frame.attrs["units"] = units

        # # update time from hourly time step to date
        starting_date = self.options["startdate"]["value"]
        date = self.convert_to_datetime(starting_date)
        dt = pd.to_timedelta(results_data_frame['Time'], unit='h')
        results_data_frame['Time'] = [date + step for step in dt]

        self.mrf = results_data_frame
        return units

    def get_element_results(self):
        """
        Function assigns a dictionary to self as self.element_results. The keys in the dictionary represent a data
        frame with the content of the .invpixel file and each subsequent node. Key for .invpixel is "invar",
        and for nodes is simply the node ID. For each node this function reads in element file and create data frame
        of results and is assigned to the aforementioned dictionary. TODO:
        """

        pattern = r'(\D+)(\d+)\.pixel(?:\.(\d+))?\.?$'

        # List to store first integers
        node_id = []

        # Path to the directory containing the files
        directory_path = self.options["outfilename"]["value"]
        # Find the last occurrence of '/'
        last_slash_index = directory_path.rfind('/')

        # Truncate the string at the last occurrence of '/'
        directory_path = directory_path[:last_slash_index + 1]
        # read in .ivpixel if it exists
        try:
            invar_path = self.options["outfilename"]["value"] + ".ivpixel"
            invar_data_frame = pd.read_csv(invar_path, sep=r"\s+", header=0)
            self.element = {"invar": invar_data_frame}
        except FileNotFoundError:
            print("Invariant pixel files not found. Continuing to read in .pixel files")

        # Search for files matching the pattern
        if os.path.exists(directory_path):
            file_list = glob.glob(directory_path + '*.*')
        else:
            print('Cannot find results directory. Returning nothing.')
            return

        if len(file_list) == 0:
            print("Pixel files not found. Returning nothing.")
            return

        # Iterate through the files
        for file_name in file_list:
            file = file_name.split('/')[-1]
            match = re.search(r'(\d+)\.pixel', file)
            if match:
                print(f"Reading in: {file}")
                first_integer = int(match.group(1))
                node_id.append(first_integer)
                self.element.update({first_integer: self.read_element_files(file_name)})

    def read_element_files(self, element_results_file):
        """
        Reads in .pixel from tRIBS model results and updates hourly timestep to time
        """

        results_data_frame = pd.read_csv(element_results_file, sep=r"\s+", header=0)

        # update time from hourly time step to date
        starting_date = self.options["startdate"]["value"]
        date = self.convert_to_datetime(starting_date)
        dt = pd.to_timedelta(results_data_frame['Time_hr'], unit='h')
        results_data_frame['Time'] = [date + step for step in dt]

        return results_data_frame

    def get_mrf_water_balance(self, method):
        """
        """
        # need to get catchment averaged porosity, bedrock, drainage_area

        # drainage area from max CAr or watershed outline

        _waterbalance.get_mrf_water_balance(self, method)

    def get_element_water_balance(self, method, node_file=None):
        """
        This function loops through element_results and assigns water_balance as second item in the list for a given
        node/key. The user can specify a method for calculating the time frames over which the water balance is
        calculated.
        """
        _waterbalance.get_element_water_balance(self, method, node_file)
