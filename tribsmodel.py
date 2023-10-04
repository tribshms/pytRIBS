# tribsmodel.py

import numpy as np
import argparse
import pandas as pd
import json
import os
import subprocess
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class Model(object):
    """
    A tRIBS Model class.

    This class provides access to the underlying framework of a tRIBS (TIN-based Real-time Integrated Basin
    Simulator) simulation. It includes three nested classes: Preprocessing, Simulation, and Results. The Model class
    is initialized at the top-level to facilitate model setup, simultation, post-processing and can be used
    for mainpulating and generating multiple simulations in an efficient manner.

    The Preprocessing class allows for visualization, analysis, and bias correction of the data used in a given
    model. The Simulation class executes the tRIBS model and implements necessary recording of model steps,
    including preserving model logs, input files, and differences between the base model simulation. The Simulation
    class also has the Result class nested within it to allow quick and easy visualization and analysis of model
    results.

    Attributes:
        input_options (dict): A dictionary of the necessary keywords for a tRIBS .in file.
        model_input_file (str): Path to a template .in file with the specified paths for model results, inputs, etc.

    Methods:
         __init__(self, model_input_file):  # Constructor method
            :param model_input_file: Path to a template .in file.
            :type param1: str

        read_input_vars(self):
            Reads in templateVars file.

        get_input_var(self, var):
            Read variable specified by var from .in file.
            :param var: A keyword from the .in file.
            :type var: str
            :return: the line following the keyword argument from the .in file.
            :rtype: str

        read_node_list(self,file_path):
            Returns node list provide by .dat file.
            :param file_path: Relative or absolute file path to .dat file.
            :type file_path: str
            :return: List of nodes specified by .dat file
            :rtype: list

        def convert_to_datetime(starting_date):
            Returns a pandas date-time object.
            :param starting_date: The start date of a given model simulation, note needs to be in tRIBS format.
            :type starting_date: str
            :return: A pandas Timestamp object.

    Example:
        Provide an example of how to create and use an instance of this class
        >>> from tribsmodel import Model
        >>> m = Model('/path/to/.in')
    """

    def __init__(self, model_input_file):
        # fields
        self.input_options = None
        self.model_input_file = model_input_file
        self.read_input_vars()

        # nested classes
        self.Results = Results(self)

    # CONSTRUCTOR AND BASIC I/O FUNCTIONS
    def get_input_var(self, var):
        """
        Read variable specified by var from .in file.

        This function reads in the line following var, where var is required keyword or argument contained in the tRIBS
        .in input file.

        Parameters:
            var (str): a keyword or argument from the .in file. Alternatively the keyword can be past from input_options

        Examples:
             Example 1:
                >>> from tribsmodel import Model
                >>> m = Model('/path/to/.in')
                >>> m.get_input_var('STARTDATE:')
                    '06/01/2002/00/00'
             Example 2:
                >>> from tribsmodel import Model
                >>> s = Model('/path/to/.in')
                >>> s.get_input_var(s.input_options['parallelmode'])
                    '0'
        """
        try:
            file_path = self.model_input_file
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith(var):
                        var_out = next(f).strip()
                return var_out
        except FileNotFoundError:
            print("Error: '" + self.model_input_file + "' file not found.")

    def read_soil_table(self):
        pass

    def read_landuse_table(self):
        pass

    def read_node_list(self,file_path):
        """
        Returns node list provide by .dat file.

        The node list can be further modified or used for reading in element/pixel files and subsequent processing.

        :param file_path: Relative or absolute file path to .dat file.
        :type file_path: str
        :return: List of nodes specified by .dat file
        :rtype: list

        Example:
            >>> from tribsmodel import Model
            >>> m = Model('/path/to/.in')
            >>> node_list = m.read_node_list(m.input_options['nodeoutputlist'])
        """
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Initialize an empty list to store the IDs
            node_ids = []

            # Check if the file is empty or has invalid content
            if not lines:
                return node_ids

            # Parse the first column as the size of the array
            size = int(lines[0].strip())

            # Extract IDs from the remaining lines
            for line in lines[1:]:
                id_value = line.strip()
                node_ids.append(id_value)

            # Ensure the array has the specified size
            if len(node_ids) != size:
                print("Warning: Array size does not match the specified size in the file.")

            return node_ids
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return []

    @staticmethod
    def convert_to_datetime(starting_date):
        """
        Returns a pandas date-time object.

        :param starting_date: The start date of a given model simulation, note needs to be in tRIBS format.
        :type starting_date: str
        :rtupe: A pandas Timestamp object
        """
        month = int(starting_date[0:2])
        day = int(starting_date[3:5])
        year = int(starting_date[6:10])
        minute = int(starting_date[11:13])
        second = int(starting_date[14:16])
        date = pd.Timestamp(year=year, month=month, day=day, minute=minute)
        return date
    def read_input_vars(self):
        """
        Reads in dictionary from templateVars file.

        This function loads a dictionary of the necessary variables for a tRIBS input file. And is called upon
        initialization. The dictionary is assigned as instance variable:input_options to the Class Simulation. Note
        the templateVars file will need to be udpated if additional keywords are added to the .in file.

        Example:
            >>> from tribsmodel import Model
            >>> m = Model('/path/to/.in')
            >>> m.input_options
            {'startdate': 'STARTDATE:', 'runtime': 'RUNTIME:', 'rainsearch': 'RAINSEARCH:',...
        """
        self.input_options = {"startdate":"STARTDATE:",
        "runtime":{"key_word":"RUNTIME:","describe":"simulation length in hours"},
        "rainsearch":"RAINSEARCH:",
        "timestep":"TIMESTEP:",
        "gwstep":"GWSTEP:",
        "metstep":"METSTEP:",
        "etistep":"ETISTEP:",
        "rainintrvl":"RAININTRVL:",
        "opintrvl":"OPINTRVL:",
        "spopintrvl":"SPOPINTRVL:",
        "intstormmax":"INTSTORMMAX:",
        "baseflow":"BASEFLOW:",
        "velocitycoef":"VELOCITYCOEF:",
        "kinemvelcoef":"KINEMVELCOEF:",
        "velocityratio":"VELOCITYRATIO:",
        "flowexp":"FLOWEXP:",
        "channelroughness":"CHANNELROUGHNESS:",
        "channelwidth":"CHANNELWIDTH:",
        "channelwidthcoeff":"CHANNELWIDTHCOEFF:",
        "channelwidthexpnt":"CHANNELWIDTHEXPNT:",
        "channelwidthfile":"CHANNELWIDTHFILE:",
        "optmeshinput":"OPTMESHINPUT:",
        "rainsource":"RAINSOURCE:",
        "optevapotrans":"OPTEVAPOTRANS:",
        "hillalbopt":"HILLALBOPT:",
        "optradshelt":"OPTRADSHELT:",
        "optintercept":"OPTINTERCEPT:",
        "optlanduse":"OPTLANDUSE:",
        "optluinterp":"OPTLUINTERP:",
        "gfluxoption":"GFLUXOPTION:",
        "metdataoption":"METDATAOPTION:",
        "convertdata":"CONVERTDATA:",
        "optbedrock":"OPTBEDROCK:",
        "widthinterpolation":"WIDTHINTERPOLATION:",
        "optgwfile":"OPTGWFILE:",
        "optrunon":"OPTRUNON:",
        "optreservoir":"OPTRESERVOIR:",
        "optsoiltype":"OPTSOILTYPE:",
        "optspatial":"OPTSPATIAL:",
        "optgroundwater":"OPTGROUNDWATER:",
        "optinterhydro":"OPTINTERHYDRO:",
        "optheader":"OPTHEADER:",
        "optsnow":"OPTSNOW:",
        "inputdatafile":"INPUTDATAFILE:",
        "inputtime":"INPUTTIME:",
        "arcinfofilename":"ARCINFOFILENAME:",
        "pointfilename":"POINTFILENAME:",
        "soiltablename":"SOILTABLENAME:",
        "soilmapname":"SOILMAPNAME:",
        "landtablename":"LANDTABLENAME:",
        "landmapname":"LANDMAPNAME:",
        "gwaterfile":"GWATERFILE:",
        "demfile":"DEMFILE:",
        "rainfile":"RAINFILE:",
        "rainextension":"RAINEXTENSION:",
        "depthtobedrock":"DEPTHTOBEDROCK:",
        "bedrockfile":"BEDROCKFILE:",
        "lugrid":"LUGRID:",
        "tlinke":"TLINKE:",
        "minsntemp":"MINSNTEMP:",
        "snliqfrac":"SNLIQFRAC:",
        "templapse":"TEMPLAPSE:",
        "preclapse":"PRECLAPSE:",
        "hydrometstations":"HYDROMETSTATIONS:",
        "hydrometgrid":"HYDROMETGRID:",
        "hydrometconvert":"HYDROMETCONVERT:",
        "hydrometbasename":"HYDROMETBASENAME:",
        "gaugestations":"GAUGESTATIONS:",
        "gaugeconvert":"GAUGECONVERT:",
        "gaugebasename":"GAUGEBASENAME:",
        "outhydroextension":"OUTHYDROEXTENSION:",
        "ribshydoutput":"RIBSHYDOUTPUT:",
        "nodeoutputlist":"NODEOUTPUTLIST:",
        "hydronodelist":"HYDRONODELIST:",
        "outletnodelist":"OUTLETNODELIST:",
        "outfilename":"OUTFILENAME:",
        "outhydrofilename":"OUTHYDROFILENAME:",
        "forecastmode":"FORECASTMODE:",
        "forecasttime":"FORECASTTIME:",
        "forecastleadtime":"FORECASTLEADTIME:",
        "forecastlength":"FORECASTLENGTH:",
        "forecastfile":"FORECASTFILE:",
        "climatology":"CLIMATOLOGY:",
        "raindistribution":"RAINDISTRIBUTION:",
        "stochasticmode":"STOCHASTICMODE:",
        "pmean":"PMEAN:",
        "stdur":"STDUR:",
        "istdur":"ISTDUR:",
        "seed":"SEED:",
        "period":"PERIOD:",
        "maxpmean":"MAXPMEAN:",
        "maxstdurmn":"MAXSTDURMN:",
        "maxistdurmn":"MAXISTDURMN:",
        "weathertablename":"WEATHERTABLENAME:",
        "restartmode":"RESTARTMODE:",
        "restartintrvl":"RESTARTINTRVL:",
        "restartdir":"RESTARTDIR:",
        "restartfile":"RESTARTFILE:",
        "parallelmode":"PARALLELMODE:",
        "graphoption":"GRAPHOPTION:",
        "graphfile":"GRAPHFILE:",
        "optviz":"OPTVIZ:",
        "outvizfilename":"OUTVIZFILENAME:"
        }

class Simulation:
    """
    A tRIBS simulation class.

    This class provides a framework for running individual tRIBS model simulations. It takes an instance of Class Model
    and creates a subdirectory to store simulation information including, the individual model input files, simulation
    logs, and finally executes the tRIBS binary with specified conditions.


    Attributes:

    Methods:

    Example:
        Provide an example of how to create and use an instance of this class

    """
    def __init__(self, model_instance):
        self.model_instance = model_instance

    def run_simulation(self,executable,input_file,mpi_command=None,tribs_flags=None,log_path=None,store_input=None):
        """
        Run a tRIBS model simulation with optional arguments.

        Run_simulation assumes that if relative paths are used then the binary and input file are collocated in the
        same directory. That means for any keywords that depend on a relative path, must be specified from the directory
        the tRIBS bindary is executed. You can pass the location of the input file and executable as paths, in which case
        the function copies the binary and input file to same directory and then deletes both after the model run is complete.
        Optional arguments can be passed to store

        Args:
            binary_path (str): The path to the binary model executable.
            control_file_path (str): The path to the input control file for the binary.
            optional_args (str): Optional arguments to pass to the binary.

        Returns:
            int: The return code of the binary model simulation.
        """
        # Define the relative path to the binary or command
        relative_path = "../my_command"

        # Get the absolute path by resolving the relative path
        absolute_path = os.path.abspath(relative_path)

        # Now, you can use subprocess.run() with the absolute path
        result = subprocess.run([absolute_path, "arg1", "arg2"],

        # Verify that the binary and control file exist
        if not os.path.exists(binary_path):
            raise FileNotFoundError(f"Binary not found at {binary_path}")
        if not os.path.exists(control_file_path):
            raise FileNotFoundError(f"Control file not found at {control_file_path}")

        # Construct the command to run
        command = [binary_path, control_file_path] + list(optional_args)

        try:
            # Run the binary model simulation
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()

            # Wait for the process to finish and get the return code
            return_code = process.returncode

            # Print the stdout and stderr if needed
            if stdout:
                print("Standard Output:")
                print(stdout.decode())
            if stderr:
                print("Standard Error:")
                print(stderr.decode())

            return return_code

        except subprocess.CalledProcessError as e:
            print(f"Error running the binary: {e}")
            return e.returncode


# POST-PROCESSING NESTED CLASS: RESULTS
class Results:
    """
    A tRIBS Results Class.

    This class provides a framework for analyzing and visualizing individual tRIBS simulations. It takes an instance of
    Class Simulation and provides time-series and water balance analysis of the model results.


    Attributes:

    Methods:

    Example:
        Provide an example of how to create and use an instance of this class

    """
    def __init__(self, sim_instance):
        self.sim_instance = sim_instance
        self.element_results = None

    def setup_element_results(self):
        """
        Function assigns a dictionary to self as self.element_results. The keys in the dictionary represent a data
        frame with the content of the .invpixel file and each subsequent node. Key for .invpixel is "invar",
        and for nodes is simply the node ID. For each node this function reads in element file and create data frame
        of results and is assigned to the aforementioned dictionary.
        """
        # read in node list
        nodes = self.sim_instance.read_node_list()

        # read in .ivpixel
        invar_path = self.sim_instance.results_path + ".ivpixel"
        invar_data_frame = pd.read_csv(invar_path, sep=r"\s+", header=0)
        self.element_results = {"invar": invar_data_frame}

        # for each node read in element file read and set up a dictionary containing node
        for n in nodes:
            self.element_results.update({n: self.read_element_files(n)})

    def read_element_files(self, node):
        """
        Reads in .pixel from tRIBS model results and updates hourly timestep to time
        """
        element_results_file = self.sim_instance.results_path + str(
            node) + ".pixel"  # need to add catch for parallel runs and if file doesn't exist
        results_data_frame = pd.read_csv(element_results_file, sep=r"\s+", header=0)

        # update time from hourly time step to date
        starting_date = self.sim_instance.startdate
        date = self.sim_instance.convert_to_datetime(starting_date)
        dt = pd.to_timedelta(results_data_frame['Time_hr'], unit='h')
        results_data_frame['Time_hr'] = [date + step for step in dt]

        return results_data_frame

    def create_element_water_balance(self, method):
        """
        This function loops through element_results and assigns water_balance as second item in the list for a given
        node/key. The user can specify a method for calculating the time frames over which the water balance is
        calculated.
        """
        # read in node list
        nodes = self.sim_instance.read_node_list()
        invar_data_frame = self.element_results['invar']
        for n in nodes:
            porosity = invar_data_frame.Porosity[invar_data_frame.NodeID == int(n)].values[0]
            element_area = invar_data_frame.Area_m_sq[invar_data_frame.NodeID == int(n)].values[0]
            waterbalance = self.run_element_water_balance(self.element_results[n], porosity, element_area, method)
            self.element_results.update({n: [self.element_results[n], waterbalance]})

    # WATER BALANCE FUNCTIONS
    def run_element_water_balance(self, data, porosity, element_area, method):
        """
        creates a dictionary with water balance calculated specified in method: "water_year",  segments time by
        water year and discards results that do not start first or end on last water year in data. "year",
        just segments based on year, "month" segments base on month, and "cold_warm",segments on cold (Oct-April)
        and warm season (May-Sep).
        """

        begin, end, years = self.water_balance_dates(data, method)

        for n in range(0, len(years)):
            if n == 0:
                waterbalance = self.estimate_element_water_balance(data, begin[n], end[n], porosity, element_area)
            else:
                temp = self.estimate_element_water_balance(data, begin[n], end[n], porosity, element_area)

                for key, val in temp.items():

                    if key in waterbalance:
                        waterbalance[key] = np.append(waterbalance[key], val)

        return waterbalance, years

    @staticmethod
    def estimate_element_water_balance(element_data_frame, begin, end, porosity, element_area):
        """
        Computes water balance calculations for an individual computational element or node over a specified time frame. Data = pandas data
        frame of .pixel file, begin is start date, end is end date, bedrock depth is the depth to bedrock,
        porosity is well, porosity, and element area is surface area of voronoi polygon. Returns a dictionary with
        individual water components, keys with the prescript d indicate change in storage (i.e. delta) and n
        indicates net cumulative flux.
        """

        # logical index for calculating water balance
        begin_id = element_data_frame['Time_hr'].values == begin
        end_id = element_data_frame['Time_hr'].values == end
        duration_id = (element_data_frame['Time_hr'].values >= begin) & (
                element_data_frame['Time_hr'].values <= end)

        # return dictionary with values
        waterbalance = {}

        # Store ET flux as series due to complexity
        ET = element_data_frame['EvpTtrs_mm_h'] - (
                element_data_frame.SnSub_cm * 10 + element_data_frame.SnEvap_cm * 10 + element_data_frame.IntSub_cm * 10)  # Snow evaporation fluxes are subtracted due to signed behavior in snow module

        # calculate individual water balance components
        waterbalance.update(
            {'dUnsat': element_data_frame.Mu_mm.values[end_id][0] - element_data_frame.Mu_mm.values[begin_id][
                0]})  # [0] converts from array to float
        waterbalance.update(
            {'dSat': (element_data_frame.Nwt_mm.values[begin_id][0] - element_data_frame.Nwt_mm.values[end_id][
                0]) * porosity})
        waterbalance.update({'dCanopySWE': (10 * (
                element_data_frame.IntSWEq_cm.values[end_id][0] - element_data_frame.IntSWEq_cm.values[begin_id][
            0]))})  # convert from cm to mm
        waterbalance.update({'dSWE': (10 * (
                element_data_frame.SnWE_cm.values[end_id][0] - element_data_frame.SnWE_cm.values[begin_id][0]))})
        waterbalance.update({'dCanopy': element_data_frame.CanStorage_mm.values[end_id][0] -
                              element_data_frame.CanStorage_mm.values[begin_id][0]})
        waterbalance.update({'nP': np.sum(element_data_frame['Rain_mm_h'].values[duration_id])})
        waterbalance.update({'nET': np.sum(ET.values[duration_id])})
        waterbalance.update({'nQsurf': np.sum(element_data_frame['Srf_Hour_mm'].values[duration_id])})
        waterbalance.update(
            {'nQunsat': np.sum(element_data_frame['QpIn_mm_h'].values[duration_id]) - np.sum(
                element_data_frame['QpOut_mm_h'].values[duration_id])})
        waterbalance.update(
            {'nQsat': np.sum(
                element_data_frame['GWflx_m3_h'].values[
                    duration_id]) / element_area * 1000})  # convert from m^3/h to mm/h

        return waterbalance

    @staticmethod
    def water_balance_dates(results_data_frame, method):
        """
        data = pandas data frame of .pixel file, methods select approach for segmenting data['Time_hr'].
        "water_year", segments time frame by water year and discards results that do not start first or end on
        last water year in data. "year", just segments based on year, "month" segments base on month,
        and "cold_warm",segments on cold (Oct-April) and warm season (May-Sep).
        """

        min_date = min(results_data_frame.Time_hr)
        max_date = max(results_data_frame.Time_hr)

        if method == "water_year":
            years = np.arange(min_date.year, max_date.year)
            begin_dates = [pd.Timestamp(year=x, month=10, day=1, hour=0, minute=0, second=0) for x in years]
            years += 1
            end_dates = [pd.Timestamp(year=x, month=9, day=30, hour=23, minute=0, second=0) for x in years]

            # make sure water years are in data set
            while begin_dates[0] < min_date:
                begin_dates.pop(0)
                end_dates.pop(0)

            while end_dates[len(end_dates) - 1] > max_date:
                begin_dates.pop(len(end_dates) - 1)
                end_dates.pop(len(end_dates) - 1)

        if method == "year":
            years = np.arange(min_date.year, max_date.year)
            begin_dates = [pd.Timestamp(year=x, month=1, day=1, hour=0, minute=0, second=0) for x in years]
            end_dates = [pd.Timestamp(year=x, month=12, day=31, hour=23, minute=0, second=0) for x in years]

            # adjust start date according to min_date
            begin_dates[0] = min_date

            # add ending date according to end_date
            end_dates.append(max_date)

            # add last year to years
            years = np.append(years, max_date.year)

        if method == "cold_warm":
            years = np.arange(min_date.year, max_date.year + 1)
            begin_dates = [[pd.Timestamp(year=x, month=5, day=1, hour=0, minute=0, second=0),
                            pd.Timestamp(year=x, month=10, day=1, hour=0, minute=0, second=0)] for x in years]
            begin_dates = [date for sublist in begin_dates for date in sublist]
            end_dates = [[pd.Timestamp(year=x, month=9, day=30, hour=23, minute=0, second=0),
                          pd.Timestamp(year=x + 1, month=4, day=30, hour=23, minute=0, second=0)] for x in years]
            end_dates = [date for sublist in end_dates for date in sublist]

            # make sure season are in data set
            while begin_dates[0] < min_date:
                begin_dates.pop(0)
                end_dates.pop(0)

            while end_dates[len(end_dates) - 1] > max_date:
                begin_dates.pop(len(end_dates) - 1)
                end_dates.pop(len(end_dates) - 1)

        # Update date time to reflect middle of period over which the waterbalance is calculated
        years = [x + (y - x) / 2 for x, y in zip(begin_dates, end_dates)]
        return begin_dates, end_dates, years


