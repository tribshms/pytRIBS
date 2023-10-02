# tribs_sim.py

import numpy as np
import argparse
import pandas as pd
import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class Simulation(object):
    """
    This package create a python class that represents a model simulation for the distributed hydrological model:
    TIN-based Real-Time Integrated Basin Simulator (i.e. tRIBS), with the intention of increasing the overall efficiency
    of post-and-pre model simulation processing as well as calibrating and experimenting with model behavior and finally
    provides the useful integration of tRIBS with python functionality including but not limited to education and
    programming tools like Jupyter Lab.
    """

    def __init__(self, run_id, simulation_control_file):
        # fields
        self.input_options = None
        self.simulation_id = run_id
        self.simulation_control_file = simulation_control_file
        self.read_input_vars()
        var = self.input_options['startdate']
        self.startdate = self.get_input_var(var)
        var = self.input_options['outfilename']
        self.results_path = self.get_input_var(var)

        # nested classes
        self.Results = Results(self)

    # CONSTRUCTOR AND BASIC I/O FUNCTIONS
    def read_input_vars(self):
        """
        Reads in templateVars file: which contains a dictionary of the necessary variables for a tRIBS input file.
        templateVars will need to be updated in the event new features area added to tRIBS and require manipulation via
        the input file.
        """
        try:
            with open('templateVars') as f:
                data = f.read()
            self.input_options = json.loads(data)
        except FileNotFoundError:
            # Handle the case where the file doesn't exist
            print("Error: 'templateVars' file not found.\n input_options set as empty dictionary")
            self.input_options = {}

    def get_input_var(self, var):
        """
        Reads the .in file used for model simulation, and obtains the starting date of the simulation
        """
        try:
            file_path = self.simulation_control_file
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith(var):
                        var_out = next(f).strip()
                return var_out
        except FileNotFoundError:
            print("Error: '" + self.simulation_control_file + "' file not found.")

    def read_soil_table(self):
        pass

    def read_landuse_table(self):
        pass

    def read_node_list(self):
        """
        This function reads in the node list provide by the oNode.dat file. This is used for reading in element/pixel
        files and subsequent processing.
        """
        var = self.input_options['outletnodelist']
        file_path = self.get_input_var(var)
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
        month = int(starting_date[0:2])
        day = int(starting_date[3:5])
        year = int(starting_date[6:10])
        minute = int(starting_date[11:13])
        second = int(starting_date[14:16])
        date = pd.Timestamp(year=year, month=month, day=day, minute=minute)
        return date

    # POST-PROCESSING NESTED CLASS: RESULTS


class Results:
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
                waterbalance = self.water_balance_element(data, begin[n], end[n], porosity, element_area)
            else:
                temp = self.water_balance_element(data, begin[n], end[n], porosity, element_area)

                for key, val in temp.items():

                    if key in waterbalance:
                        waterbalance[key] = np.append(waterbalance[key], val)

        return waterbalance, years

    @staticmethod
    def water_balance_element(element_data_frame, begin, end, porosity, element_area):
        """
        Computes water balance calculations for an individual computational element or node. Data = pandas data
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


