import glob
import os
import re

import numpy as np
import pandas as pd


class Read():

    def get_qout_results(self):
        # currently only read for outlet, neet to add for hydronodelist
        qout_file = self.options["outhydrofilename"]["value"]+'_Outlet.qout'
        qout_df = pd.read_csv(qout_file, header=None, names=['Time_hr', 'Qstrm_m3s', 'Hlev_m'], skiprows=1, sep='\t')

        starting_date = self.options["startdate"]["value"]
        date = self.convert_to_datetime(starting_date)
        dt = pd.to_timedelta(qout_df['Time_hr'], unit='h')
        qout_df['Time'] = [date + step for step in dt]
        return qout_df
    def get_mrf_results(self, mrf_file=None):
        """

        """
        if mrf_file is None:
            runtime = self.options["runtime"]["value"]

            while len(runtime) < 4:
                runtime = '0' + runtime

            mrf_file = self.options["outhydrofilename"]["value"] + runtime + "_00.mrf"

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

        self.mrf['mrf'] = results_data_frame.iloc[0:int(self.options['runtime'][
                                                            'value']) - 1]  # This is becasue currently v5.2 tribs
        # has issues with mrf writing extra time steps

    def get_element_results(self):
        """
        Function assigns a dictionary to self as self.element_results. The keys in the dictionary represent a data
        frame with the content of the .invpixel file and each subsequent node. Key for .invpixel is "invar",
        and for nodes is simply the node ID. For each node this function reads in element file and create data frame
        of results and is assigned to the aforementioned dictionary.
        """
        # List to store first integers
        node_id = []

        # Path to the directory containing the files
        directory_path = self.options["outfilename"]["value"]
        # Find the last occurrence of '/'
        last_slash_index = directory_path.rfind('/')

        # Truncate the string at the last occurrence of '/'
        directory_path = directory_path[:last_slash_index + 1]

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
                self.element.update(
                    {first_integer: {'pixel': self.read_element_files(file_name), 'waterbalance': None}})

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

    def get_element_wb_dataframe(self, element_id):

        pixel = self.element[element_id]
        if isinstance(pixel, dict):
            pixel = pixel['pixel']  # waterbalance calc already called

        porosity = self.int_spatial_vars.loc[self.int_spatial_vars.ID == element_id, 'Porosity'].values[0]
        element_area = self.int_spatial_vars.loc[self.int_spatial_vars.ID == element_id, 'VAr'].values[0]

        df = pd.DataFrame({
            'Time': pixel['Time'],
            'Unsat_mm': pixel['Mu_mm'].values,
            'Sat_mm': pixel['Nwt_mm'].values * porosity,
            'CanopySWE_mm': 10 * pixel['IntSWEq_cm'].values,
            'SWE_mm': 10 * pixel['SnWE_cm'].values,
            'Canop_mm': pixel['CanStorage_mm'],
            'P_mm_h': pixel['Rain_mm_h'],
            'ET_mm_h': pixel['EvpTtrs_mm_h'] - (
                    pixel['SnSub_cm'] * 10 + pixel['SnEvap_cm'] * 10 + pixel['IntSub_cm'] * 10),
            'Qsurf_mm_h': pixel['Srf_Hour_mm'],
            'Qunsat_mm_h': pixel['QpIn_mm_h'] - pixel['QpOut_mm_h'],
            'Qsat_mm_h': pixel['GWflx_m3_h'] / element_area * 1000
        })

        return df

    def get_mrf_wb_dataframe(self):
        drainage_area = self.int_spatial_vars['VAr'].sum()  ## in m, TODO investigate why sum 'VAr' != max CAr ?
        weights = self.int_spatial_vars['VAr'].values / drainage_area
        porosity = np.sum(self.int_spatial_vars['Porosity'].values * weights)

        mrf = self.mrf['mrf']

        df = pd.DataFrame({
            'Time': mrf['Time'],
            'Unsat_mm': mrf['MSMU'].values * mrf['MDGW'].values * porosity,
            'Sat_mm': mrf['MDGW'] * porosity,
            'CanopySWE_mm': 10 * mrf['AvInSn'].values,
            'SWE_mm': 10 * mrf['AvSWE'].values,
            'Canop_mm': 0,  # not average canopy  storage
            'P_mm_h': mrf['MAP'],
            'ET_mm_h': mrf['MET'] - (mrf['AvSnSub'] * 10 + mrf['AvSnEvap'] * 10 + mrf['AvInSu'] * 10),
            'Qsurf_mm_h': mrf['Srf'] * 3600 * 1000 / drainage_area,
            'Qunsat_mm_h': 0,  # assumed zero, but not sure if correct?
            'Qsat_mm_h': 0  # assumed zero, but not sure if correct?
        })

        return df
