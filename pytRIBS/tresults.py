import glob
import os
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pytRIBS.mixins.infile_mixin import InfileMixin
from pytRIBS.mixins.shared_mixin import SharedMixin
from pytRIBS.results.waterbalance import WaterBalance


class Results(InfileMixin, SharedMixin, WaterBalance):
    """
    A tRIBS Results Class.

    This class provides a framework for analyzing and visualizing individual tRIBS simulations. It takes an instance of
    Class Simulation and provides time-series and water balance analysis of the model results.


    Attributes:

    Methods:

    Example:
        Provide an example of how to create and use an instance of this class
    """

    def __init__(self, input_file, EPSG=None, UTM_Zone=None):
        # setup model paths and options for Result Class
        self.options = self.create_input_file()  # input options for tRIBS model run
        self.read_input_file(input_file)

        # attributes for analysis, plotting, and archiving model results
        self.element = {}
        self.mrf = {'mrf': None, 'waterbalance': None}
        self.geo = {"UTM_Zone": UTM_Zone, "EPSG": EPSG,
                    "Projection": None}  # Geographic properties of tRIBS model domain.

        self.get_invariant_properties()  # shared

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

        self.mrf['mrf'] = results_data_frame

    def get_element_results(self):
        """
        Function assigns a dictionary to self as self.element_results. The keys in the dictionary represent a data
        frame with the content of the .invpixel file and each subsequent node. Key for .invpixel is "invar",
        and for nodes is simply the node ID. For each node this function reads in element file and create data frame
        of results and is assigned to the aforementioned dictionary. TODO:
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

    @staticmethod
    def plot_water_balance(waterbalance, saved_fig=None):
        """

        :param saved_fig:
       :param waterbalance:
       :return:
       """

        # plt.style.use('bmh')
        barwidth = 0.25
        fig, ax = plt.subplots()

        ax.bar(np.arange(len(waterbalance)) + barwidth, waterbalance['nP'], align='center', width=barwidth,
               color='grey', label='nP')
        rects = ax.patches

        # Make some labels.
        labels = ["%.0f" % (p - waterbalance) for p, waterbalance in
                  zip(waterbalance['nP'], waterbalance['dS'] + waterbalance['nQ'] + waterbalance['nET'])]
        netdiff = [p - waterbalance for p, waterbalance in
                   zip(waterbalance['nP'], waterbalance['dS'] + waterbalance['nQ'] + waterbalance['nET'])]

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom"
            )

        ax.text(len(waterbalance.index), max(waterbalance.nP), "mean difference: " + "%.0f" % np.mean(netdiff))

        waterbalance.plot.bar(ax=ax, y=["nQ", "nET", "dS"], stacked=True, width=barwidth,
                              color=['tab:blue', 'tab:red', 'tab:cyan'])
        ax.legend(bbox_to_anchor=(1.35, 0.85), loc='center right',
                  labels=["Precip.", "Runoff", "Evapo. Trans.", "$\Delta$ Storage"])
        ax.set_ylabel("Water Flux & $\Delta$ Storage (mm)")
        ax.set_xticks(range(len(waterbalance.index)), waterbalance.index.strftime("%Y-%m"), rotation=45)
        fig.autofmt_xdate()
        plt.show()

        if saved_fig is not None:
            plt.savefig(saved_fig, bbox_inches='tight')

        return fig, ax

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

    def create_animation(self, outfile, df_dict, frames, var, fps=4, vlims=None, nan_color='gray',
                         nan_edge_color='red', cmap='viridis'):
        """
        Create and save an animation based on a dictionary of DataFrames of tRIBS dynamic files.

        Parameters:
            outfile (str): The file path for saving the animation, format is determined from file extension (.mp4,.gif,.avi,.html).
            df_dict (dict): A dictionary where keys represent animation frames and values are DataFrames to be plotted.
            frames (iterable): Iterable containing keys from df_dict representing the frames to include in the animation.
            var (str): The column name in DataFrames to be plotted.
            fps (int, optional): Frames per second for the animation (default is 4).
            vlims (tuple, optional): Tuple containing minimum and maximum values for color normalization (default is None).
            nan_color (str, optional): Color for NaN values in the plot (default is 'gray').
            nan_edge_color (str, optional): Edge color for NaN values in the plot (default is 'red').

        Returns:
            None

        Raises:
            ValueError: If outfile is not a valid file path or frames is empty.

        Notes:
            - This method creates an animation by iterating over frames specified in the frames parameter.
            - Each frame corresponds to a key in the df_dict dictionary, and the corresponding DataFrame is plotted.
            - NaN values in the DataFrame are flagged with the specified nan_color and nan_edge_color.
            - The animation format is dependent on the outfile extension with the specified frames per second (fps).

        Example:
            # Assuming instance is an instance of the class containing create_animation method
            instance.create_animation("animation.gif", df_dict, frames=['0','1','2','3'], var="ET", fps=10)
        """

        def update_plot(key, ax, cax):
            """
            Update the plot for each frame in the animation.

            Parameters:
                key: The key representing the current frame in df_dict.
                ax: The main axes object for the plot.
                cax: The colorbar axes object.
                df_dict: A dictionary containing DataFrames for each frame.
                results_class: An instance of the class containing the voronoi attribute.
                var: The variable to be plotted from DataFrames.
                vlims: Tuple containing minimum and maximum values for color normalization.

            Returns:
                None
            """
            ax.clear()

            df = df_dict[key]
            gdf = self.voronoi.copy()
            gdf = gdf.merge(df, on="ID", how="inner")

            if vlims is not None:
                gdf.plot(ax=ax, column=var, legend=True, cmap=cmap, vmin=min(vlims), vmax=max(vlims), cax=cax)
            else:
                gdf.plot(ax=ax, column=var, legend=True, cmap=cmap, cax=cax)

            # flag and plot nans
            if len(gdf[gdf[var].isnull()]) != 0:
                gdf[gdf[var].isnull()].plot(ax=ax, color=nan_color, edgecolor=nan_edge_color)

            ax.set_title(f'{var}: {key}')

            plt.axis('off')
            plt.xticks([])  # Remove x-axis ticks and labels
            plt.yticks([])  # Remove y-axis ticks and labels

        # Create a figure and axis
        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        # Create the animation
        animation = FuncAnimation(fig, update_plot, frames=frames, fargs=(ax, cax),
                                  repeat=False)

        # To save the animation as a GIF
        animation.save(outfile, fps=fps)

        plt.show()
