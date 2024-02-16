import os
import numpy as np
from rosetta import rosetta, SoilData

from pytRIBS.model.inout import InOut


### PREPROCESSIN
# 1 delineate watersheds (pyshed)
# 1.creat points file for hydrological conditioned tin Mesh
# 2 ingest soil and landuse (pyshed)
# 3 apply pedo transfer functions
# others?

class Preprocess(InOut):
    """

    """

    # coding=utf-8
    # This script computes soil textural class using the USDA Soil Triangle and soil parameters required for tRIBS
    # using the Rosetta Python package.
    # Manual modifications will need to be to adjust how rasters are read in if script is applied elsewhere,
    # unit conversions, number of soil properties fed into Rosetta, and assumptions used.
    # Josh Cederstrom October 2022

    # Compute Soil Texture Class
    @staticmethod
    def soiltexturalclass(sand, clay):
        """
        Returns USDA soil textural class given percent sand and clay.
        :param sand:
        :param clay: Location to write input file to.
        """
        silt = 100 - sand - clay

        if sand + clay > 100 or sand < 0 or clay < 0:
            raise ValueError('Inputs add up to more than 100% or are negative')
        elif silt + 1.5 * clay < 15:
            textural_class = 1  # sand
        elif silt + 1.5 * clay >= 15 and silt + 2 * clay < 30:
            textural_class = 2  # loamy sand
        elif (clay >= 7 and clay < 20 and sand > 52 and silt + 2 * clay >= 30) or (
                clay < 7 and silt < 50 and silt + 2 * clay >= 30):
            textural_class = 3  # sandy loam
        elif clay >= 7 and clay < 27 and silt >= 28 and silt < 50 and sand <= 52:
            textural_class = 4  # loam
        elif (silt >= 50 and clay >= 12 and clay < 27) or (silt >= 50 and silt < 80 and clay < 12):
            textural_class = 5  # silt loam
        elif silt >= 80 and clay < 12:
            textural_class = 6  # silt
        elif clay >= 20 and clay < 35 and silt < 28 and sand > 45:
            textural_class = 7  # 'sandy clay loam'
        elif clay >= 27 and clay < 40 and sand > 20 and sand <= 45:
            textural_class = 8  # 'clay loam'
        elif clay >= 27 and clay < 40 and sand <= 20:
            textural_class = 9  # 'silty clay loam'
        elif clay >= 35 and sand > 45:
            textural_class = 10  # 'sandy clay'
        elif clay >= 40 and silt >= 40:
            textural_class = 11  # 'silty clay'
        elif clay >= 40 > silt and sand <= 45:
            textural_class = 12
        else:
            textural_class = 'na'

        return textural_class

    def process_raw_soil(self, grid_input, output=None):
        """
        Writes ascii grids Ks, theta_s, theta_r, psib, and m from gridded soil data for % sand, silt, clay, bulk density, and volumetric water content at 33 and 1500 kPa.

        Parameters:
        - grid_input (list of dict or str): If a dictionary list, keys are "grid_type" and "path" for each soil property.
                                    Format of dictionary list follows:
                                    [{'type':'sand_fraction', 'path':'path/to/grid'},
                                    {'type':'silt_fraction', 'path':'path/to/grid'},
                                    {'type':'clay_fraction', 'path':'path/to/grid'},
                                    {'type':'bulk_density', 'path':'path/to/grid'},
                                    {'type':'vwc_33', 'path':'path/to/grid'},
                                    {'type':'vwc_1500', 'path':'path/to/grid'}]

                                    If a string is provided, it is treated as the path to a configuration file. The
                                    configuration file must be written in json format.

        - output (list, optional): List of output file names for different soil properties.

        Note:
        - The 'grid_types' key should contain a list of dictionaries, each specifying a grid type and its corresponding file path.
        - The 'output_files' key should contain a list of exactly 5 output file names for different soil properties.
        - The file paths in 'grid_types' should be valid, and the 'output_files' list should have the correct size.
        """

        # Check if grid_input is a string (path to a config file)
        if isinstance(grid_input, str):
            # Read configuration from the file
            config = self.read_config(grid_input)
            grids = config['grid_types']
            output_files = config['output_files']
        elif isinstance(grid_input, list):
            # Use provided dictionary
            grids = grid_input
            output_files = output or ['Ks.asc', 'theta_r.asc', 'theta_s.asc', 'psib.asc', 'm.asc']
        else:
            print('Invalid input format. Provide either a dictionary or a path to a configuration file.')
            return

        # Check if each file specified in the dictionary or config exists
        for g in grids:
            grid_type, path = g['type'], g['path']
            if not os.path.isfile(path):
                raise FileNotFoundError(f'Cannot find: {path} for grid type: {grid_type}')

        # Ensure output_files is a list of the correct size (5 elements)
        if output_files is not None and (not isinstance(output_files, list) or len(output_files) != 5):
            print('Output must be a list with 5 elements.')
            return

        sg250_data = None
        size = None
        geo_tiff = None

        # Loop through specified file paths
        for cnt,g in enumerate(grids):
            grid_type, path = g['type'], g['path']
            print(f"Ingesting {grid_type} from: {path}")
            geo_tiff = self.read_ascii(path)
            array = geo_tiff['data']
            size = array.shape

            if cnt == 0:
                sg250_data = np.zeros((6, size[0], size[1]))

            # each z layer follows:[sa (%), si (%), cl (%), bd (g/cm3), th33, th1500]
            if grid_type == 'sand_fraction':
                array = array / 1000 * 100  # convert SSC from g/kg to % SSC
                sg250_data[0, :, :] = array
            elif grid_type == 'silt_fraction':
                array = array / 1000 * 100  # convert SSC from g/kg to % SSC
                sg250_data[1, :, :] = array
            elif grid_type == 'clay_fraction':
                array = array / 1000 * 100  # convert SSC from g/kg to % SSC
                sg250_data[2, :, :] = array
            elif grid_type == 'bulk_density':
                array = array / 100  # convert bulk density from cg/cm3 to g/cm3
                sg250_data[3, :, :] = array
            elif grid_type == 'vwc_33':
                array = array / 100  # convert bulk density from cg/cm3 to g/cm3
                sg250_data[4, :, :] = array
            elif grid_type == 'vwc_1500':
                array = array / 100  # convert bulk density from cg/cm3 to g/cm3
                sg250_data[5, :, :] = array

        profile = geo_tiff['profile']


        # Initialize parameter grids, 3 grids - 1 mean values, 2 std deviations, 3 code/flag
        theta_r, theta_s, ks, psib, m = np.zeros((3, *size)), np.zeros((3, *size)), np.zeros((3, *size)), np.zeros(
            (1, *size)), np.zeros((1, *size))

        # Loop through raster's and compute soil properties using rosetta-soil package
        # Codes/Flags
        # 2	sand, silt, clay (SSC)
        # 3	SSC + bulk density (BD)
        # 4	SSC + BD + field capacity water content (TH33)
        # 5	SSC + BD + TH33 + wilting point water content (TH1500)
        # -1 no result returned, inadequate or erroneous data
        # each z layer follows:[sa (%), si (%), cl (%), bd (g/cm3), th33, th1500]
        # i.e SoilData([sa (%), si (%), cl (%), bd (g/cm3), th33, th1500])

        for i in range(0, size[0]):
            for j in range(0, size[1]):
                # Organize array for input into packag
                data = [sg250_data[x, i, j] for x in np.arange(0,6)]
                soil_data = SoilData.from_array(data)
                mean, stdev, codes = rosetta(3, soil_data)  # apply Rosetta version 3
                theta_r[:, i, j] = [mean[0, 0], stdev[0, 0], codes[0]]
                theta_s[:, i, j] = [mean[0, 1], stdev[0, 1], codes[0]]
                # Convert ks from log10(cm/day) into mm/hr
                ks[:, i, j] = [(10 ** mean[0, 4]) * 10 / 24, (10 ** stdev[0, 4]) * 10 / 24, codes[0]]

                # Alpha parameter from rosetta corresponds approximately to the inverse of the air-entry value, cm−1
                # https://doi.org/10.1029/2019MS001784
                # Convert from log10(cm) into -1/mm
                psib[0, i, j] = -1 / ((10 ** mean[0, 2]) * 10)

                # Pore-size Distribution can be calculated from n using m = 1-1/n
                # http://dx.doi.org/10.4236/ojss.2012.23025
                # Convert from log10(n) into n
                m[0, i, j] = 1 - (1 / (10 ** mean[0, 3]))

        soil_prop = [ks, theta_r, theta_s, psib, m]

        for soil_property, name in zip(soil_prop, output_files):
            temp = {'data': soil_property, 'profile': profile}
            self.write_ascii(temp, name)
