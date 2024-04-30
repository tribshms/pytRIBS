import re
import os

import numpy as np
from owslib.wcs import WebCoverageService
from rosetta import rosetta, SoilData
from scipy.optimize import curve_fit
from pytRIBS.shared.inout import InOut
from pytRIBS.shared.aux import Aux


class _Soil:
    # Assigning references to the methods
    write_ascii = InOut.write_ascii
    read_ascii = InOut.read_ascii
    read_json = InOut.read_json

    @staticmethod
    def fillnodata(files, overwrite=False, **kwargs):
        Aux.fillnodata(files, overwrite=overwrite, **kwargs)

    @staticmethod
    def write_ascii(files, overwrite=False, **kwargs):
        InOut.write_ascii(files, overwrite=overwrite, **kwargs)

    @staticmethod
    def read_ascii(file_path):
        raster = InOut.read_ascii(file_path)
        return raster

    @staticmethod
    def read_json(raster_dict, output_file_path, dtype='float32'):
        input = InOut.read_json(raster_dict, output_file_path, dtype)
        return input

    def read_soil_table(self, textures=False, file_path=None):
        """
        Soil Reclassification Table Structure (*.sdt)
        #Types #Params
        ID Ks thetaS thetaR m PsiB f As Au n ks Cs
        an o
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

        if textures:
            param_standard += 1

        if num_params != param_standard:
            print(f"The number parameters in {file_path} do not conform with standard soil .sdt format.")
            return

        for l in lines:
            soil_info = l.strip().split()

            if len(soil_info) == param_standard:
                if textures:
                    _id, ks, theta_s, theta_r, m, psi_b, f, a_s, a_u, n, _ks, c_s, textures = soil_info
                    station = {
                        "ID": _id,
                        "Ks": ks,
                        "thetaS": theta_s,
                        "thetaR": theta_r,
                        "m": m,
                        "PsiB": psi_b,
                        "f": f,
                        "As": a_s,
                        "Au": a_u,
                        "n": n,
                        "ks": _ks,
                        "Cs": c_s,
                        "Texture": textures
                    }
                else:
                    _id, ks, theta_s, theta_r, m, psi_b, f, a_s, a_u, n, _ks, c_s = soil_info
                    station = {
                        "ID": _id,
                        "Ks": ks,
                        "thetaS": theta_s,
                        "thetaR": theta_r,
                        "m": m,
                        "PsiB": psi_b,
                        "f": f,
                        "As": a_s,
                        "Au": a_u,
                        "n": n,
                        "ks": _ks,
                        "Cs": c_s
                    }

                soil_list.append(station)

        if len(soil_list) != num_types:
            print("Error: Number of soil types does not match the specified count.")
        return soil_list

    @staticmethod
    def write_soil_table(soil_list, file_path, textures=False):
        """
        Writes out Soil Reclassification Table(*.sdt) file with the following format:
        #Types #Params
        ID Ks thetaS thetaR m PsiB f As Au n ks Cs

        :param soil_list: List of dictionaries containing soil information specified by .sdt structure above.
        :param file_path: Path to save *.sdt file.
        """
        param_standard = 12

        if textures:
            param_standard += 1

        with open(file_path, 'w') as file:
            # Write metadata line
            metadata = f"{len(soil_list)} {param_standard}\n"
            file.write(metadata)

            # Write station information
            for type in soil_list:

                if textures:
                    line = f"{str(type['ID'])}   {str(type['Ks'])}    {str(type['thetaS'])}    {str(type['thetaR'])}    {str(type['m'])}    {str(type['PsiB'])}    " \
                           f"{str(type['f'])}    {str(type['As'])}    {str(type['Au'])}    {str(type['n'])}    {str(type['ks'])}    {str(type['Cs'])} {str(type['Texture'])}\n"
                else:
                    line = f"{str(type['ID'])}   {str(type['Ks'])}    {str(type['thetaS'])}    {str(type['thetaR'])}    {str(type['m'])}    {str(type['PsiB'])}    " \
                           f"{str(type['f'])}    {str(type['As'])}    {str(type['Au'])}    {str(type['n'])}    {str(type['ks'])}    {str(type['Cs'])}\n"

                file.write(line)

    def get_soil_grids(self, bbox, depths, soil_vars, stats, replace=False):
        """
        Retrieves soil data from ISRIC WCS service and saves it as GeoTIFFs and returns list of paths to downloaded files.

        Args:
            bbox (list): The bounding box coordinates in the format [x1, y1, x2, y2].
            depths (list): List of soil depths to retrieve data for.
            soil_vars (list): List of soil variables to retrieve.
            stats (list): List of statistics to compute for each variable and depth.

        Example:
            bbox = [387198, 3882394, 412385, 3901885]  # x1,y1,x2,y2
            depths = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm']
            soil_vars = ['bdod', 'clay', 'sand', 'silt', 'wv1500', 'wv0033', 'wv0010'], see https://maps.isric.org/
            stats = ['mean'], see prediciton quantiles at https://www.isric.org/explore/soilgrids/faq-soilgrids
        """
        epsg = self.geo['EPSG']

        if epsg is None:
            print("No EPSG code found. Please update model attribute .geo['EPSG'] with EPSG code.")
            return

        complete = False

        match = re.search(r'EPSG:(\d+)', epsg)
        if match:
            epsg = match.group(1)

        data_dir = 'sg250'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Directory '{data_dir}' created.")

        os.chdir(data_dir)

        crs = f'urn:ogc:def:crs:EPSG::{epsg}'

        files = []

        print('Downloading data, this may take several minutes or more...')
        for var in soil_vars:
            wcs = WebCoverageService(f'http://maps.isric.org/mapserv?map=/map/{var}.map', version='1.0.0', timeout=300)
            for depth in depths:

                # for a given variable, depth, and stat write out a geotif
                for stat in stats:
                    soil_key = f'{var}_{depth}_{stat}'
                    file = f'{soil_key}.tif'
                    files.append(file)

                    if (os.path.isfile(file) and replace == True) or not os.path.isfile(file):
                        response = wcs.getCoverage(identifier=soil_key, crs=crs, bbox=bbox, resx=250, resy=250,
                                                   format='GEOTIFF_INT16', timeout=120)
                        with open(file, 'wb') as file:
                            file.write(response.read())
                        complete = True

        if complete:
            print('Download of SoilGrids250 data complete')
        else:
            print('No SoilGrids250 data was downloaded check inputs or set replace == False.')

        os.chdir('..')
        return files

    def create_soil_map(self, grid_input, output=None):
        """
        Parameters:
        - grid_input (list of dict or str): If a dictionary list, keys are "grid_type" and "path" for each soil property.
                                    Format of dictionary list follows:
                                    [{'type':'sand', 'path':'path/to/grid'},
                                    {'type':'clay', 'path':'path/to/grid'},
        """

        if isinstance(grid_input, str):
            # Read configuration from the file
            config = self.read_json(grid_input)
            grids = config['grid_types']
            output_file = config['output_files']
        elif isinstance(grid_input, list):
            # Use provided dictionary
            grids = grid_input
            output_file = output or ['soil_class.soi']

        def soiltexturalclass(sand, clay):
            """
            Returns USDA soil textural class given percent sand and clay.
            :param sand:
            :param clay:
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

        # Loop through specified file paths
        texture_data = []

        for cnt, g in enumerate(grids):
            grid_type, path = g['type'], g['path']
            print(f"Ingesting {grid_type} from: {path}")
            geo_tiff = self.read_ascii(path)
            array = geo_tiff['data']
            size = array.shape

            if cnt == 0:
                texture_data = np.zeros((2, size[0], size[1]))

            if grid_type == 'sand':
                array = array / 1000 * 100  # convert SSC from g/kg to % SSC
                texture_data[0, :, :] = array
            elif grid_type == 'clay':
                array = array / 1000 * 100  # convert SSC from g/kg to % SSC
                texture_data[1, :, :] = array

        soil_class = np.zeros((1, size[0], size[1]), dtype=int)

        for i in range(0, size[0]):
            for j in range(0, size[1]):
                # Organize array for input into packag
                data = [texture_data[x, i, j] for x in np.arange(0, 2)]
                sand = data[0]
                clay = data[1]
                soil_class[0, i, j] = int(soiltexturalclass(sand, clay))

        soil_classification = {1: 'sand', 2: 'loamy_sand', 3: 'sandy_loam', 4: 'loam', 5: 'silt_loam', 6: 'silt',
                               7: 'sandy_clay_loam', 8: 'clay_loam', 9: 'silty_clay_loam', 10: 'sandy_clay',
                               11: 'silty_clay', 12: 'clay'}

        classes = np.unique(soil_class[0])

        filtered_classes = {}

        count = 1
        for key in soil_classification.keys():
            if key in classes:
                filtered_classes[count] = soil_classification[key]
                soil_class[soil_class == key] = int(count)
                count += 1

        # Need to re-write soil map so that classes start from 1 and sequentially thereafter
        soi_raster = {'data': soil_class[0], 'profile': geo_tiff['profile']}
        self.write_ascii(soi_raster, output_file, dtype='int16')

        # create soil table with  nodata for rasyes
        parameters = ['ID', 'Ks', 'thetaS', 'thetaR', 'm', 'PsiB', 'f', 'As', 'Au', 'n', 'ks', 'Cs', 'Texture']
        soil_list = []
        count = 1
        nodata = 9999.99

        for key, item in filtered_classes.items():
            d = {}
            for p in parameters:
                # reset ID
                if p == 'ID':
                    d.update({p: count})

                # give textural class that need to be updated via user or calibration
                elif p in (['As', 'Au', 'Cs', 'ks', 'Texture']):
                    d.update({p: item})

                # set grid data to nodata value in table
                else:
                    d.update({p: nodata})
            count += 1

            soil_list.append(d)

        return soil_list

    def process_raw_soil(self, grid_input, output=None, ks_only=False):
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

                                    If a string is provided, it is treated as the path to a file containing the
                                    configuration file must be written in json format.

        - output (list, optional): List of output file names for different soil properties.
        - ks_only True will only write rasters for ks, this is useful if using compute_decay_ks

        Note:
        - The 'grid_types' key should contain a list of dictionaries, each specifying a grid type and its corresponding file path.
        - The 'output_files' key should contain a list of exactly 5 output file names for different soil properties.
        - The file paths in 'grid_types' should be valid, and the 'output_files' list should have the correct size.
        """

        # Check if grid_input is a string (path to a config file)
        if isinstance(grid_input, str):
            # Read configuration from the file
            config = self.read_json(grid_input)
            grids = config['grid_types']
            output_files = config['output_files']
        elif isinstance(grid_input, list):
            # Use provided dictionary
            grids = grid_input
            output_files = output or ['Ks.asc', 'theta_r.asc', 'theta_s.asc', 'psib.asc', 'm.asc']
        else:
            print(
                'Invalid input format. Provide either a list of dictionaries specifying type and path, or a path to a configuration file.')
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
        for cnt, g in enumerate(grids):
            grid_type, path = g['type'], g['path']
            print(f"Ingesting {grid_type} from: {path}")
            geo_tiff = self.read_ascii(path)
            array = geo_tiff['data']
            size = array.shape

            if cnt == 0:
                sg250_data = np.zeros((6, size[0], size[1]))

            # each z layer follows:[sa (%), si (%), cl (%), bd (g/cm3), th33, th1500]
            if grid_type == 'sand':
                array = array / 1000 * 100  # convert SSC from g/kg to % SSC
                sg250_data[0, :, :] = array
            elif grid_type == 'silt':
                array = array / 1000 * 100  # convert SSC from g/kg to % SSC
                sg250_data[1, :, :] = array
            elif grid_type == 'clay':
                array = array / 1000 * 100  # convert SSC from g/kg to % SSC
                sg250_data[2, :, :] = array
            elif grid_type == 'bdod':
                array = array / 100  # convert bulk density from cg/cm3 to g/cm3
                sg250_data[3, :, :] = array
            elif grid_type == 'wv0033':
                array = array / 1000  # convert bulk density from cg/cm3 to g/cm3
                sg250_data[4, :, :] = array
            elif grid_type == 'wv1500':
                array = array / 1000  # convert bulk density from cg/cm3 to g/cm3
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
                data = [sg250_data[x, i, j] for x in np.arange(0, 6)]
                soil_data = SoilData.from_array([data])
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

        # for now only write out mean values
        soil_prop = [ks[0, :, :], theta_r[0, :, :], theta_s[0, :, :], psib[0, :, :], m[0, :, :]]

        if ks_only:
            soi_raster = {'data': soil_prop[0], 'profile': profile}
            self.write_ascii(soi_raster, output_files[0])
        else:
            for soil_property, name in zip(soil_prop, output_files):
                soi_raster = {'data': soil_property, 'profile': profile}
                self.write_ascii(soi_raster, name)

    def compute_ks_decay(self, grid_input, output=None):
        """
        Produces raster for the conductivity decay parameter f, following Ivanov et al., 2004.
        :param dict grid_input: If a dictionary list, keys are "depth" and "path" for each soil property.
                                Depth should be provided in units of mm.
                                    Format of dictionary list follows (from shallowest to deepest):
                                    [{'depth':25 , 'path':'path/to/25_mm_ks'},
                                    ...
                                    {'depth':800 , 'path':'path/to/800_mm_ks'},]

                                    If a string is provided, it is treated as the path to a configuration file. The
                                    configuration file must be written in json format.
        :param str output: Location to save raster with conductivity decay parameter f.
        """
        # Check if grid_input is a string (path to a config file)
        if isinstance(grid_input, str):
            # Read configuration from the file
            config = self.read_json(grid_input)
            grids = config['grid_depth']
            output_file = config['output_file']
        elif isinstance(grid_input, list):
            # Use provided dictionary
            grids = grid_input
            output_file = output or ['f.asc']
        else:
            print('Invalid input format. Provide either a list or a path to a configuration file.')
            return

        # Check if each file specified in the dictionary or config exists
        for g in grids:
            grid_type, path = g['depth'], g['path']
            if not os.path.isfile(path):
                raise FileNotFoundError(f'Cannot find: {path} for grid type: {grid_type}')

        ks_data = None
        size = None
        raster = None
        depth_vec = np.zeros(len(grids))

        # Loop through specified file paths
        for cnt, g in enumerate(grids):
            depth, path = g['depth'], g['path']
            print(f"Ingesting Ks grid at {depth} from: {path}")
            raster = self.read_ascii(path)
            array = raster['data']
            depth_vec[cnt] = depth

            if cnt == 0:
                size = array.shape
                ks_data = np.zeros((len(grids), size[0], size[1]))
                ks_data[cnt, :, :] = array
            else:
                ks_data[cnt, :, :] = array

        # Ensure that ks grids are sorted from surface to the deepest depth
        depth_sorted = np.argsort(depth_vec)
        ks_data = ks_data[depth_sorted]

        depth_vec = depth_vec[depth_sorted]
        depth_vec = depth_vec.astype(float)  # ensure float for fitting

        profile = raster['profile']  # for writing later

        # Initialize parameter grids
        f_grid = np.zeros(np.shape(array))  # parameter grid
        fcov = np.zeros(np.shape(array))  # coef of variance grid

        # Loop through raster's and compute soil properties using rosetta-soil package
        for i in range(0, size[0]):
            for j in range(0, size[1]):
                y = np.array([ks_data[n, i, j] for n in np.arange(0, len(grids))])

                if np.any(y == profile['nodata']):
                    f_grid[i, j] = profile['nodata']
                    fcov[i, j] = profile['nodata']
                else:

                    try:
                        # ensure float
                        y = y.astype(float)
                    except ValueError:
                        raise ValueError("Input data must be convertible to float")

                    k0 = y[0]

                    # define exponential decay function, Ivanov et al. (2004) eqn 17
                    decay = lambda x, f, k=k0: k * (f * x / (np.exp(f * x) - 1.0))

                    # perform curve fitting, set limits on f and surface Ksat for stability
                    minf, maxf = 1E-7, k0 - 1E-5
                    minks = 1

                    param, param_cov = curve_fit(decay, depth_vec, y, bounds=([minf, maxf], [minks, k0]))

                    # Write Curve fitting results to grid
                    f_grid[i, j] = param[0]
                    fcov[i, j] = param_cov[0, 0]

        f_raster = {'data': f_grid, 'profile': profile}
        self.write_ascii(f_raster, output_file)
