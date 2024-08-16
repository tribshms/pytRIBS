import subprocess
import os
import shutil
import pandas as pd
import rasterio
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from rasterio import fill
import  matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer


class Aux:

    def utm_to_latlong(self, easting, northing, epsg=None):
        """
        Convert UTM coordinates to latitude and longitude using an EPSG code with pyproj.

        Parameters:
        easting (float): UTM easting coordinate.
        northing (float): UTM northing coordinate.
        epsg_code (int): EPSG code representing the UTM projection.

        Returns:
        tuple: A tuple containing latitude and longitude.
        """
        if epsg is None:
            epsg = self.meta['EPSG']

        transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(easting, northing)

        return lat, lon

    @staticmethod
    def discrete_cmap(N, base_cmap='viridis'):
        """Create an N-bin discrete colormap from the specified input map."""
        base = plt.get_cmap(base_cmap)
        color_list = base(np.linspace(0, 1, N))
        cmap = ListedColormap(color_list, name=base.name + str(N))
        return cmap

    @staticmethod
    def fillnodata(files, overwrite=False, **kwargs):
        """
        Fills nodata gaps in raster files based on a maximum search distance.

        Parameters:
        files (list): List of paths to raster files.
        overwrite (bool): If True, the original files will be overwritten with filled data. If False, new files with "_filled" suffix will be created.
        **kwargs: Additional keyword arguments to be passed to rasterio.fill.fillnodata.

        Note:
        This function essentially wraps rasterio.fill.fillnodata.
        """
        for file_path in files:
            with rasterio.open(file_path) as src:
                data = src.read(1)
                msk = src.read_masks(1)
                filled_data = fill.fillnodata(data, mask=msk, **kwargs)
                if overwrite:
                    with rasterio.open(file_path, 'w', **src.profile) as dst:
                        dst.write(filled_data, 1)
                else:
                    base_name, ext = os.path.splitext(file_path)
                    filled_file_path = f"{base_name}_filled{ext}"
                    with rasterio.open(filled_file_path, 'w', **src.profile) as dst:
                        dst.write(filled_data, 1)
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
    def print_tags(self, tag_name):
        """
        Prints .in options for a specified tag.
        :param tag_name: Currently: "io", input/output, "physical", physical model params, "time", time parameters,
        "opts", parameters for model options, "restart", restart capabilities, "parallel", parallel options.

        Example:
            >>> m.print_tags("io")
        """

        data = self.options  # Assuming m.options is a dictionary with sub-dictionaries

        # Filter sub-dictionaries where "io" is in the "tags" list
        result = [item for item in data.values() if any(tag_name in tag for tag in item.get("tags", []))]

        # Display the filtered sub-dictionaries
        for dictionary in result:
            for item in dictionary:
                if item != "tags":
                    print(item + ": " + str(dictionary[item]))
                elif item == "tags":
                    print("\n")

    # MODEL FUNCTIONS
    @staticmethod
    def run(executable, input_file, mpi_command=None, tribs_flags=None, log_path=None,
            store_input=None, timeit=True, verbose=True):
        """
        Run a tRIBS model simulation with optional arguments.

        Run_simulation assumes that if relative paths are used then the binary and input file are collocated in the
        same directory. That means for any keywords that depend on a relative path, must be specified from the directory
        the tRIBS binary is executed. You can pass the location of the input file and executable as paths, in which case
        the function copies the binary and input file to same directory and then deletes both after the model run is complete.
        Optional arguments can be passed to store

        Args:
            binary_path (str): The path to the binary model executable.
            control_file_path (str): The path to the input control file for the binary.
            optional_args (str): Optional arguments to pass to the binary.

        Returns:
            int: The return code of the binary model simulation.
        """
        if mpi_command is not None:
            command = mpi_command.split()
            command.extend([executable, input_file])
        else:
            command = [executable, input_file]

        if tribs_flags is not None:
            command.extend(tribs_flags.split())

        if log_path is not None:
            command.append(log_path)

        if timeit:
            command.insert(0, "time")

        print(command)

        if verbose:
            subprocess.run(command)
        else:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @staticmethod
    def build(source_file, build_directory, verbose=True, exe="tRIBS", parallel="ON", cxx_flags="-O2"):
        """
        Run a tRIBS model simulation with optional arguments.

        Run_simulation assumes that if relative paths are used then the binary and input file are collocated in the
        same directory. That means for any keywords that depend on a relative path, must be specified from the directory
        the tRIBS binary is executed. You can pass the location of the input file and executable as paths, in which case
        the function copies the binary and input file to same directory and then deletes both after the model run is complete.
        Optional arguments can be passed to store

        Args:
            binary_path (str): The path to the binary model executable.
            control_file_path (str): The path to the input control file for the binary.
            optional_args (str): Optional arguments to pass to the binary.

        Returns:
            int: The return code of the binary model simulation.
        """
        # TODO: add check if build directory already exists, prompt user if they want to remove
        if os.path.exists(build_directory):
            print(f"The directory '{build_directory}' exists.")
            user_input = input("Would you like to remove it? [y/n]: ")
            if user_input:
                try:
                    # Remove the directory and its contents
                    shutil.rmtree(build_directory)
                    print(f"Directory '{build_directory}' and its contents have been removed.")
                except FileNotFoundError:
                    print(f"Directory '{build_directory}' does not exist.")
                except PermissionError:
                    print(f"Permission denied while attempting to remove '{build_directory}'.")
                except Exception as e:
                    print(f"An error occurred: {str(e)}")

        # Allow modification of CMakeList.txt
        modified_lines = []
        source_file = os.path.expanduser(source_file)
        file_path = os.path.join(source_file, "CMakeLists.txt")

        # Define the variables to search for and their corresponding replacements
        variables_to_replace = {
            "exe": exe,
            "parallel": parallel,
            "cxx_flags": cxx_flags
        }

        # Read the contents of the CMakeLists.txt file and modify lines as needed
        with open(file_path, 'r') as file:
            for line in file:
                for variable, value in variables_to_replace.items():
                    if line.strip().startswith(f'set({variable} "'):
                        line = f'set({variable} "{value}")\n'
                modified_lines.append(line)

        # Write the modified lines back to the file
        with open(file_path, 'w') as file:
            file.writelines(modified_lines)

        if verbose:
            cmake_configure_command = ["cmake", "-B", build_directory, "-S", source_file]
            subprocess.run(cmake_configure_command)

            cmake_build_command = ["cmake", "--build", build_directory, "--target", "all"]
            result = subprocess.run(cmake_build_command)
        else:
            cmake_configure_command = ["cmake", "-B", build_directory, "-S", source_file]
            subprocess.run(cmake_configure_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            cmake_build_command = ["cmake", "--build", build_directory, "--target", "all"]
            result = subprocess.run(cmake_build_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return result.returncode

    def clean(self):
        pass
