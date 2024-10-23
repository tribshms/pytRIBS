# shared_mixin.py
import os
import glob
import sys

import numpy as np

import geopandas as gpd
import pandas as pd
import pyvista as pv
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon


class Meta:
    """
    Class for project metadata.
    """
    def __init__(self):
        self.meta = {"Name": None, "Scenario": None, "EPSG": None}


class Shared:
    """
    Shared methods betweens the pytRIBS Classes.
    """

    def read_input_file(self, file_path):
        """
        Reads .in file for tRIBS model simulation and assigns values to options attribute.
        :param file_path: Path to .in file.

        """
        with open(file_path, 'r') as file:
            lines = file.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()  # Remove leading/trailing whitespace
            for key in self.options.keys():
                # Convert key to lowercase for case-insensitive comparison
                key_lower = key.lower()
                # Convert line to lowercase for case-insensitive comparison
                line_lower = line.lower()
                if line_lower.startswith(key_lower):
                    # Extract the portion of the line after the key
                    if i + 1 < len(lines):
                        # Extract the value from the next line
                        value = lines[i + 1].strip()
                        self.options[key]['value'] = value
            i += 1

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

    def read_voi_file(self, filename=None):
        """
        Returns GeoDataFrame containing voronoi polygons from tRIBS model domain.
        :param filename: Set to read _reach file specified from OUTFILENAME,but can be changed.
        :return: GeoDataFrame

        """

        if filename is None:
            filename = self.options["outfilename"]["value"] + "_voi"

        ids = []
        polygons = []
        points = []
        line_count = 0

        if os.path.exists(filename):
            with open(filename, 'r') as file:
                current_id = None
                current_voi_points = []
                current_node_points = []

                for line in file:

                    line_count += 1

                    if line.strip() != "END":
                        parts = line.strip().split(',')

                        if parts:
                            if len(parts) == 3:
                                id_, x, y = map(float, parts)
                                current_id = id_
                                current_node_points.append((x, y))
                            elif len(parts) == 2:
                                x, y = map(float, parts)
                                current_voi_points.append((x, y))

                    elif line.strip() == "END":

                        if current_id is None:
                            break  ## catch end of file w/ two ends in a row

                        ids.append(current_id)
                        polygons.append(Polygon(current_voi_points))
                        points.append(Point(current_node_points))

                        current_id = None
                        current_voi_points = []
                        current_node_points = []

            if line_count <= 1:
                print(filename + "is empty.")
                return None

            # Package Voronoi
            if not ids or not polygons:
                raise ValueError("No valid data found in " + filename)

            voi_features = {'ID': ids, 'geometry': polygons}
            node_features = {'ID': ids, 'geometry': points}

            if self.meta["EPSG"] is not None:
                voi = gpd.GeoDataFrame(voi_features, crs=self.meta["EPSG"])
                nodes = gpd.GeoDataFrame(node_features, crs=self.meta["EPSG"])
            else:
                voi = gpd.GeoDataFrame(voi_features)
                nodes = gpd.GeoDataFrame(node_features)
                print("Coordinate Reference System (CRS) was not added to the GeoDataFrame")
            return voi, nodes

        else:
            print("Voi file not found.")
            return None

    @staticmethod
    def read_node_list(file_path):
        """
        Returns node list provide by .dat file.

        The node list can be further modified or used for reading in element/pixel files and subsequent processing.

        :param file_path: Relative or absolute file path to .dat file.
        :type file_path: str
        :return: List of nodes specified by .dat file
        :rtype: list

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

    def read_reach_file(self, filename=None):
        """
        Returns GeoDataFrame containing reaches from tRIBS model domain.
        :param filename: Set to read _reach file specified from OUTFILENAME,but can be changed.
        :return: GeoDataFrame
        """

        if filename is None:
            filename = self.options["outfilename"]["value"] + "_reach"

        with open(filename, 'r') as file:
            lines = file.readlines()

        features = []
        current_id = None
        coordinates = []

        for line in lines:
            line = line.strip()
            if line == "END":
                if current_id is not None:
                    line_string = LineString(coordinates)
                    features.append({"ID": current_id, "geometry": line_string})
                    current_id = None
                    coordinates = []
            else:
                if current_id is None:
                    current_id = int(line)
                else:
                    x, y = map(float, line.split(','))
                    coordinates.append((x, y))
        if self.meta["EPSG"] is not None:
            gdf = gpd.GeoDataFrame(features, crs=self.meta["EPSG"])
        else:
            gdf = gpd.GeoDataFrame(features)
            print("Coordinate Reference System (CRS) was not added to the GeoDataFrame")

        return gdf

    def merge_parallel_voi(self, join=None, result_path=None, format=None, save=False):
        """
        Returns geodataframe of merged vornoi polygons from parallel tRIBS model run.

        :param join: Data frame of dynamic or integrated tRIBS model output (optional).
        :param save: Set to True to save geodataframe (optional, default True).
        :param result_path: Path to save geodateframe (optional, default OUTFILENAME).
        :param format: Driver options for writing geodateframe (optional, default = ESRI Shapefile)

        :return: GeoDataFrame

        """

        outfilename = self.options["outfilename"]["value"]
        path_components = outfilename.split(os.path.sep)
        # Exclude the last directory as its actually base name
        outfilename = os.path.sep.join(path_components[:-1])

        parallel_voi_files = [f for f in os.listdir(outfilename) if 'voi.' in f]  # list of _voi.d+ files

        if len(parallel_voi_files) == 0:
            print(f"Cannot find voi files at: {outfilename}. Returning None")
            return None

        voi_list = []
        processor_list = []
        # gdf = gpd.GeoDataFrame(columns=['ID', 'geometry'])

        for file in parallel_voi_files:
            voi = self.read_voi_file(f"{outfilename}/{file}")
            if voi is not None:
                voi_list.append(voi[0])
                processor = int(file.split("voi.")[-1])  # Extract processor number from file name
                processor_list.extend(np.ones(len(voi[0])) * int(processor))
            else:
                print(f'Voi file {file} is empty.')

        combined_gdf = gpd.pd.concat(voi_list, ignore_index=True)
        combined_gdf['processor'] = processor_list  # Add 'processor' column
        combined_gdf = combined_gdf.sort_values(by='ID')

        if join is not None:
            combined_gdf = combined_gdf.merge(join, on="ID", how="inner")

            # Check for non-matching IDs
            non_matching_ids = join[~join["ID"].isin(combined_gdf["ID"])]

            if not non_matching_ids.empty:
                print("Warning: Some IDs from the dynamic or integrated data frame do not match with the voronoi IDs.")

        if save:
            if result_path is None:
                result_path = os.path.join(outfilename, "_mergedVoi")

            if format is None:
                format = "ESRI Shapefile"

            combined_gdf.to_file(result_path, driver=format)

        return combined_gdf

    def merge_parallel_spatial_files(self, suffix="_00d", dtime=0, write=True, header=True, colnames=None,
                                     single=True):
        """
        Returns dictionary of combined spatial outputs for intervals specified by tRIBS option: "SPOPINTRVL".
        :param str suffix: Either _00d for dynamics outputs or _00i for time-integrated ouputs.
        :param int dtime : Option to specify time step at which to start merge of files.
        :param bool write: Option to write dataframes to file.
        :param bool header: Set to False if headers are not provided with spatial files.
        :param bool colnames: If header = False, column names can be provided for the dataframe--but it is expected the first column is ID.
        :param bool single: If single = True then only spatial files specified at dtime are merged.
        :return: Dictionary of pandas dataframes.
        # TODO: Rename as get_spatial_files, and enable it to read parallel or serial results.
        # TODO add a clean option to store .0 t0 .n files, then zip, probably would only want this if you are saving them out.
        # TODO also return file names if saved out, also add serial version or a serial flag...so people can reaou
        """

        runtime = int(self.options["runtime"]["value"])
        spopintrvl = int(self.options["spopintrvl"]["value"])
        outfilename = self.options["outfilename"]["value"]

        dyn_data = {}
        times = [dtime + i * spopintrvl for i in range((runtime - dtime) // spopintrvl + 1)]
        times.append(runtime)

        for _time in times:
            processes = 0
            otime = str(_time).zfill(4)
            dynfile = f"{outfilename}.{otime}{suffix}.{processes}"

            if os.path.exists(dynfile):
                while os.path.exists(dynfile):
                    if processes == 0:
                        processes += 1
                        try:
                            if header:
                                df = pd.read_csv(dynfile, header=0)
                            else:
                                df = pd.read_csv(dynfile, header=None, names=colnames)

                        except pd.errors.EmptyDataError:
                            print(f'The first file is empty: {dynfile}.\n Can not merge files.')
                            break

                        dynfile = f"{outfilename}.{otime}{suffix}.{processes}"

                    else:
                        processes += 1
                        try:

                            if header:
                                df = pd.concat([df, pd.read_csv(dynfile, header=0)])
                            else:
                                df = pd.concat([df, pd.read_csv(dynfile, header=None, names=colnames)])

                        except pd.errors.EmptyDataError:
                            print(f'The following file is empty: {dynfile}')
                        dynfile = f"{outfilename}.{otime}{suffix}.{processes}"

                if header:
                    df = df.sort_values(by='ID')

                if write:
                    df.to_csv(f"{outfilename}.{otime}{suffix}", index=False)

                dyn_data[otime] = df

                if single:
                    break


            elif os.path.exists(dynfile):
                print("Cannot find dynamic output file:" + dynfile)
                break

        return dyn_data

    def mesh2vtk(self, outfile):
        """
        Converts mesh data files into a VTK file format for visualization.

        This function reads node, triangle, and elevation data from files and writes them to a VTK file.
        The VTK file will be an unstructured grid dataset containing points and cells, with associated scalar data.

        Parameters
        ----------
        outfile : str
            Path to the output VTK file where the mesh data will be written.

        Returns
        -------
        None

        Notes
        -----
        - The function expects the following files in the directory specified by the 'outfilename' option:
            - A node file with a `.nodes` extension containing node coordinates and boundary codes.
            - A triangle file with a `.tri` extension containing triangle vertex indices.
            - A z-file with a `.z` extension containing elevation values.
        - The node file should contain columns for x, y coordinates, and a boundary code.
        - The triangle file should contain columns for vertex indices of triangles.
        - The z-file should contain elevation values for each node.
        - The output VTK file will include point data (coordinates and elevations) and cell data (triangles).
        - Boundary codes are used to set NaN values in the altitude scalars in the VTK file.

        Example
        -------
        >>> self.mesh2vtk('output_mesh.vtk')

        Raises
        ------
        FileNotFoundError
            If the required node, triangle, or z files cannot be found in the specified directory.
        IndexError
            If there is an issue reading data from the node, triangle, or z files, which may indicate file corruption.
        """
        outfilename = self.options["outfilename"]["value"]
        last_slash_index = outfilename.rfind('/')
        directory_path = outfilename[:last_slash_index + 1]

        if os.path.exists(directory_path):
            node_file = glob.glob(directory_path + '*.nodes*')
        else:
            print(f'Cannot find node file at: {directory_path}. Exiting.')
            return

        if os.path.exists(directory_path):
            tri_file = glob.glob(directory_path + '*.tri*')
        else:
            print(f'Cannot find tri file at: {directory_path}. Exiting.')
            return

        if os.path.exists(directory_path):
            z_file = glob.glob(directory_path + '*.z*')
        else:
            print(f'Cannot find z file at: {directory_path}. Exiting.')
            return

        # read in node,tri,z files:
        try:

            with open(node_file[0], 'r') as f:
                lines = f.readlines()  # skip first since it's relic feature

                # Check if there's at least one line
                if lines:
                    num_nodes = int(lines[1])
                    store_nodes = np.zeros((num_nodes, 2))
                    boundary_code = np.zeros((num_nodes, 1))

                    # Iterate from the second line onward
                    for l in range(2, num_nodes + 2):
                        try:
                            line = lines[l].split()
                            store_nodes[l - 2, 0] = float(line[0])
                            store_nodes[l - 2, 1] = float(line[1])
                            boundary_code[l - 2, 0] = float(line[3])
                        except IndexError as e:
                            print(f'Node file may be corrupted, check line {l}')
                            print(f"Error: {e}")
                            sys.exit(1)

            with open(tri_file[0], 'r') as f:
                lines = f.readlines()

                # Check if there's at least one line
                if lines:
                    num_tri = int(lines[1])
                    store_tri = np.zeros((num_tri, 3))

                    # Iterate from the second line onward
                    for l in range(2, num_tri + 2):
                        try:
                            line = lines[l].split()
                            store_tri[l - 2, 0] = float(line[0])
                            store_tri[l - 2, 1] = float(line[1])
                            store_tri[l - 2, 2] = float(line[2])
                        except IndexError as e:
                            print(f'Tri file may be corrupted, check line {l}')
                            print(f"Error: {e}")
                            sys.exit(1)

            with open(z_file[0], 'r') as f:
                lines = f.readlines()

                # Check if there's at least one line
                if lines:
                    num_z = int(lines[1])
                    store_z = np.zeros((num_z, 1))

                    # Iterate from the second line onward
                    for l in range(2, num_z + 2):
                        try:
                            line = lines[l].split()
                            store_z[l - 2, 0] = float(line[0])
                        except IndexError as e:
                            print(f'Z file may be corrupted, check line {l}')
                            print(f"Error: {e}")
                            sys.exit(1)

            with open(outfile, 'w') as f:
                f.write("# vtk DataFile Version 3.0\n")
                f.write("tRIBS\n")
                f.write("ASCII\n")
                f.write("DATASET UNSTRUCTURED_GRID\n")
                f.write('POINTS {0:10d} float\n'.format(num_nodes))
                for I in range(num_nodes):
                    f.write(
                        "{0:15.5f} {1:15.5f} {2:15.5f}\n".format(store_nodes[I, 0], store_nodes[I, 1], store_z[I, 0]))

                f.write("CELLS {0:10d} {1:10d}\n".format(num_tri, 4 * num_tri))
                for I in range(num_tri):
                    f.write('3 {0:10d} {1:10d} {2:10d}\n'.format(int(store_tri[I, 0]), int(store_tri[I, 1]),
                                                                 int(store_tri[I, 2])))

                f.write("CELL_TYPES {0:10d}\n".format(num_tri))
                for I in range(num_tri):
                    f.write("5\n")

                f.write("POINT_DATA {0:10d}\n".format(num_nodes))
                f.write("SCALARS Altitude float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for I in range(num_nodes):
                    if boundary_code[I, 0] == 1:
                        f.write('NaN' + '\n')
                    else:
                        f.write(str(store_z[I, 0]) + "\n")

                f.write('SCALARS BC_code float 1\n')
                f.write('LOOKUP_TABLE BC_LUT\n')

                for I in range(num_nodes):
                    f.write(str(float(boundary_code[I, 0])) + '\n')

                # possible to add additional scalars
                # f.write("SCALARS Shear_stress float 1\n")
                # f.write("LOOKUP_TABLE default\n")
                # for I in range(num_nodes):
                #     f.write(str(TABTAU[I, 0]) + "\n")

        except FileNotFoundError:
            return

    @staticmethod
    def plot_mesh(mesh, scalar=None, **kwargs):
        """
        Plots a 3D mesh using PyVista with optional scalar data.

        This method visualizes a mesh object, optionally using scalar data to color the mesh. It handles meshes
        from a file path or PyVista object and allows for customizing the plot with additional keyword arguments.

        Parameters
        ----------
        mesh : str or pv.PolyData
            If a string is provided, it should be a path to a mesh file that will be read using PyVista. If a PyVista
            `PolyData` object is provided, it will be used directly for plotting.

        scalar : array-like, optional
            Scalar data to be used for coloring the mesh. If not provided, it defaults to the 'Elevation' array of the
            mesh. The scalar data must match the number of points or cells in the mesh.

        **kwargs : keyword arguments
            Additional keyword arguments passed to `pyvista.Plotter.add_mesh` for further customization of the plot.

        Returns
        -------
        pv.Plotter
            A PyVista `Plotter` object configured to display the mesh.

        Notes
        -----
        - If `scalar` is provided, it will be used to color the mesh. Closed points or cells (where 'BoundaryCode' is 1)
          are set to NaN.
        - If the length of `scalar` matches the number of points, NaNs are assigned to closed points.
        - If the length of `scalar` matches the number of cells, NaNs are assigned to closed cells.
        - The plot camera is set to view from the top-down (xy plane) with north up.

        Example
        -------
        >>> mesh = pv.read('path_to_mesh_file.vtk')
        >>> plotter = plot_mesh(mesh, scalar=my_scalar_data, cmap='viridis')
        >>> plotter.show()

        Raises
        ------
        ValueError
            If the length of `scalar` does not match either the number of points or cells in the mesh.
        """
        if isinstance(mesh, str):
            # check if path exists
            mesh = pv.read(mesh)

        if scalar is None:
            scalar = mesh.get_array('Elevation')

        # set closed points or cells to nan
        if len(scalar) == mesh.n_points:
            scalar[mesh['BoundaryCode'] == 1] = np.nan
            mesh.point_data['scale'] = scalar
        elif len(scalar) == mesh.n_cells:
            extracted = mesh.extract_points(mesh['BoundaryCode'] == 1, adjacent_cells=True)
            scalar[extracted.cell_data['vtkOriginalCellIds']] = np.nan
            mesh.point_data['scale'] = scalar
        else:
            print("Scalar dimensions must match either the number of points or cells in the mesh.")

        plotter = pv.Plotter()
        plotter.add_mesh(mesh, scalars='scale', **kwargs)
        plotter.camera_position = 'xy'  # Set camera to view from top-down (xz plane)
        plotter.view_vector = [0, 0, 1]  # Set view direction vector to [0, 0, 1] (north is up)

        return plotter

    def get_invariant_properties(self):
        """
        Reads and processes invariant spatial properties based on the parallel mode setting.

        This method handles the integration of spatial variables and Voronoi files depending on the mode specified
        in the options. It merges parallel files or reads single files, computes weights, and loads Voronoi data.

        The method does the following:
        - Checks the `parallelmode` setting to determine if parallel processing is enabled.
        - Merges parallel spatial files if in parallel mode, or reads a single spatial file if not.
        - Computes weights based on the `VAr` column if in non-parallel mode.
        - Loads Voronoi files based on the `parallelmode` setting.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        - If `parallelmode` is set to 1, the method merges files with a `_00i` suffix and integrates spatial variables
          based on runtime values.
        - If `parallelmode` is set to 0, it reads a single file based on the `outfilename` and `runtime` values,
          and computes weights using the `VAr` column.
        - Voronoi files are read or merged based on the parallel mode setting.
        - If the `parallelmode` is not recognized, it prints an error message and sets the spatial variables and Voronoi data to `None`.

        Example
        -------
        >>> obj.get_invariant_properties()

        Raises
        ------
        ValueError
            If there are issues merging files or reading Voronoi data.
        """

        parallel_flag = int(self.options["parallelmode"]['value'])

        # read in integrated spatial vars for waterbalance calcs and spatial maps
        if parallel_flag == 1:
            temp = self.merge_parallel_spatial_files(suffix="_00i", dtime=int(self.options['runtime']['value']))

            if not temp:
                print(f'Failed to merge parallel files, check the correct file path was provided')

            runtime = self.options["runtime"]["value"]

            while len(runtime) < 4:
                runtime = '0' + runtime

            self.int_spatial_vars = temp[runtime]

        elif parallel_flag == 0:
            runtime = self.options["runtime"]["value"]

            if len(runtime) < 4:
                while len(runtime) < 4:
                    runtime = '0' + runtime

            outfilename = self.options["outfilename"]["value"]
            intfile = f"{outfilename}.{runtime}_00i"

            self.int_spatial_vars = pd.read_csv(intfile)

            # Note one could use max CAr, but it overestimates area according to Voi geomerty
            self.int_spatial_vars['weight'] = self.int_spatial_vars.VAr.values / self.int_spatial_vars.VAr.sum()

        else:
            print('Unable To Read Integrated Spatial File (*_00i).')
            self.int_spatial_vars = None

        # read in voronoi files only once
        if parallel_flag == 1:
            self.voronoi = self.merge_parallel_voi()

        elif parallel_flag == 0:
            self.voronoi, _ = self.read_voi_file()
        else:
            print('Unable To Load Voi File(s).')
            self.voronoi = None
