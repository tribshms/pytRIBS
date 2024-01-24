import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon


def read_voi_file(instance, filename=None):
    """
    Returns GeoDataFrame containing voronoi polygons from tRIBS model domain.
    :param filename: Set to read _reach file specified from OUTFILENAME,but can be changed.
    :return: GeoDataFrame

    """

    if filename is None:
        filename = instance.options["outfilename"]["value"] + "_voi"

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

        if instance.geo["EPSG"] is not None:
            voi = gpd.GeoDataFrame(voi_features, crs=instance.geo["EPSG"])
            nodes = gpd.GeoDataFrame(node_features, crs=instance.geo["EPSG"])
        else:
            voi = gpd.GeoDataFrame(voi_features)
            nodes = gpd.GeoDataFrame(node_features)
            print("Coordinate Reference System (CRS) was not added to the GeoDataFrame")
        return [voi, nodes]

    else:
        print("Voi file not found.")
        return None


def read_reach_file(instance, filename=None):
    """
    Returns GeoDataFrame containing reaches from tRIBS model domain.
    :param filename: Set to read _reach file specified from OUTFILENAME,but can be changed.
    :return: GeoDataFrame
    """

    if filename is None:
        filename = instance.options["outfilename"]["value"] + "_reach"

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
    if instance.geo["EPSG"] is not None:
        gdf = gpd.GeoDataFrame(features, crs=instance.geo["EPSG"])
    else:
        gdf = gpd.GeoDataFrame(features)
        print("Coordinate Reference System (CRS) was not added to the GeoDataFrame")

    return gdf


def merge_parllel_voi(instance, join=None, result_path=None, format=None, save=True):
    """
    Returns geodataframe of merged vornoi polygons from parallel tRIBS model run.

    :param join: Data frame of dynamic or integrated tRIBS model output (optional).
    :param save: Set to True to save geodataframe (optional, default True).
    :param result_path: Path to save geodateframe (optional, default OUTFILENAME).
    :param format: Driver options for writing geodateframe (optional, default = ESRI Shapefile)

    :return: GeoDataFrame
    """

    outfilename = instance.options["outfilename"]["value"]
    path_components = outfilename.split(os.path.sep)
    # Exclude the last directory as its actually base name
    outfilename = os.path.sep.join(path_components[:-1])

    parallel_voi_files = [f for f in os.listdir(outfilename) if 'voi.' in f]  # list of _voi.d+ files

    if len(parallel_voi_files) == 0:
        print(f"Cannot find voi files at: {outfilename}. Returning None")
        return None

    voi_list = []
    # gdf = gpd.GeoDataFrame(columns=['ID', 'geometry'])

    for file in parallel_voi_files:
        voi = instance.read_voi_file(f"{outfilename}/{file}")
        if voi is not None:
            voi_list.append(voi[0])
        else:
            print(f'Voi file {file} is empty.')

    combined_gdf = gpd.pd.concat(voi_list, ignore_index=True)
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


def merge_parllel_spatial_files(instance, suffix="_00d", dtime=0, write=True, header=True, colnames=None, single=True):
    """
    Returns dictionary of combined spatial outputs for intervals specified by tRIBS option: "SPOPINTRVL".
    :param str suffix: Either _00d for dynamics outputs or _00i for time-integrated ouputs.
    :param int dtime : Option to specify time step at which to start merge of files.
    :param bool write: Option to write dataframes to file.
    :param bool header: Set to False if headers are not provided with spatial files.
    :param bool colnames: If header = False, column names can be provided for the dataframe--but it is expected the first column is ID.
    :param bool single: If single = True then only spatial files specified at dtime are merged.
    :return: Dictionary of pandas dataframes.
    """

    runtime = int(instance.options["runtime"]["value"])
    spopintrvl = int(instance.options["spopintrvl"]["value"])
    outfilename = instance.options["outfilename"]["value"]

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
