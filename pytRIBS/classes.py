# system packages

# external packages

# pytRIBS
from pytRIBS.shared.aux import Aux
from pytRIBS.model.diagnose import Diagnostics
from pytRIBS.shared.infile_mixin import InfileMixin
from pytRIBS.shared.inout import InOut
from pytRIBS.shared.shared_mixin import SharedMixin, Meta
from pytRIBS.results.waterbalance import WaterBalance
from pytRIBS.results.read import Read
from pytRIBS.results.visualize import Viz
from pytRIBS.results.evaluate import Evaluate
from pytRIBS.model.run_docker import tRIBSDocker

# preprocessing componets
from pytRIBS.soil.soil import _Soil
from pytRIBS.met.met import _Met
from pytRIBS.mesh.mesh import Preprocess, GenerateMesh
from pytRIBS.land.land import _Land

import os



class Project:
    """
    Represents a tRIBS project for managing directories and metadata in a specified base directory.

    This class initializes with a base directory, a project name, and an EPSG code. It sets up a
    predefined set of directories for data, results, and various sub-categories. It also provides
    functionality to create these directories if they do not already exist.

    Attributes
    ----------
    base_dir : str
        The base directory where all project-related directories will be created.
    meta : dict
        A dictionary to store metadata about the project, including 'Name' and 'EPSG'.
    directories : dict
        A dictionary defining the structure of directories to be created within the base directory.

    Parameters
    ----------
    base_dir : str
        The base directory path for the project.
    name : str
        The name of the project.
    epsg : int
        The EPSG code representing the coordinate system.

    Methods
    -------
    create_directories()
        Creates the directories defined in the `directories` attribute within the base directory.

    Example
    -------
    >>> project = Project("/path/to/base", "MyProject", 4326)
    >>> # This initializes the project, creates necessary directories, and sets metadata.
    """
    def __init__(self, base_dir, name, epsg):
        self.base_dir = base_dir
        Meta.__init__(self)
        self.meta['Name'] = name
        self.meta['EPSG'] = epsg
        self.directories = {
            "model": os.path.join("data", "model"),
            "preprocessing": os.path.join("data", "preprocessing"),
            "results": "results",
            "soil": os.path.join("data", "model", "soil"),
            "land": os.path.join("data", "model", "land"),
            "met_precip": os.path.join("data", "model", "met", "precip"),
            "met_meteor": os.path.join("data", "model", "met", "meteor"),
            "mesh": os.path.join("data", "model", "mesh")
        }
        self.create_directories()

    def create_directories(self):
        """
        Creates directories defined in the `directories` attribute within the base directory.

        This method uses `os.makedirs` with `exist_ok=True` to ensure that all required directories are
        created if they do not already exist. It traverses through the `directories` dictionary, combining
        the base directory with each relative path to create the full directory paths.
        """
        for key, rel_path in self.directories.items():
            full_path = os.path.join(self.base_dir, rel_path)
            os.makedirs(full_path, exist_ok=True)

class Model(InfileMixin, SharedMixin, Aux, Diagnostics, Preprocess, InOut):
    """
    A tRIBS Model class.

    This class provides access to the underlying framework of a tRIBS (TIN-based Real-time Integrated Basin
    Simulator) simulation. It includes one nested class: Results. The Model class is initialized at the top-level to
    facilitate model setup, simulation, post-processing, and can be used for manipulating and generating multiple
    simulations efficiently.

    Attributes
    ----------
    input_options : dict
        A dictionary of the necessary keywords for a tRIBS `.in` file.
    model_input_file : str
        Path to a template `.in` file with the specified paths for model results, inputs, etc.

    Parameters
    ----------
    input_file : str, optional
        Path to a template `.in` file. Default is `None`.
    met : object, optional
        pytRIBS Met Class object Default is `None`.
    land : object, optional
        pytRIBS Land object. Default is `None`.
    soil : object, optional
        pytRIBS Soil object. Default is `None`.
    mesh : object, optional
        pytRIBS Mesh object. Default is `None`.
    meta : dict, optional
        pytRIBS Meta object Default is `None`.

        """

    def __init__(self, input_file=None, met=None, land=None, soil=None, mesh=None, meta=None):
        # attributes
        self.options = self.create_input_file()  # input options for tRIBS model run

        if input_file is not None:
            self.read_input_file(input_file)

        Meta.__init__(self)

        if meta is not None:
            self.meta = meta

        # Initialize with provided instances
        self.met = met
        self.land = land
        self.soil = soil
        self.mesh = mesh

        # Merge options from provided instances
        self.update_shared_options(met=met,land=land,soil=soil,mesh=mesh)

    # SIMULATION METHODS
    def __getattr__(self, name):
        if name in self.options:
            return self.options[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __dir__(self):
        # Include the keys from the options dictionary and the methods of the class
        return list(
            set(super().__dir__() + list(self.options.keys()))) if self.options is not None else super().__dir__()

    def update_shared_options(self, met=None, land=None, soil=None, mesh=None):
        # List of provided instances
        instances = [met, land, soil, mesh]

        for instance in instances:
            if instance:
                # Update only shared options
                for key in instance.__dict__:
                    if key in self.options:
                        self.options[key] = instance.__dict__[key]

    @staticmethod
    def run_tribs_docker(volume_path, input_file, execution_mode='serial', num_processes=None):
        """Main function to run the TRIBSDocker class."""
        docker_instance = tRIBSDocker(volume_path, input_file, execution_mode, num_processes)
        docker_instance.start_docker_desktop()
        docker_instance.initialize_docker_client()
        docker_instance.pull_image()
        docker_instance.run_container()
        docker_instance.execute()
        docker_instance.cleanup_container()


class Results(InfileMixin, SharedMixin, WaterBalance, Read, Viz, Evaluate):
    """
    A tRIBS Results Class.

    This class provides a framework for analyzing and visualizing individual tRIBS simulations. It takes an instance of
    the `Simulation` class and provides time-series and water balance analysis of the model results.

    Attributes
    ----------
    options : dict
        A dictionary of input options for the tRIBS model run.
    element : dict
        A dictionary for storing elements related to the results.
    mrf : dict
        A dictionary containing `mrf` and `waterbalance`, which are initialized to `None`.
    meta : dict, optional
        Metadata dictionary for additional information. Default is `None`.

    Example
    -------
    To create and use an instance of the `Results` class:

    >>> results = Results(input_file="path/to/input_file.in")
    >>> results.create_input_file()
    >>> results.read_input_file("path/to/input_file.in")
    >>> results.get_invariant_properties()
    """
    def __init__(self, input_file, meta=None):
        # setup model paths and options for Result Class
        self.options = self.create_input_file()  # input options for tRIBS model run
        self.read_input_file(input_file)

        # attributes for analysis, plotting, and archiving model results
        self.element = {}
        self.mrf = {'mrf': None, 'waterbalance': None}
        Meta.__init__(self)

        if meta is not None:
            self.meta = meta

        self.get_invariant_properties()  # shared


class Soil(_Soil):
    """
    A tRIBS Soil Class.

    This class handles soil-related data and options for the tRIBS model. It manages attributes related to soil mapping,
    soil tables, and groundwater files. The class inherits from `_Soil` and initializes its attributes based on the provided
    input file or default options.

    Attributes
    ----------
    soilmapname : str
        The name of the soil map file.
    soiltablename : str
        The name of the soil table file.
    scgrid : str
        The path to the SCGRID file.
    optsoiltype : int
        Option for soil type.
    optgroundwater : int
        Option for groundwater.
    optgwfile : int
        Option for groundwater file.
    optbedrock : int
        Option for bedrock.
    bedrockfile : str
        The path to the bedrock file.
    gwaterfile : str
        The path to the groundwater file.

    Example
    -------
    To create and use an instance of the `Soil` class:

    >>> soil = Soil(input_file="path/to/input_file.in")
    >>> print(soil.soilmapname)
    'path/to/soilmap'
    >>> print(soil.optsoiltype)
    1
    """

    def __init__(self, input_file=None,meta=None):

        Meta.__init__(self)

        if meta is not None:
            self.meta=meta

        if input_file is not None:
            options = SharedMixin.read_input_file(input_file)
        else:
            options = InfileMixin.create_input_file()

        # Initialize attributes
        self.soilmapname = options['soilmapname']
        self.soiltablename = options['soiltablename']
        self.scgrid = options['scgrid']
        self.optsoiltype = options['optsoiltype']
        self.optgroundwater = options['optgroundwater']
        self.optgwfile = options['optgwfile']
        self.optbedrock = options['optbedrock']
        self.bedrockfile = options['bedrockfile']
        self.gwaterfile = options['gwaterfile']


class Land(_Land):
    """
    A tRIBS Land Class.

    This class handles land-related data and options for the tRIBS model. It manages attributes related to land mapping,
    land tables, and land use grids. The class inherits from `_Land` and initializes its attributes based on the provided
    input file or default options.

    Attributes
    ----------
    landmapname : str
        The name of the land map file.
    landtablename : str
        The name of the land table file.
    lugrid : str
        The path to the LUGRID file.
    optlanduse : int
        Option for land use.
    optluintercept : int
        Option for land use interpolation.

    Example
    -------
    To create and use an instance of the `Land` class:

    >>> land = Land(input_file="path/to/input_file.in", meta={"some_key": "some_value"})
    >>> print(land.landmapname)
    'path/to/landmap'
    >>> print(land.optlanduse)
    1
    """

    def __init__(self, input_file=None,meta=None):

        Meta.__init__(self)

        if meta is not None:
            self.meta=meta

        if input_file is not None:
            options = SharedMixin.read_input_file(input_file)
        else:
            options = InfileMixin.create_input_file()

        # Initialize attributes
        self.landmapname = options['landmapname']
        self.landtablename = options['landtablename']
        self.lugrid = options['lugrid']
        self.optlanduse = options['optlanduse']
        self.optluintercept = options['optluinterp']


class Mesh:
    """
    A pytRIBS Mesh Class.

    This class manages the creation and processing of mesh data for tRIBS simulations. It handles preprocessing of
    watershed and stream network data, and integrates with mesh generation routines. The class is initialized with
    various parameters and options, and it supports setting up mesh-related attributes based on input files or
    provided arguments.

    Attributes
    ----------
    pointfilename : str
        The name of the file containing the mesh points.
    graphfile : str
        The name of the file containing the mesh graph.
    optmeshinput : int
        Option flag for mesh input processing.
    graphoption : int
        Option for graph generation.
    demfile : str
        The name of the file containing the Digital Elevation Model (DEM) data.
    preprocess : Preprocess, optional
        An instance of the Preprocess class used for initial data extraction and processing.
    mesh_generator : GenerateMesh, optional
        An instance of the GenerateMesh class used for mesh generation.

    Parameters
    ----------
    preprocess_args : tuple, optional
        Arguments for initializing the Preprocess class. Required if `generate_mesh_args` is provided.
    generate_mesh_args : tuple, optional
        Arguments for initializing the GenerateMesh class. Must be provided if `preprocess_args` is given.
    input_file : str, optional
        Path to the input file for initializing attributes.
    meta : dict, optional
        Metadata associated with the mesh.

    Example
    -------
    To create and use an instance of the `Mesh` class:

    >>> mesh = Mesh(preprocess_args=(arg1, arg2, arg3), generate_mesh_args=(arg4, arg5, arg6, arg7))
    >>> print(mesh.pointfilename)
    'path/to/pointfile'
    >>> print(mesh.demfile)
    'path/to/demfile'
    """

    def __init__(self, preprocess_args=None, generate_mesh_args=None,
                 input_file=None, meta= None):
        Meta.__init__(self)

        if meta is not None:
            self.meta = meta

        if preprocess_args is not None:  # TODO need to catch if generate_mesh_args is NONE
            self.preprocess = Preprocess(*preprocess_args)
            _, bound_path, stream_path, out_path, _ = generate_mesh_args
            self.preprocess.extract_watershed_and_stream_network(out_path, bound_path,
                                                                 stream_path)
            self.meta = self.preprocess.meta

        if generate_mesh_args is not None:
            self.mesh_generator = GenerateMesh(*generate_mesh_args)

        if input_file is not None:
            options = SharedMixin.read_input_file(input_file)
        else:
            options = InfileMixin.create_input_file()

        # Initialize attributes
        self.pointfilename = options['pointfilename']
        self.graphfile = options['graphfile']
        self.optmeshinput = options['optmeshinput']
        self.graphoption = options['graphoption']
        self.demfile = options['demfile']


class Met(_Met):
    """
    A tRIBS Met Class.

    This class handles the meteorological data for tRIBS simulations. It initializes various parameters related to
    meteorological stations, rain files, and other related metadata. The class is used to configure and manage the
    meteorological input options required for the simulation.

    Attributes
    ----------
    hydrometstations : str
        The path or name of the file containing hydrometeorological station data.
    gaugestations : str
        The path or name of the file containing gauge station data.
    hydrometbasename : str
        The base name for hydrometeorological data files.
    rainfile : str
        The path or name of the file containing rainfall data.
    hydrometgrid : str
        The path or name of the file containing the hydrometeorological grid data.
    metdataoption : int
        Option flag for meteorological data processing.
    rainsource : str
        The source of the rainfall data.
    gaugebasename : str
        The base name for gauge data files.
    rainextension : str
        The file extension for the rainfall data files.

    Parameters
    ----------
    input_file : str, optional
        Path to the input file containing the necessary options for initializing the `Met` class attributes.
    meta : dict, optional
        Metadata associated with the `Met` instance.

    Example
    -------
    To create and use an instance of the `Met` class:

    >>> met = Met(input_file="path/to/input_file.in")
    >>> print(met.hydrometstations)
    'path/to/hydrometstations'
    >>> print(met.rainfile)
    'path/to/rainfile'
    """

    def __init__(self, input_file=None, meta=None):

        Meta.__init__(self)

        if meta is not None:
            self.meta = meta

        if input_file is not None:
            options = SharedMixin.read_input_file(input_file)
        else:
            options = InfileMixin.create_input_file()

        self.hydrometstations = options['hydrometstations']
        self.gaugestations = options['gaugestations']
        self.hydrometbasename = options['hydrometbasename']
        self.rainfile = options['rainfile']
        self.hydrometgrid = options['hydrometgrid']
        self.metdataoption = options['metdataoption']
        self.rainsource = options['rainsource']
        self.gaugebasename = options['gaugebasename']
        self.rainextension = options['rainextension']
