# system packages

# external packages

# pytRIBS
from pytRIBS.shared.aux import Aux
from pytRIBS.model.diagnose import Diagnostics
from pytRIBS.model.preprocess import Preprocess
from pytRIBS.shared.infile_mixin import InfileMixin
from pytRIBS.shared.shared_mixin import SharedMixin, GeoMixin
from pytRIBS.results.waterbalance import WaterBalance
from pytRIBS.results.read import Read
from pytRIBS.results.visualize import Viz

#preprocessing componets
from pytRIBS.soil.soil import _Soil

class Model(InfileMixin, SharedMixin, Aux, Diagnostics, Preprocess):
    """
    A tRIBS Model class.

    This class provides access to the underlying framework of a tRIBS (TIN-based Real-time Integrated Basin
    Simulator) simulation. It includes one nested class: Results. The Model class is initialized at the top-level to
    facilitate model setup, simultation, post-processing and can be used for mainpulating and generating multiple
    simulations in an efficient manner.

    Attributes:
        input_options (dict): A dictionary of the necessary keywords for a tRIBS .in file.
        model_input_file (str): Path to a template .in file with the specified paths for model results, inputs, etc.

    Methods:
         __init__(self, model_input_file):  # Constructor method
            :param model_input_file: Path to a template .in file.
            :type param1: str

        create_input(self):
             Creates a dictionary with tRIBS input options assigne to attribute input_options.

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


    """

    def __init__(self):
        # attributes
        self.options = self.create_input_file()  # input options for tRIBS model run
        self.geo = {"UTM_Zone": None, "EPSG": None, "Projection": None}
        GeoMixin.__init__(self)
        #self.area = None

    # SIMULATION METHODS
    def __getattr__(self, name):
        if name in self.options:
            return self.options[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __dir__(self):
        # Include the keys from the options dictionary and the methods of the class
        return list(
            set(super().__dir__() + list(self.options.keys()))) if self.options is not None else super().__dir__()



class Results(InfileMixin, SharedMixin, WaterBalance, Read, Viz):
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
        GeoMixin.__init__(self)

        self.get_invariant_properties()  # shared

class Soil(_Soil):
    """
    A tRIBS Soil Class.

    """
    def __init__(self, input_file=None):

        GeoMixin.__init__(self)

        if input_file is not None:
            options = SharedMixin.read_input_file(input_file)
        else:
            options = InfileMixin.create_input_file()

        # Initialize attributes
        self.soil_class_map = options['soilmapname']
        self.soil_table = options['soiltablename']
        self.soil_gdf = options['scgrid']
        self.soil_opts = [options['optsoiltype'], options['optgroundwater'], options['optgwfile'], options['optbedrock']]
        self.bed_rock_map = options['bedrockfile']
        self.initial_gw_map = options['gwaterfile']



class Land():
    """
    A tRIBS Land Class.

    """

    def __init__(self, input_file=None):

        GeoMixin.__init__(self)

        if input_file is not None:
            options = SharedMixin.read_input_file(input_file)
        else:
            options = InfileMixin.create_input_file()

        # Initialize attributes
        self.land_class_map = options['landmapname']
        self.land_table = options['landtablename']
        self.land_gdf = options['lugrid']
        self.land_opts = {options['optlanduse'], options['optluintercept']}

class Mesh():
    """
    A tRIBS Met Class.

    """

    def __init__(self, input_file=None):

        GeoMixin.__init__(self)

        if input_file is not None:
            options = SharedMixin.read_input_file(input_file)
        else:
            options = InfileMixin.create_input_file()

        # Initialize attributes
        self.point_file = options['pointfilename']
        self.graph_file = options['graphfile']
        # what about .edge, .node .tri .z
        self.mesh_opts = [options['optmeshinput'], options['graphoption']]
        self.dem_file = options['demfile']
        #self.outlet =

class Met():
    """
    A tRIBS Met Class.

    """
    def __init__(self,input_file=None):

        GeoMixin.__init__(self)

        if input_file is not None:
            options = SharedMixin.read_input_file(input_file)
        else:
            options = InfileMixin.create_input_file()

        self.weather_sdf = options['hydrometstations']
        self.precip_sdf = options['gaugestations']
        self.precip_radar = options['rainfile']
        self.weather_gdf = options['hydrometgrid']
        self.met_opts = {options['metdataoption'],options['rainsource'],options['hydrometbasename'],options['gaugebasename'],options['rainextension']}


