# system packages

# external packages

# pytRIBS
from pytRIBS.shared.aux import Aux
from pytRIBS.model.diagnose import Diagnostics
from pytRIBS.model.preprocess import Preprocess
from pytRIBS.shared.infile_mixin import InfileMixin
from pytRIBS.shared.shared_mixin import SharedMixin, Meta
from pytRIBS.results.waterbalance import WaterBalance
from pytRIBS.results.read import Read
from pytRIBS.results.visualize import Viz
from pytRIBS.results.evaluate import Evaluate

# preprocessing componets
from pytRIBS.soil.soil import _Soil
from pytRIBS.met.met import _Met


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

    def __init__(self, input_file=None, met=None, land=None, soil=None, mesh=None):
        # attributes
        self.options = self.create_input_file()  # input options for tRIBS model run

        if input_file is not None:
            self.read_input_file(input_file)

        self.meta = {"UTM_Zone": None, "EPSG": None, "Projection": None}
        Meta.__init__(self)

        # Initialize with provided instances
        self.met = met
        self.land = land
        self.soil = soil
        self.mesh = mesh

        # Merge options from provided instances
        if met:
            self.options.update(met.__dict__)
        if land:
            self.options.update(land.__dict__)
        if soil:
            self.options.update(soil.__dict__)
        if mesh:
            self.options.update(mesh.__dict__)

    # SIMULATION METHODS
    def __getattr__(self, name):
        if name in self.options:
            return self.options[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __dir__(self):
        # Include the keys from the options dictionary and the methods of the class
        return list(
            set(super().__dir__() + list(self.options.keys()))) if self.options is not None else super().__dir__()


class Results(InfileMixin, SharedMixin, WaterBalance, Read, Viz, Evaluate):
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
        Meta.__init__(self)

        self.get_invariant_properties()  # shared


class Soil(_Soil):
    """
    A tRIBS Soil Class.

    """

    def __init__(self, input_file=None):

        Meta.__init__(self)

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

class Land():
    """
    A tRIBS Land Class.

    """

    def __init__(self, input_file=None):

        Meta.__init__(self)

        if input_file is not None:
            options = SharedMixin.read_input_file(input_file)
        else:
            options = InfileMixin.create_input_file()

        # Initialize attributes
        self.landmapname = options['landmapname']
        self.landtablename = options['landtablename']
        self.lugrid = options['lugrid']
        self.optlanduse = options['optlanduse']
        self.optluintercept = options['optluintercept']


class Mesh():
    """
    A tRIBS Met Class.

    """

    def __init__(self, input_file=None):

        Meta.__init__(self)

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

    """

    def __init__(self, input_file=None):

        Meta.__init__(self)

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
        self.rainsource= options['rainsource']
        self.gaugebasename = options['gaugebasename']
        self.rainextension = options['rainextension']