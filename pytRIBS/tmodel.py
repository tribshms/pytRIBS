
from pytRIBS.mixins.infile_mixin import InfileMixin
from pytRIBS.mixins.shared_mixin import SharedMixin
from pytRIBS.model.aux import Aux
from pytRIBS.model.inout import InOut
from pytRIBS.model.diagnose import Diagnostics



class Model(InfileMixin, SharedMixin, Aux, InOut, Diagnostics):
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
        self.area = None

    # SIMULATION METHODS
    def __getattr__(self, name):
        if name in self.options:
            return self.options[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __dir__(self):
        # Include the keys from the options dictionary and the methods of the class
        return list(
            set(super().__dir__() + list(self.options.keys()))) if self.options is not None else super().__dir__()
