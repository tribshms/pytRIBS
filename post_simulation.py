from simulation import Simulation
class PostSimulation(Simulation):
    """
    This package create a python class that represents a model simulation for the distributed hydrological model:
    TIN-based Real-Time Integrated Basin Simulator (i.e. tRIBS), with the intention of increasing the overall efficiency
    of post-and-pre model simulation processing as well as calibrating and experimenting with model behavior and finally
    provides the useful integration of tRIBS with python functionality including but not limited to education and
    programming tools like Jupyter Lab.
    """
    def __init__(self):

        self.simulation_id = run_id
        self.simulation_control_file = simulation_control_file
        self.input_options = None
        self.read_input_vars()
        var = self.input_options['startdate']
        self.startdate = self.get_input_var(var)
        var = self.input_options['outfilename']
        self.results_path = self.get_input_var(var)
        self.nodes = self.read_node_list()