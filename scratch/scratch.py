import sys
import os

os.chdir(os.path.abspath('/Users/wr/Documents/Repos/Sycamore'))

extra_path = f"{os.environ['HOME']}/Documents/Repos/tP4"
if extra_path not in sys.path:
    sys.path.append(extra_path)

from tribsmodel import Model as model

extra_path = f"{os.environ['HOME']}/Documents/Repos/Sycamore/src/python_scripts/"
if extra_path not in sys.path:
    sys.path.append(extra_path)

# paths
scenarios = 'MS4_scenario1_sol.in'
scenarios_subdir = 'one'
experiment_date = "2024-01-10"

m = model()
path = f"results/sycMS4/{experiment_date}_Sol/{scenarios_subdir}/{scenarios}"
m.read_input_file(path)
m.geo['EPSG'] = "EPSG:26912"
m.outfilename['value'] = f'results/sycMS4/{experiment_date}_Sol/{scenarios_subdir}/ms4_'
m.outhydrofilename['value'] = f'results/sycMS4/{experiment_date}_Sol/{scenarios_subdir}/hyd/ms4_'

m.Results.get_element_results()