import sys
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime

extra_path = f"{os.environ['HOME']}/Documents/Repos/tP4"
if extra_path not in sys.path:
    sys.path.append(extra_path)

from pytRIBS.tresults import Results as results

os.chdir('/Users/lraming/Library/CloudStorage/GoogleDrive-wren.raming@gmail.com/My Drive/Work/Hydro/tRIBS/resource-code-testing/Benchmarks/SMALL_BENCH/')
infile = 'in_files/run_A.in'
r = results(infile)
r.get_mrf_results()
r.get_mrf_water_balance('full')