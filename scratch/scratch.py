import os
import sys
import glob

extra_path = f"{os.environ['HOME']}/Documents/Repos/tP4"
if extra_path not in sys.path:
    sys.path.append(extra_path)

from pytRIBS.tmodel import Model as model
from pytRIBS.tresults import Results as results

m = model()

os.chdir('/Users/wr/Documents/Repos/tP4/jupyter examples/soil_preprocess_workflow/')

m.compute_ks_decay('decay_data.json')