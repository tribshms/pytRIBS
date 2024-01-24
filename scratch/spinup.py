import sys
from datetime import datetime
import os


extra_path = "/"
if extra_path not in sys.path:
    sys.path.append(extra_path)

from tribsmodel import Model as model

m = model()
# setup experiment vars
input_file = 'sycMS2.in'
base_name = 'ms2_'

m.read_input_file(input_file)

# gw voi file from intial run will want to update in future
m.options["optgwfile"]["value"] = 1
gwvoi = "results/2023-10-19/ms2_0_gwvoi"
end_time = int(m.options["runtime"]["value"])

# Get today's date and make results directory
today = datetime.now().date()
today = today.strftime('%Y-%m-%d')
todays_results = 'results/' + today + '/'
os.mkdir(todays_results)

# currently just running spinup for 3 years total, but maybe better approach.

iterations = [1, 2]

for i in iterations:
    result_path = todays_results + str(i)
    os.mkdir(result_path)
    outfiles = result_path + '/' + base_name
    m.options["outfilename"]["value"] = outfiles
    m.options["outhydrofilename"]["value"] = outfiles
    m.options['gwaterfile']['value'] = gwvoi
    m.write_input_file(input_file)
    m.run('./tRIBS', input_file, mpi_command='mpirun -n 4')

    # update gwvoi file
    gwvoi = m.options["outfilename"]["value"] + str(i) + "_gwvoi"

    # merger parallel files and write ouyt gwvoi file
    df = m.merge_parallel_spatial_files(dtime=end_time, write=False)
    df = df[str(end_time)]
    df[['ID', 'Nwt']].to_csv(gwvoi, sep='\t', index=False, header=False)

