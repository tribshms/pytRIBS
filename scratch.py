import os
import shutil
#
from tribsmodel import Model as model
m = model()
os.chdir("HJ_BenchMark")
m.read_input_file("Template.in")
m.write_input_file("HJ.in")
m.build("~/Documents/Repos/Forked/tribsDev/","build",exe="test")
shutil.copy2("build/tRIBS","./tRIBS")
#m.run("./tRIBS","HJ.in",mpi_command="mpirun -n 1")
#
# m.Results.get_mrf_results()
# m.Results.get_mrf_water_balance("water_year",0.44,15000,98)
#
# m.Results.get_element_results()
# m.Results.get_element_water_balance("water_year")
#

import os
from tribsmodel import Model as model
m = model()
os.chdir("HJ_BenchMark")
m.read_input_file("Template.in")
m.add_descriptor_files()