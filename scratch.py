import os
from tribsmodel import Model as model
m = model()
os.chdir("HJ_BenchMark")
m.read_input_file("Template.in")
m.Results.get_mrf_results()
m.Results.get_mrf_water_balance("water_year",0.44,15000)




