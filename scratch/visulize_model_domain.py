import os
from tribsmodel import Model as model
m = model()

os.chdir("/Users/lraming/Documents/Repos/Sycamore")

print(os.getcwd())

m.geo['EPSG'] = "EPSG:26912"
m.read_input_file('pytRIBS/infiles/scenarios/ms2/MS2_scenario1_sol.in')
m.outfilename['value'] = "results/sycMS2/2023-11-28_Sol/one/ms2_"

m.Results.get_element_results()


