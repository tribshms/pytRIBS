import os
from tribsmodel import Model as model
m = model()

os.chdir("/Users/lraming/Documents/Repos/Sycamore")

print(os.getcwd())

m.geo['EPSG'] = "EPSG:26912"
m.read_input_file('src/infiles/sycMS2.in')

#print(m.outfilename)

dyn = m.merge_parllel_spatial_files()

m.merge_parllel_voi(join=dyn["35040"])




# # os.chdir("HJ_BenchMark")
# m.options["outfilename"]["value"] = "HJ_BenchMark/syc_domain/ms2_"
# m.geo["EPSG"] = "EPSG:32612"
# reaches = m.read_reach_file()
# reaches.plot(column="ID")
# voi = m.read_voi_file()
# #
