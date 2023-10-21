
from tribsmodel import Model as model

m = model()
# os.chdir("HJ_BenchMark")
m.options["outfilename"]["value"] = "HJ_BenchMark/syc_domain/ms2_"
m.geo["EPSG"] = "EPSG:32612"
reaches = m.read_reach_file()
reaches.plot(column="ID")
voi = m.read_voi_file()
#
