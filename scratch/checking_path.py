import os
from tribsmodel import Model as model
m = model()
os.chdir("HJ_BenchMark")
m.check_paths()

m.read_input_file("Template.in")
m.check_paths()