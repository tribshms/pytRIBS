import os
from tribsmodel import Model as model

m = model()

os.chdir('/Users/lraming/Documents/Repos/Sycamore')

m.read_input_file('sycMS2.in')
end_time = int(m.options["runtime"]["value"])

number = 0
out_file = m.options["outfilename"]["value"]+str(number)+"_gwvoi"
df = m.merge_parallel_spatial_files(dtime=end_time, write=False)
df = df[str(end_time)]

df[['ID', 'Nwt']].to_csv(out_file, sep='\t', index=False, header=False)
