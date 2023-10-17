import os
import shutil
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch
#
from tribsmodel import Model as model
m = model()
#os.chdir("HJ_BenchMark")
m.options["outfilename"]["value"] = "HJ_BenchMark/syc_domain/ms2_"
m.geo["EPSG"] = "EPSG:32612"
reaches = m.read_reach_file()
reaches.plot(column="ID")
voi = m.read_voi_file()
#
# fig = px.choropleth(voi, geojson=voi.geometry, locations=voi.index, color='ID',)
#
# # Show the map
# fig.show()

# instead, extract each polygon as a descartes patch, and add to a matplotlib patch collection...
# note, this code assumes all geometries are Polygons - if you have MultiPolygons, handle separately
patches = [PolygonPatch(geometry) for geometry in voi['geometry']]
pc = PatchCollection(patches, facecolor='#3399cc', linewidth=1, alpha=0.1)
fig, ax = plt.subplots(figsize=(5,5))
ax.add_collection(pc)

# ...then set the figure bounds to the polygons' bounds
left, bottom, right, top = voi.total_bounds
ax.set_xlim((left,right))
ax.set_ylim((bottom,top))
plt.show()