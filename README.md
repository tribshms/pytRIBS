# pytRIBS
A pre-to-post processing python package designed to allow users to setup, simulate, and analyze TIN-based Real-time Integrated Basin Simulator (tRIBS) model runs through a python interface.
Note this packages is currently under development and is subject to further changes. Additionally, much of the functionality here has had limited testing, consequently responsibility is on the user to verify package functionality. 

## Release/Version Notes
PytRIBS uses semantic versioning. Currently, we are in the initial development phase--anything MAY change at any time and
this package SHOULD NOT be considered stable.

### Version 0.2.0 (4/25/2024)
This minor update includes:
* updates to the infile_mixin, with updates for 
model documentation
* addition of Paul Tol's colormaps (https://personal.sron.nl/~pault/)
* In shared mixin:
  * added processor # to the attribute voronoi
  * added plot_mesh()
  * fixed other syntax bugs
* model.inout.py
  * added read added write_point_file()
  * fixed syntax bugs in several functions
* Fixed several bugs in preporcess.py and waterbalance.py
* Added create_animation() to Results()
