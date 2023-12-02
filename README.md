# tP4
tRIBS pre-to-post procsessing package is a python module designed to allow users to setup, simulate, and analyze TIN-based Real-time Integrated Basin Simulator model runs though a python interface. 

### TODO
- put in prerpocessing pre-model run checks
  * check to make sure that the number of parameters in first line matches the number of parameters in file for data files
  * check that theta*_t and theta*_s are within range of theta_s and theta_r
  * check that generic restart file path is correct, currently says it doesnt exist even when it does but because it has the paralle .# suffix it doesn't recognize it.
- post-processing:
  * After Merge calls compress parallel files--should save tons on memory storage and remove clutter from results.
