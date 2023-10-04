# TIN-Based Real-Time Integrated Basin Simulator single point benchmark

This repository hosts the setup for executing a single point model of the [Happy Jack Snotel Site](https://wcc.sc.egov.usda.gov/nwcc/site?sitenum=969) in Northern Arizona, USA, using the TIN-Based Real-Time Integrated Basin Simulator ([tRIBS](https://tribshms.readthedocs.io/en/latest/)).

## The structure of this repository is as follows:
### data
Contains all the necessary data to run tRIBS at Happy Jack and includes Snotel and SWANN data as a calibration/validation set.
### doc 
Contains relevant documentation for running this specific benchmark case and other information regarding working with this repository, including requirements and instructions for building tRIBS.
### src
Is designed to contain source code for for the tRIBS executable, which can be obtained [here](https://github.com/tribshms/tribs_sub2020).
### bin
Directory for building and storing tRIBS executable, with instructions [here](doc/CMake.md).
### results
Directory for results from the Happy Jack point tRIBS simulation. 

<ins>Note:</ins> There is no content in **bin** and **results**. These directories are intentionally left empty as they should be populated through building and running the model which can be further described in[doc](/doc). 

<!--- The content of this folder is designed to test tRIBS performance against SNOTEL data at the [Happy Jack](https://wcc.sc.egov.usda.gov/nwcc/site?sitenum=969) snotel site. This bench is based of the graduate work of Gretchen Hawkins and Josh Cederstrom.

Snotel data from:

Sun N, H Yan, M Wigmosta, R Skaggs, R Leung, and Z Hou. 2019. “Regional snow parameters estimation for large-domain hydrological applications in the western United States.” Journal of Geophysical Research: Atmospheres. doi: 10.1029/2018JD030140

Yan H, N Sun, M Wigmosta, R Skaggs, Z Hou, and R Leung. 2018. “Next-generation intensity-duration-frequency curves for hydrologic design in snow-dominated environments.” Water Resources Research, 54(2), 1093–1108.
BCQC Data Format

The file of each SNOTEL station is named as “bcqc_<latitude>_<longitude>.txt” (i.e., bcqc_44.43000_-120.33000.txt). In each text file, there are 8 columns separated by a space delimiter:

1st col: year
2nd col: month
3rd col: day
4th col: daily precipitation, in inch
5th col: maximum air temperature, in °F
6th col: minimum air temperature, in °F
7th col: mean air temperature, in °F
8th col: SWE, in inch, reset to zero on 1 October at the start of each water year
In each text file, “nan” indicates missing or filtered out data after the BCQC procedures. 

Note: For each SNOTEL station, you can find its information (based on latitude and longitude) in the summary file, which details the SNOTEL ID, location (in latitude and longitude), located state, elevation, name, data start date, and end date. 


Within these cases there are mutliple different options that can be turned 
on/off in the input file:
 - Sheltering DEM (OPTRADSHELT)
 - Gridded vegetation parameters (OPTLANDUSE)
	- You must have option, OPTLUINTERP set to 1 for this to work
--->
