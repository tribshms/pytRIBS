class InfileMixin:
    """
    Mixin for .in file parameters and definitions shared by both pytRIBS Classes Model & Results.
    """

    @staticmethod
    def create_input_file():
        """
        Creates a dictionary with tRIBS input options assigne to attribute input_options.

        This function loads a dictionary of the necessary variables for a tRIBS input file. And is called upon
        initialization. The dictionary is assigned as instance variable:input_options to the Class Simulation. Note
        the templateVars file will need to be udpated if additional keywords are added to the .in file.

        Each subdictionary has a tags key. With the tag indicating the role of the given option or variable in the model
        simulation.

        Tags:
        time    - parameters related to model simulation times and time steps
        mesh    - options for reading mesh
        flow    - flow routing parameters
        hydro    - options and physical parameters for hydrological components of modle
        spatial   - input raster files and tables for bedrock, groundwater, landuse, and soil properties.
        meterological    -  options for meterological data
        output    - options and paths for model outputs
        forecast - suite of options for forecast mode
        stochastic    - suite of options for stochastic mode
        restart   - options for restart functionality
        parallel  - options for parallel functionality
        """
        input_file = {

            # ==================================================================================================
            # TIME PARAMETERS
            # ==================================================================================================

            "startdate": {"key_word": "STARTDATE:", "describe": "Starting time (MM/DD/YYYY/HH/MM)", "value": None,
                          "tags": ["time"]},
            "runtime": {"key_word": "RUNTIME:", "describe": "Simulation length in hours", "value": None,
                        "tags": ["time"]},
            "rainsearch": {"key_word": "RAINSEARCH:", "describe": "Rainfall search interval (hours)", "value": 24,
                           "tags": ["time"]},
            "timestep": {"key_word": "TIMESTEP:", "describe": "Unsaturated zone computational time step (mins)",
                         "value": 3.75, "tags": ["time"]},
            "gwstep": {"key_word": "GWSTEP:", "describe": "Saturated zone computational time step (mins)",
                       "value": 30.0, "tags": ["time"]},
            "metstep": {"key_word": "METSTEP:", "describe": "Meteorological data time step (mins)", "value": 60.0,
                        "tags": ["time"]},
            "etistep": {"key_word": "ETISTEP:", "describe": "ET, interception and snow time step (hours)", "value": 1,
                        "tags": ["time"]},
            "rainintrvl": {"key_word": "RAININTRVL:", "describe": "Time interval in rainfall input (hours)",
                           "value": 1, "tags": ["time"]},
            "intstormmax": {"key_word": "INTSTORMMAX:", "describe": "Interstorm interval (hours)", "value": 10000,
                            "tags": ["time"]},
            # ==================================================================================================
            # MESH PARAMETERS
            # ==================================================================================================
            "optmeshinput": {"key_word": "OPTMESHINPUT:", "describe": "Mesh input data option\n" + \
                                                                      "1  tMesh data\n" + \
                                                                      "2  Point file\n" + \
                                                                      "3  ArcGrid (random)\n" + \
                                                                      "4  ArcGrid (hex)\n" + \
                                                                      "5  Arc/Info *.net\n" + \
                                                                      "6  Arc/Info *.lin,*.pnt\n" + \
                                                                      "7  Scratch\n" + \
                                                                      "8  Point Triangulator", "value": 8,
                             "tags": ["mesh"]},
            "inputdatafile": {"key_word": "INPUTDATAFILE:", "describe": "tMesh input file base name for Mesh files",
                              "value": None, "tags": ["mesh"]},
            "inputtime": {"key_word": "INPUTTIME:", "describe": "Deprecated, will be removed", "value": None, "tags": ["mesh"]},
            "arcinfofilename": {"key_word": "ARCINFOFILENAME:", "describe": "tMesh input file base name Arc files",
                                "value": None, "tags": ["mesh"]},
            "pointfilename": {"key_word": "POINTFILENAME:", "describe": "tMesh input file name Points files",
                              "value": None, "tags": ["mesh"]},
            # ==================================================================================================
            # ROUTING PARAMETERS
            # ==================================================================================================

            "baseflow": {"key_word": "BASEFLOW:", "describe": "Baseflow discharge (m3/s)", "value": 0.2,
                         "tags": ["flow"]},
            "velocitycoef": {"key_word": "VELOCITYCOEF:", "describe": "Discharge-velocity coefficient", "value": 1.2,
                             "tags": ["flow"]},
            "kinemvelcoef": {"key_word": "KINEMVELCOEF:", "describe": "Kinematic routing velocity coefficient",
                             "value": 3, "tags": ["flow"]},
            "velocityratio": {"key_word": "VELOCITYRATIO:", "describe": "Stream to hillslope velocity coefficient",
                              "value": 60, "tags": ["flow"]},
            "flowexp": {"key_word": "FLOWEXP:", "describe": "Nonlinear discharge coefficient", "value": 0.3,
                        "tags": ["flow"]},
            "channelroughness": {"key_word": "CHANNELROUGHNESS:", "describe": "Uniform channel roughness value",
                                 "value": 0.15, "tags": ["flow"]},
            "channelwidth": {"key_word": "CHANNELWIDTH:", "describe": "Uniform channel width  (meters)", "value": 12,
                             "tags": ["flow"]},
            "channelwidthcoeff": {"key_word": "CHANNELWIDTHCOEFF:",
                                  "describe": "Coefficient in width-area relationship", "value": 2.33,
                                  "tags": ["flow"]},
            "channelwidthexpnt": {"key_word": "CHANNELWIDTHEXPNT:", "describe": "Exponent in width-area relationship",
                                  "value": 0.54, "tags": ["flow"]},
            "channelwidthfile": {"key_word": "CHANNELWIDTHFILE:", "describe": "Filename that contains channel widths",
                                 "value": None, "tags": ["flow"]},
            "widthinterpolation": {"key_word": "WIDTHINTERPOLATION:",
                                   "describe": "Option for interpolating width values", "value": 0, "tags": ["flow"]},

            # ==================================================================================================
            # HYDROLOGIC PARAMETERS
            # ==================================================================================================

            "optevapotrans": {"key_word": "OPTEVAPOTRANS:", "describe": "Option for evapoTranspiration scheme\n" + \
                                                                        "0  Inactive evapotranspiration\n" + \
                                                                        "1  Penman-Monteith method\n" + \
                                                                        "2  Deardorff method\n" + \
                                                                        "3  Priestley-Taylor method\n" + \
                                                                        "4  Pan evaporation measurements", "value": 1,
                              "tags": ["hydro"]},
            "optintercept": {"key_word": "OPTINTERCEPT:", "describe": "Option for interception scheme\n" + \
                                                                      "0  Inactive interception\n" + \
                                                                      "1  Canopy storage method\n" + \
                                                                      "2  Canopy water balance method", "value": 2,
                             "tags": ["hydro"]},
            "optlanduse": {"key_word": "OPTLANDUSE:", "describe": "Option for static or dynamic land cover\n" + \
                                                                  "0  Static land cover maps\n" + \
                                                                  "1  Dynamic updating of land cover maps", "value": 0,
                           "tags": ["hydro"]},
            "optluinterp": {"key_word": "OPTLUINTERP:", "describe": "Option for interpolation of land cover\n" + \
                                                                    "0  Constant (previous) values between land cover\n" + \
                                                                    "1  Linear interpolation between land cover",
                            "value": 1, "tags": ["hydro"]},
            "optsoiltype": {"key_word": "OPTSOILTYPE:", "describe": "Option for soil input type: 0 soil table, 1 soil "
                                                                    "rasters\n See SCGRID, SOILTABLENAME, and SOILMAPNAME",
                            "value": 0, "tags": ["hydro"]},
            "gfluxoption": {"key_word": "GFLUXOPTION:", "describe": "Option for ground heat flux\n" + \
                                                                    "0  Inactive ground heat flux\n" + \
                                                                    "1  Temperature gradient method\n" + \
                                                                    "2  Force_Restore method", "value": 2,
                             "tags": ["hydro"]},
            "optbedrock": {"key_word": "OPTBEDROCK:", "describe": "Option for uniform or variable depth\n"
                                                                  "0 Uniform bedrock depth\n"
                                                                  "1 Spatial grid of bedrock depth\n"
                                                                  "See DEPTHTOBEDROCK and BEDROCKFILE\n", "value": 0,
                           "tags": ["hydro"]},
            "optgroundwater": {"key_word": "OPTGROUNDWATER","describe": "Option for groundwater module, 0 off, 1 on",
                                "value": 1, "tags": ['hydro']},

            "optgwfile": {"key_word": "OPTGWFILE:", "describe": "Option for groundwater initial file\n" + \
                                                                "0 Resample ASCII grid file in GWATERFILE\n" + \
                                                                "1 Read in Voronoi polygon file with GW levels",
                          "value": 0, "tags": ["hydro"]},

            "optrunon": {"key_word": "OPTRUNON:", "describe": "Option for runon in overland flow paths [IN DEVELOPMENT]: 0 off, 1 on", "value": 0,
                         "tags": ["hydro"]},
            "optreservoir": {"key_word": "OPTRESERVOIR:", "describe": 'Option for leve pool routing: 0 off, 1 on',
                             "value": 0, "tags": ["hydro"]},
            ### added
            "respolygonid": {"key_word": "RESPOLYGONID:", "describe": "Path to file of node IDs representing reservoirs",
                             "value":None, "tags": ["hydro"]},
            "resdata": {"key_word": "RESDATA:", "describe": "Path to file of elevation-discharge-storage information for each type of reservoir",
                        "value":None,"tags": ["hydro"]},

            "optsnow": {"key_word": "OPTSNOW:", "describe": "Enable single layer snow module : 0 off, 1 on", "value": 1,
                        "tags": ["hydro"]},
            "minsntemp": {"key_word": "MINSNTEMP:", "describe": "Minimum snow temperature", "value": -50.0,
                          "tags": ["hydro"]},
            "snliqfrac": {"key_word": "SNLIQFRAC:", "describe": "Maximum fraction of liquid water in snowpack",
                          "value": 0.065, "tags": ["hydro"]},
            "depthtobedrock": {"key_word": "DEPTHTOBEDROCK:", "describe": "Uniform depth to bedrock (meters), see OPTBEDROCK",
                               "value": 15, "tags": ["hydro"]},
            "optpercolation": {"key_word": "OPTPERCOLATION:", "describe": "Option to for percolation losses from channels\n"
                               "0 off\n1 on"

                , "value": 0,
                               "tags": ["hydro"]},
            "channelconductivity": {"key_word": "CHANNELCONDUCTIVITY:", "describe": "Conductivity in channel", "value": 0,
                                    "tags": ["hydro"]},
            "transientconductivity": {"key_word": "TRANSIENTCONDUCTIVITY:", "describe": "Conductivity during transient period",
                                      "value": 0, "tags": ["hydro"]},
            "transienttime": {"key_word": "TRANSIENTTIME:", "describe": "Time until transient period ends", "value": 0,
                              "tags": ["hydro"]},
            "channelporosity": {"key_word": "CHANNELPOROSITY:", "describe": "Porosity in channel", "value": 0,
                                "tags": ["hydro"]},
            "chanporeindex": {"key_word": "CHANPOREINDEX:", "describe": "Channel pore index in channel", "value": 0,
                              "tags": ["hydro"]},
            "chanpsib": {"key_word": "CHANPSIB:", "describe": "Matric potential in channel", "value": 0, "tags": ["hydro"]},
            # ==================================================================================================
            # GRID RESAMPLE
            # ==================================================================================================

            "lugrid": {"key_word": "LUGRID:", "describe": "Land cover grid data file (*.gdf)", "value": None,
                       "tags": ["spatial"]},
            "scgrid": {"key_word": "SCGRID:", "describe": "Soil grid data file (*.gdf). Note OPTSOILTYPE must = 1 if "
                                                          "inputing soil grids", "value": None,
                       "tags": ["spatial"]},
            "soiltablename": {"key_word": "SOILTABLENAME:", "describe": "Soil parameter reference table (*.sdt)",
                              "value": None, "tags": ["spatial"]},
            "soilmapname": {"key_word": "SOILMAPNAME:", "describe": "Soil texture ASCII grid (*.soi)", "value": None,
                            "tags": ["spatial"]},
            "landtablename": {"key_word": "LANDTABLENAME:", "describe": "Land use parameter reference table",
                              "value": None, "tags": ["spatial"]},
            "landmapname": {"key_word": "LANDMAPNAME:", "describe": "Land use ASCII grid (*.lan)", "value": None,
                            "tags": ["spatial"]},
            "gwaterfile": {"key_word": "GWATERFILE:", "describe": "Ground water ASCII grid (*iwt)", "value": None,
                           "tags": ["spatial"]},
            "bedrockfile": {"key_word": "BEDROCKFILE:", "describe": "Bedrock depth ASCII grid (*.brd), see OPTBEDROCK", "value": None,
                            "tags": ["spatial"]},
            "demfile": {"key_word": "DEMFILE:", "describe": "DEM ASCII grid for sky and land view factors (*.dem)",
                        "value": None, "tags": ["spatial"]},

            # ==================================================================================================
            # METEROLOGICAL DATA
            # ==================================================================================================
            "rainsource": {"key_word": "RAINSOURCE:", "describe": "Rainfall data source option\n" +
                                                                  "1  Stage III radar\n" +
                                                                  "2  WSI radar\n" +
                                                                  "3  Rain gauges", "value": 3, "tags": ["meterological"]},
            "rainfile": {"key_word": "RAINFILE:", "describe": "Base name of the radar ASCII grid", "value": None,
                         "tags": ["meterological"]},
            "rainextension": {"key_word": "RAINEXTENSION:", "describe": "Extension for the radar ASCII grid",
                              "value": None, "tags": ["meterological"]},
            "metdataoption": {"key_word": "METDATAOPTION:", "describe": "Option for meteorological data\n" + \
                                                                        "0  Inactive meteorological data\n" + \
                                                                        "1  Weather station point data\n" +
                                                                        "2  Gridded meteorological data", "value": 1,
                              "tags": ["meterological"]},
            "hydrometstations": {"key_word": "HYDROMETSTATIONS:",
                                 "describe": "Hydrometeorological station file (*.sdf)", "value": None, "tags": ["meterological"]},
            "hydrometgrid": {"key_word": "HYDROMETGRID:", "describe": "Hydrometeorological grid data file (*.gdf)",
                             "value": None, "tags": ["meterological"]},
            "hydrometconvert": {"key_word": "HYDROMETCONVERT:",
                                "describe": "Hydrometeorological data conversion file (*.mdi)", "value": None,
                                "tags": ["meterological"]},
            "hydrometbasename": {"key_word": "HYDROMETBASENAME:",
                                 "describe": "Hydrometeorological data BASE name (*.mdf)", "value": None,
                                 "tags": ["meterological"]},
            "gaugestations": {"key_word": "GAUGESTATIONS:", "describe": "Rain Gauge station file (*.sdf)",
                              "value": None, "tags": ["meterological"]},
            "gaugeconvert": {"key_word": "GAUGECONVERT:", "describe": "Rain Gauge data conversion file (*.mdi)",
                             "value": None, "tags": ["meterological"]},
            "gaugebasename": {"key_word": "GAUGEBASENAME:", "describe": "Rain Gauge data BASE name (*.mdf)",
                              "value": None, "tags": ["meterological"]},
            "outhydroextension": {"key_word": "OUTHYDROEXTENSION:", "describe": "Extension for hydrograph output",
                                  "value": "mrf", "tags": ["meterological"]},
            "ribshydoutput": {"key_word": "RIBSHYDOUTPUT:", "describe": "Compatibility with RIBS User Interphase,\nDepracted and will be removed.",
                              "value": 0, "tags": ["meterological"]},
            "convertdata": {"key_word": "CONVERTDATA:", "describe": "Option to convert met data format", "value": 0,
                            "tags": ["meterological"]},
            "templapse": {"key_word": "TEMPLAPSE:", "describe": "Temperature lapse rate", "value": -0.0065,
                          "tags": ["meterological"]},
            "preclapse": {"key_word": "PRECLAPSE:", "describe": "Precipitation lapse rate", "value": 0,
                          "tags": ["meterological"]},
            "hillalbopt": {"key_word": "HILLALBOPT:", "describe": "Option for albedo of surrounding hillslopes\n" + \
                                                                  "0  Snow albedo for hillslopes\n" + \
                                                                  "1  Land-cover albedo for hillslopes\n" + \
                                                                  "2  Dynamic albedo for hillslopes", "value": 0,
                           "tags": ["meterological"]},
            "optradshelt": {"key_word": "OPTRADSHELT:", "describe": "Option for local and remote radiation sheltering" +
                                                                    "0  Local controls on shortwave radiation\n" + \
                                                                    "1  Remote controls on diffuse shortwave\n" + \
                                                                    "2  Remote controls on entire shortwave\n" + \
                                                                    "3  No sheltering", "value": 0, "tags": ["meterological"]},
            "tlinke": {"key_word": "TLINKE:", "describe": "Atmospheric turbidity parameter", "value": 2.5,
                       "tags": ["meterological"]},
            # ==================================================================================================
            # OUTPUT DATA
            # ==================================================================================================
            "outfilename": {"key_word": "OUTFILENAME:", "describe": "Base name of the tMesh and variable",
                            "value": None, "tags": ["output"]},
            "outhydrofilename": {"key_word": "OUTHYDROFILENAME:", "describe": "Base name for hydrograph output",
                                 "value": None, "tags": ["output"]},
            "optspatial": {"key_word": "OPTSPATIAL:", "describe": "Enable dynamic spatial output", "value": 0,
                           "tags": ["output"]},
            "optinterhydro": {"key_word": "OPTINTERHYDRO:", "describe": "Enable intermediate hydrograph output",
                              "value": 0, "tags": ["output"]},
            "optheader": {"key_word": "OPTHEADER:", "describe": "Enable headers in output files", "value": 1,
                          "tags": ["output"]},
            "optviz": {"key_word": "OPTVIZ:", "describe": "Option to write binary output files for visualization\n" + \
                                                          "0  Do NOT write binary output files for viz\n" + \
                                                          "1  Write binary output files for viz", "value": 0,
                       "tags": ["output"]},
            "outvizfilename": {"key_word": "OUTVIZFILENAME:", "describe": "Filename for viz binary files",
                               "value": None, "tags": ["output"]},
            "nodeoutputlist": {"key_word": "NODEOUTPUTLIST:",
                               "describe": "Filename with Nodes for Dynamic Output (*.nol)", "value": None,
                               "tags": ["output"]},
            "hydronodelist": {"key_word": "HYDRONODELIST:",
                              "describe": "Filename with Nodes for HydroModel Output (*.nol)", "value": None,
                              "tags": ["output"]},
            "outletnodelist": {"key_word": "OUTLETNODELIST:",
                               "describe": "Filename with Interior Nodes for  Output (*.nol)", "value": None,
                               "tags": ["output"]},
            "opintrvl": {"key_word": "OPINTRVL:", "describe": "Output interval (hours)", "value": 1, "tags": ["output"]},
            "spopintrvl": {"key_word": "SPOPINTRVL:", "describe": "Spatial output interval (hours)", "value": 50000,
                           "tags": ["output"]},
            # ==================================================================================================
            # FORECAST MODE
            # ==================================================================================================
            "forecastmode": {"key_word": "FORECASTMODE:", "describe": "Rainfall Forecasting Mode Option", "value": 0,
                             "tags": ["forecast"]},
            # TODO need to update model mode descriptions
            "forecasttime": {"key_word": "FORECASTTIME:", "describe": "Forecast Time (hours from start)", "value": 0,
                             "tags": ["forecast"]},
            "forecastleadtime": {"key_word": "FORECASTLEADTIME:", "describe": "Forecast Lead Time (hours) ",
                                 "value": 0, "tags": ["forecast"]},
            "forecastlength": {"key_word": "FORECASTLENGTH:", "describe": "Forecast Window Length (hours)", "value": 0,
                               "tags": ["forecast"]},
            "forecastfile": {"key_word": "FORECASTFILE:", "describe": "Base name of the radar QPF grids",
                             "value": None, "tags": ["forecast"]},
            "climatology": {"key_word": "CLIMATOLOGY:", "describe": "Rainfall climatology (mm/hr)", "value": 0,
                            "tags": ["forecast"]},
            "raindistribution": {"key_word": "RAINDISTRIBUTION:", "describe": "Distributed or MAP radar rainfall",
                                 "value": 0, "tags": ["forecast"]},
            # ==================================================================================================
            # STOCHASTIC MODE
            # ==================================================================================================
            "stochasticmode": {"key_word": "STOCHASTICMODE:", "describe": "Stochastic Climate Mode Option", "value": 0,
                               "tags": ["stochastic"]},
            "pmean": {"key_word": "PMEAN:", "describe": "Mean rainfall intensity (mm/hr)	", "value": 0,
                      "tags": ["stochastic"]},
            "stdur": {"key_word": "STDUR:", "describe": "Mean storm duration (hours)", "value": 0,
                      "tags": ["stochastic"]},
            "istdur": {"key_word": "ISTDUR:", "describe": "Mean time interval between storms (hours)", "value": 0,
                       "tags": ["stochastic"]},
            "seed": {"key_word": "SEED:", "describe": "Random seed", "value": 0, "tags": ["stochastic"]},
            "period": {"key_word": "PERIOD:", "describe": "Period of variation (hours)", "value": 0,
                       "tags": ["stochastic"]},
            "maxpmean": {"key_word": "MAXPMEAN:", "describe": "Maximum value of mean rainfall intensity (mm/hr)",
                         "value": 0, "tags": ["stochastic"]},
            "maxstdurmn": {"key_word": "MAXSTDURMN:", "describe": "Maximum value of mean storm duration (hours)",
                           "value": 0, "tags": ["stochastic"]},
            "maxistdurmn": {"key_word": "MAXISTDURMN:", "describe": "Maximum value of mean interstorm period (hours)",
                            "value": 0, "tags": ["stochastic"]},
            "weathertablename": {"key_word": "WEATHERTABLENAME:", "describe": "File with Stochastic Weather Table",
                                 "value": None, "tags": ["stochastic"]},
            # ==================================================================================================
            # RESTART MODE
            # ==================================================================================================
            "restartmode": {"key_word": "RESTARTMODE:", "describe": "Restart Mode Option\n" + \
                                                                    "0 No reading or writing of restart\n" + \
                                                                    "1 Write files (only for initial runs)\n" + \
                                                                    "2 Read file only (to start at some specified time)\n" + \
                                                                    "Read a restart file and continue to write",
                            "value": 0, "tags": ["restart"]},
            "restartintrvl": {"key_word": "RESTARTINTRVL:", "describe": "Time set for restart output (hours)",
                              "value": None, "tags": ["restart"]},
            "restartdir": {"key_word": "RESTARTDIR:", "describe": "Path of directory for restart output",
                           "value": None, "tags": ["restart"]},
            "restartfile": {"key_word": "RESTARTFILE:", "describe": "Actual file to restart a run", "value": None,
                            "tags": ["restart"]},
            # ==================================================================================================
            # PARALLEL MODE
            # ==================================================================================================
            "parallelmode": {"key_word": "PARALLELMODE:", "describe": "Parallel or Serial Mode Option\n" + \
                                                                      "0  Run in serial mode\n" + \
                                                                      "1  Run in parallel mode",
                             "value": 0, "tags": ["parallel"]},
            "graphoption": {"key_word": "GRAPHOPTION:", "describe": "Graph File Type Option\n" + \
                                                                    "0  Default partitioning of the graph\n" + \
                                                                    "1  Reach-based partitioning\n" + \
                                                                    "2  Inlet/outlet-based partitioning", "value": 0,
                            "tags": ["parallel"]},
            "graphfile": {"key_word": "GRAPHFILE:", "describe": "Reach connectivity filename (graph file option 1,2)",
                          "value": None, "tags": ["parallel"]}
        }

        return input_file
