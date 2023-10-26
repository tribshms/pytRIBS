import os


def merge_parallel_files(self):
    """
    Merges output files from parallel model run.
    """

    runtime = int(self.options["runtime"]["value"])
    spopintrvl = int(self.options["spopintrvl"]["value"])
    outfilename = self.options["outfilename"]["value"]
    outhydrofilename = self.options["outhydrofilename"]["value"]

    # MERGE VORONOI
    dtime = 0
    isuffix = "_00i"

    print("Merging outputs from parallel model run: \n")

    while dtime <= runtime:
        otime = str(dtime).zfill(4)
        intFile = f"{outfilename}.{otime}{isuffix}"

        # Merge
        print(f"Merging {intFile}.* ...")
        self._private__sort_merge_file(intFile, True)

        # Next time
        dtime += spopintrvl

    wsuffix = "_width"
    widthFile = f"{outfilename}{wsuffix}"
    print(f"Merging {widthFile}.* ...")
    self._private__sort_merge_file(widthFile, False)

    asuffix = "_area"
    areaFile = f"{outfilename}{asuffix}"
    print(f"Merging {areaFile}.* ...")
    self._private__sort_merge_file(areaFile, False)

    vsuffix = "_voi"
    voiFile = f"{outfilename}{vsuffix}"
    tsuffix = ".tmp"
    tempfile = f"{voiFile}{tsuffix}"
    msuffix = ".merge"
    mergefile = f"{voiFile}{msuffix}"
    first = True
    print(f"Merging {voiFile}.* ...")

    # Merge multi-line records
    next_files = [f for f in os.listdir() if f.startswith(f"{voiFile}.")]

    for nextf in next_files:
        if first:
            os.system(f"cp {nextf} {tempfile}")
            first = False
        else:
            with open(tempfile, 'r') as tf, open(nextf, 'r') as nf, open(mergefile, 'w') as mf:
                tend = False
                nend = False
                tbuffer = next(tf).strip()
                nbuffer = next(nf).strip()

                while not (tend or nend):
                    tfields = tbuffer.split(',')
                    nfields = nbuffer.split(',')

                    if int(tfields[0]) <= int(nfields[0]):
                        mf.write(tbuffer + "\n")
                        while tbuffer != "END":
                            tbuffer = next(tf).strip()
                            mf.write(tbuffer + "\n")

                        if tbuffer == "END":
                            tend = True
                    else:
                        mf.write(nbuffer + "\n")
                        while nbuffer != "END":
                            nbuffer = next(nf).strip()
                            mf.write(nbuffer + "\n")

                        if nbuffer == "END":
                            nend = True

                if tend:
                    if not tf.closed:
                        mf.write(nbuffer + "\n")
                        mf.writelines(nf.readlines())
                else:
                    if not nf.closed:
                        mf.write(tbuffer + "\n")
                        mf.writelines(tf.readlines())

                os.system(f"mv {mergefile} {tempfile}")
    os.system(f"mv {tempfile} {voiFile}")
    # Voronoi and Mesh files are merged

    csuffix = ".cntrl"
    rsuffix = ".reach"
    cntrlFile = f"{outhydrofilename}{csuffix}"
    reachFile = f"{outhydrofilename}{rsuffix}"
    print(f"Merging {cntrlFile}.* ...")
    rmin = 1  # Minimum reach number
    rmax = 0  # Maximum reach number

    # Merge multi-line records
    cntrl_files = [f for f in os.listdir() if f.startswith(f"{cntrlFile}.")]
    for nextf in cntrl_files:
        try:
            with open(nextf, 'r') as cn:
                for line in cn:
                    if "REACH" in line:
                        fields = line.split()
                        reachNum = int(fields[4])
                        # Check for max/min
                        rmin = min(reachNum, rmin)
                        rmax = max(reachNum, rmax)
                        # Open output file
                        reach_filename = f"{reachFile}.{reachNum}"
                        with open(reach_filename, 'w') as re:
                            re.write(line)  # Reach line
                            # Read width, length, roughness, slope, C, Y1, Y2, Y3
                            for i in range(16):
                                buffer = next(cn)
                                re.write(buffer)

        except Exception as e:
            print(f"An error occurred: {e}")

    # Concatenate all reach files together
    os.system(f"mv {reachFile}.{rmin} {cntrlFile}")
    for i in range(rmin + 1, rmax + 1):
        os.system(f"cat {reachFile}.{i} >> {cntrlFile}")
    os.system(f"rm {reachFile}.*")


# sub merge parllel functions
def _private__sort_merge_file(self, mfile, hascomma):
    """
    Helper Function for merge_parallel_files
    """
    tempfile = f"{mfile}-xxx"
    sort_command = f"sort -g -k 1 {'--field-separator=,' if hascomma else ''} {mfile}.* > {tempfile}"

    try:
        os.system(sort_command)
        self._private__remove_extra_headers(mfile)
    except Exception as e:
        print(f"An error occurred: {e}")


def _private__remove_extra_headers(self, mfile):
    """
    Helper Function for merge_parallel_files
    """
    try:
        with open(mfile, 'w') as mf, open(f"{mfile}-xxx", 'r') as xf:
            firstid = True
            firstblank = True

            for line in xf:
                if "ID" in line:
                    if firstid:
                        mf.write(line)
                        firstid = False
                elif line == '\n':
                    if firstblank:
                        mf.write(line)
                        firstblank = False
                else:
                    mf.write(line)

        # Remove temporary file with extra headers
        os.remove(f"{mfile}-xxx")
    except Exception as e:
        print(f"An error occurred: {e}")

