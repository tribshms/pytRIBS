import sys
import gzip
from datetime import datetime


def usage():
    print("Usage: {} <input file> <output file> <node cnt> <extra weight flag> <flux weight flag> <big small flag> <edge type>".format(sys.argv[0]))
    print("       The files can be .gz files, which are handled transparently.")

if len(sys.argv) != 7:
    usage()
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]
node_weight_flag = int(sys.argv[3])
extra_weight_flag = int(sys.argv[4])
flux_weight_flag = int(sys.argv[5])
big_weight_flag = int(sys.argv[6])
edge_type = int(sys.argv[7])

def open_file(file_name):
    if file_name.endswith('.gz'):
        return gzip.open(file_name, 'rt')
    else:
        return open(file_name, 'r')

with open_file(input_file) as input_file_handle, open_file(output_file) as output_file_handle:
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_file_handle.write(f"% METIS input file generated from {input_file} on {date}.\n")
    output_file_handle.write("%\n")

    line = input_file_handle.readline().strip()  # Skip header

    fields = input_file_handle.readline().strip().split()
    cnt = int(fields[0])  # Number of reaches

    # Initialize arrays for outside weights
    iweight = [0] * cnt
    fweight = [0] * cnt
    bweight = [0] * cnt

    lcnt = 0

    vweight = [0] * cnt
    headid = [0] * cnt
    outid = [0] * cnt
    edges = [[] for _ in range(cnt)]

    for i in range(cnt):
        line = input_file_handle.readline().strip()
        fields = line.split()
        id = int(fields[0])
        vweight[id] = int(fields[1])
        if vweight[id] > 2000:
            bweight[id] = 1
        headid[id] = int(fields[2])
        outid[id] = int(fields[3])
        ndown = int(fields[4])

        # Read in downstream vertices
        jstart = 5
        jend = jstart

        if ndown > 0:
            jend = ndown + jstart
            for j in range(jstart, jend):
                # Save edges if edge type not "flux only"
                if edge_type != 1:
                    edges[id].append(int(fields[j]))
                    edges[fields[j]].append(id)
                    lcnt += 1
                # Set as inside
                iweight[fields[j]] = 1

        # Read in flux vertices
        nflux = int(fields[jend])
        # Set as flux weight
        fweight[id] = nflux
        if nflux > 0:
            jstart = jend + 1
            jend = jstart + nflux
            for j in range(jstart, jend):
                # Check for duplicates before adding
                # Save edge if edge type is not "flow only"
                if edge_type > 0:
                    ine = 0
                    if edge_type == 2:
                        esize = len(edges[id])
                        for k in range(esize):
                            if edges[id][k] == int(fields[j]):
                                ine = 1
                    if ine != 1:
                        edges[id].append(int(fields[j]))
                        edges[fields[j]].append(id)
                        lcnt += 1

    # Write the number of vertices and edges.  Currently, only vertices are weighted.
    if edge_type == 1:
        lcnt /= 2

    output_file_handle.write("% Next line lists # vertices and # edges.\n")
    output_file_handle.write("{} {}\n".format(cnt, lcnt))

    # Add number of weights
    nweights = 0
    if node_weight_flag == 1:
        nweights += 1
    if extra_weight_flag == 1:
        nweights += 1
    if flux_weight_flag == 1:
        nweights += 1
    if big_weight_flag == 1:
        nweights += 1

    if nweights > 0:
        output_file_handle.write(" 10 {}\n".format(nweights))
    else:
        output_file_handle.write("\n")

    output_file_handle.write("%\n")
    output_file_handle.write("% Format of vertex/edge entries:\n")
    output_file_handle.write("%----------------------------------------------------------------------\n")
    output_file_handle.write("% % <index of current vertex> <reach id> <weight> <head node id> <outlet node id>\n")
    output_file_handle.write("% <vertex weight> <indices of neighbors of current vertex, separated by spaces>\n")
    output_file_handle.write("%----------------------------------------------------------------------\n")
    output_file_handle.write("%\n")

    # For each vertex, write out the adjacent vertices.  The i'th line contains all neighbors of vertex i.
    index = 1
    for i in range(cnt):
        output_file_handle.write("% {} {} {} {} {}\n".format(index, i, vweight[i], headid[i], outid[i])

        if node_weight_flag == 1:
            output_file_handle.write("{} ".format(vweight[i]))

        # Add extra inside/outside weight if requested
        if extra_weight_flag == 1:
            output_file_handle.write("{} ".format(iweight[i]))

        # Add extra flux weight if requested
        if flux_weight_flag == 1:
            output_file_handle.write("{} ".format(fweight[i]))

        # Add big weight flag if requested
        if big_weight_flag == 1:
            output_file_handle.write("{} ".format(bweight[i]))

        # Eliminate duplicates
        seen = dict()
        for item in edges[i]:
            seen[item] = seen.get(item, 0) + 1

        for unode in seen.keys():
            # Print the index number instead of label.
            jindex = unode + 1
            output_file_handle.write("{} ".format(jindex))

        index += 1
        output_file_handle.write("\n")
