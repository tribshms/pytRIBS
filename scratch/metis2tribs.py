import sys

def usage():
    print("Usage: {} <input file> <conn file> <output file>".format(sys.argv[0]))

if len(sys.argv) != 4:
    usage()
    sys.exit(1)

input_file = sys.argv[1]
conn_file = sys.argv[2]
output_file = sys.argv[3]

input_data = {}
partitions = []
partition_weights = {}

# Read input file to collect reaches per partition
with open(input_file, 'r') as input_handle:
    max_partition = 0
    reach_id = 0
    for line in input_handle:
        fields = line.strip().split()
        partition_id = int(fields[0])
        if partition_id > max_partition:
            max_partition = partition_id
        if partition_id not in input_data:
            input_data[partition_id] = partition_id
        partitions.append((partition_id, reach_id))
        reach_id += 1

# Read connection file for weights per node
with open(conn_file, 'r') as conn_handle:
    for line in conn_handle:
        fields = line.strip().split()
        partition_id = int(fields[0])
        partition_weights[partition_id] = int(fields[1])

with open(output_file, 'w') as output_handle:
    partition_count = len(input_data)
    print("Actual number of partitions =", partition_count)
    print("Maximum partition =", max_partition)
    print("Number of reaches =", reach_id)

    # Write the tRIBS partition file
    for i in range(max_partition + 1):
        total_weight = 0
        for partition_id, reach_id in partitions:
            if partition_id == i:
                output_handle.write(f"{i} {reach_id}\n")
                total_weight += partition_weights.get(reach_id, 0)
        print(f"Partition {i} weight = {total_weight}")
