#!/bin/bash

today_date=$(date +%F)

### updates from github repo
git pull

### make new dated directories in results and data
mkdir -p "results/$today_date"
mkdir -p "data/$today_date"

# enable git to keep track of runall and summarize scripts but ignores everything else in results
echo "*\n\!runall*\n\!summarize*" > results/$today_date/.gitignore

# ### EXAMPLE OF USING rsync to update data and result from data and results stored somewhere else

# ### pull any updates from dropbox
# source_directory="${HOME}/example"
# destination_directory="${HOME}/example"
# exclude_one="first file to exclude"
# exclude_two="second file to exclude"

# # Use rsync to synchronize changes from source to destination
# rsync -av --exclude "$exclude_one" --exclude "$exclude_two" "$source_directory/" "$destination_directory/"

# source_directory="${HOME}/example"
# destination_directory="${HOME}/example"
# exclude_one="first file to exclude"

# # Use rsync to synchronize changes from source to destination
# rsync -av --exclude "$exclude_one" "$source_directory/" "$destination_directory/"
